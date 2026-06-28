"""
preprocessing/optical_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Sentinel-2 optical imagery preprocessing pipeline.

Steps
─────
1.  Download from Copernicus Hub (sentinelsat)
2.  Atmospheric correction check (sen2cor / already L2A)
3.  Band extraction and DN → reflectance scaling
4.  Cloud masking using SCL band
5.  Reproject to project CRS (UTM)
6.  Clip to command-area boundary
7.  Resample all bands to 10 m
8.  Stack into a single multi-band GeoTIFF
9.  Compute and write spectral index stack
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

from config import settings
from utils import (
    logger,
    reproject_raster,
    clip_raster_to_boundary,
    resample_raster,
    stack_bands,
    write_band,
    compute_all_indices,
)

# ── Band mapping: Sentinel-2 L2A filename suffixes ────────────────────────────
S2_BAND_FILES = {
    "B02": "_B02_10m.jp2",
    "B03": "_B03_10m.jp2",
    "B04": "_B04_10m.jp2",
    "B08": "_B08_10m.jp2",
    "B8A": "_B8A_20m.jp2",
    "B11": "_B11_20m.jp2",
    "B12": "_B12_20m.jp2",
    "SCL": "_SCL_20m.jp2",   # Scene Classification Layer
}

# SCL classes to mask out (cloud, shadow, snow, defective)
MASK_SCL_CLASSES = {0, 1, 2, 3, 8, 9, 10, 11}

REFLECTANCE_SCALE = 10_000.0   # ESA DN → [0, 1]


class OpticalPreprocessor:
    """End-to-end Sentinel-2 optical preprocessor."""

    def __init__(
        self,
        sat_dir: Path | None = None,
        boundary_shp: Path | None = None,
        target_crs: str = "EPSG:32644",
        target_res_m: float = 10.0,
    ):
        self.sat_dir = sat_dir or settings.satellite_dir
        self.boundary_shp = boundary_shp or settings.boundary_shp
        self.target_crs = target_crs
        self.target_res_m = target_res_m

        self.tmp_dir = self.sat_dir / "_tmp_optical"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ───────────────────────────────────────────────────────────────

    def download(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        cloud_cover_max: float | None = None,
        footprint_geojson: Path | None = None,
    ) -> list[Path]:
        """
        Query Copernicus Hub and download Sentinel-2 L2A products.

        Returns a list of downloaded .zip / .SAFE paths.
        """
        if not settings.satellite.sentinel_user:
            logger.warning("Sentinel credentials not set — skipping download.")
            return []

        api = SentinelAPI(
            settings.satellite.sentinel_user,
            settings.satellite.sentinel_pass,
            settings.satellite.sentinel_api_url,
        )

        end = end_date or date.today()
        start = start_date or (end - timedelta(days=settings.satellite.revisit_days))
        cloud_pct = cloud_cover_max or settings.satellite.cloud_cover_threshold

        footprint = (
            geojson_to_wkt(read_geojson(footprint_geojson))
            if footprint_geojson
            else None
        )

        products = api.query(
            area=footprint,
            date=(start.strftime("%Y%m%d"), end.strftime("%Y%m%d")),
            platformname="Sentinel-2",
            producttype="S2MSI2A",
            cloudcoverpercentage=(0, cloud_pct),
        )
        logger.info(f"Found {len(products)} Sentinel-2 products")

        downloaded = api.download_all(products, directory_path=self.sat_dir)
        paths = [Path(v["path"]) for v in downloaded.downloaded.values()]
        logger.success(f"Downloaded {len(paths)} products")
        return paths

    # ── Extract bands from SAFE archive ───────────────────────────────────────

    def extract_bands(self, safe_path: Path) -> dict[str, Path]:
        """
        Locate band files inside a .SAFE directory or zip archive.

        Returns
        -------
        dict  band_name → Path of the .jp2 file
        """
        if safe_path.suffix == ".zip":
            with zipfile.ZipFile(safe_path) as z:
                z.extractall(self.tmp_dir)
            safe_path = next(self.tmp_dir.glob("*.SAFE"))

        band_paths: dict[str, Path] = {}
        for band, suffix in S2_BAND_FILES.items():
            matches = list(safe_path.rglob(f"*{suffix}"))
            if matches:
                band_paths[band] = matches[0]
            else:
                logger.warning(f"Band {band} not found in {safe_path.name}")

        return band_paths

    # ── Cloud mask ─────────────────────────────────────────────────────────────

    def build_cloud_mask(self, scl_path: Path) -> np.ndarray:
        """
        Build a boolean mask from the Scene Classification Layer.
        True = valid pixel, False = masked (cloud, shadow, etc.)
        """
        with rasterio.open(scl_path) as src:
            scl = src.read(1).astype(np.uint8)
        mask = ~np.isin(scl, list(MASK_SCL_CLASSES))
        cloud_pct = 100.0 * (~mask).sum() / mask.size
        logger.info(f"Cloud / invalid pixels: {cloud_pct:.1f}%")
        return mask

    # ── Per-band normalisation ─────────────────────────────────────────────────

    def dn_to_reflectance(self, band_path: Path, out_path: Path) -> Path:
        """
        Scale ESA DN values to top-of-canopy reflectance ∈ [0, 1].
        """
        with rasterio.open(band_path) as src:
            data = src.read(1).astype(np.float32) / REFLECTANCE_SCALE
            profile = src.profile
        write_band(data, profile, out_path)
        return out_path

    # ── Master pipeline ────────────────────────────────────────────────────────

    def run(self, safe_path: Path) -> Path:
        """
        Execute the full preprocessing pipeline for one SAFE product.

        Returns
        -------
        stacked_path : Path
            Multi-band GeoTIFF (B02, B03, B04, B08, B8A, B11, B12) + index stack.
        """
        logger.info(f"Starting optical preprocessing: {safe_path.name}")
        scene_id = safe_path.stem
        out_dir = self.sat_dir / "processed" / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract
        band_paths = self.extract_bands(safe_path)
        if len(band_paths) < 7:
            raise RuntimeError(f"Missing bands in {safe_path.name}")

        # 2. Scale to reflectance
        ref_paths: dict[str, Path] = {}
        for band, path in band_paths.items():
            if band == "SCL":
                ref_paths["SCL"] = path
                continue
            rfl = out_dir / f"{scene_id}_{band}_ref.tif"
            self.dn_to_reflectance(path, rfl)
            ref_paths[band] = rfl

        # 3. Reproject all bands
        reproj_paths: dict[str, Path] = {}
        for band, path in ref_paths.items():
            rp = out_dir / f"{scene_id}_{band}_reproj.tif"
            reproject_raster(path, rp, self.target_crs)
            reproj_paths[band] = rp

        # 4. Resample to 10 m
        rs_paths: dict[str, Path] = {}
        for band, path in reproj_paths.items():
            rsp = out_dir / f"{scene_id}_{band}_{self.target_res_m:.0f}m.tif"
            resample_raster(path, rsp, self.target_res_m)
            rs_paths[band] = rsp

        # 5. Cloud masking
        cloud_mask = self.build_cloud_mask(rs_paths["SCL"])

        # 6. Clip to boundary
        clip_paths: dict[str, Path] = {}
        spectral_bands = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
        for band in spectral_bands:
            cp = out_dir / f"{scene_id}_{band}_clipped.tif"
            clip_raster_to_boundary(rs_paths[band], self.boundary_shp, cp)
            clip_paths[band] = cp

        # 7. Apply cloud mask to clipped bands
        for band, path in clip_paths.items():
            with rasterio.open(path, "r+") as src:
                data = src.read(1).astype(np.float32)
                # Resize mask to match clipped dimensions if needed
                if data.shape != cloud_mask.shape:
                    from skimage.transform import resize
                    cm = resize(cloud_mask.astype(float), data.shape, order=0) > 0.5
                else:
                    cm = cloud_mask
                data[~cm] = np.nan
                src.write(data.astype(np.float32), 1)

        # 8. Stack bands
        ordered_paths = [clip_paths[b] for b in spectral_bands]
        stacked = out_dir / f"{scene_id}_stack.tif"
        stack_bands(ordered_paths, stacked, band_names=spectral_bands)

        # 9. Compute spectral indices
        self._write_index_stack(clip_paths, out_dir, scene_id)

        # 10. Cleanup tmp
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

        logger.success(f"Optical preprocessing complete → {stacked}")
        return stacked

    # ── Index stack writer ─────────────────────────────────────────────────────

    def _write_index_stack(
        self,
        clip_paths: dict[str, Path],
        out_dir: Path,
        scene_id: str,
    ) -> None:
        """Read clipped bands, compute all indices, write individual GeoTIFFs."""

        def _read(band: str) -> np.ndarray:
            with rasterio.open(clip_paths[band]) as s:
                profile = s.profile
                return s.read(1).astype(np.float32), profile

        blue, profile  = _read("B02")
        green, _       = _read("B03")
        red, _         = _read("B04")
        nir, _         = _read("B08")
        nir_broad, _   = _read("B8A")
        swir1, _       = _read("B11")
        swir2, _       = _read("B12")

        indices = compute_all_indices(blue, green, red, nir, nir_broad, swir1, swir2)

        for name, arr in indices.items():
            out = out_dir / f"{scene_id}_{name}.tif"
            write_band(arr, profile, out)
            logger.debug(f"  Wrote index {name} → {out.name}")
