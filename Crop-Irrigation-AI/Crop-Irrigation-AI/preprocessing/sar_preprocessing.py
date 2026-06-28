"""
preprocessing/sar_preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Sentinel-1 SAR preprocessing pipeline.

Steps
─────
1.  Apply orbit file
2.  Thermal noise removal
3.  Calibration → sigma0 backscatter
4.  Speckle filtering (Lee filter, 5×5)
5.  Terrain correction (Range-Doppler, SRTM 30m DEM)
6.  Conversion to dB
7.  Reproject, clip, resample
8.  Derive soil-moisture proxy from VV/VH polarisations
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from config import settings
from utils import (
    logger,
    reproject_raster,
    clip_raster_to_boundary,
    resample_raster,
    stack_bands,
    write_band,
    sar_soil_moisture_proxy,
)


# ── Lee speckle filter ────────────────────────────────────────────────────────

def lee_filter(image: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Adaptive Lee speckle filter implemented in pure NumPy.

    Parameters
    ----------
    image : linear power backscatter array (not dB)
    size  : filter window size (pixels)

    Returns
    -------
    filtered array (float32)
    """
    img = image.astype(np.float64)
    img_mean = uniform_filter(img, size)
    img_sq_mean = uniform_filter(img ** 2, size)
    img_variance = img_sq_mean - img_mean ** 2

    overall_variance = np.nanvar(img)
    weight = np.where(
        overall_variance > 0,
        img_variance / (img_variance + overall_variance),
        0.0,
    )
    filtered = img_mean + weight * (img - img_mean)
    return filtered.astype(np.float32)


# ── dB conversion helpers ──────────────────────────────────────────────────────

def linear_to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear power backscatter to decibels (dB)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(linear > 0, linear, np.nan))
    return db.astype(np.float32)


def db_to_linear(db: np.ndarray) -> np.ndarray:
    return (10.0 ** (db / 10.0)).astype(np.float32)


# ── Cross-pol ratio ────────────────────────────────────────────────────────────

def vv_vh_ratio(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """
    VV/VH ratio in dB — useful for differentiating surface types.
    Higher ratio → smoother surface (bare soil).
    """
    return (vv_db - vh_db).astype(np.float32)


# ── RVI — Radar Vegetation Index ───────────────────────────────────────────────

def radar_vegetation_index(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Sentinel-1 Radar Vegetation Index (Kim et al. 2012).
    RVI = (4 * VH) / (VV + VH)    (linear power)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rvi = np.where((vv + vh) > 0, (4 * vh) / (vv + vh), np.nan)
    return rvi.astype(np.float32)


# ── Main Preprocessor ──────────────────────────────────────────────────────────

class SARPreprocessor:
    """
    Sentinel-1 GRD/SLC preprocessing.

    Note
    ────
    Graph-Builder XML execution (apply-orbit-file → thermal-noise-removal →
    calibration → terrain-correction) requires ESA SNAP installed on the system
    and reachable via *gpt* CLI.  If SNAP is unavailable the class falls back to
    loading pre-processed sigma0 GeoTIFFs from *sat_dir*.
    """

    SNAP_GPT = "gpt"   # override if gpt is not on $PATH

    def __init__(
        self,
        sat_dir: Path | None = None,
        boundary_shp: Path | None = None,
        target_crs: str = "EPSG:32644",
        target_res_m: float = 10.0,
        filter_size: int = 5,
    ):
        self.sat_dir = sat_dir or settings.satellite_dir
        self.boundary_shp = boundary_shp or settings.boundary_shp
        self.target_crs = target_crs
        self.target_res_m = target_res_m
        self.filter_size = filter_size

        self.out_dir = self.sat_dir / "processed_sar"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── SNAP graph execution ───────────────────────────────────────────────────

    def _run_snap_graph(self, graph_xml: Path, product: Path, out: Path) -> bool:
        """
        Execute a SNAP Graph Builder XML via the gpt command-line tool.
        Returns True on success, False if SNAP is not available.
        """
        import shutil
        import subprocess

        if not shutil.which(self.SNAP_GPT):
            logger.warning("SNAP gpt not found — skipping graph execution.")
            return False

        cmd = [
            self.SNAP_GPT, str(graph_xml),
            f"-Pinput={product}",
            f"-Poutput={out}",
            "-q", "4",          # 4 CPU threads
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"SNAP error: {result.stderr}")
            return False
        return True

    # ── Speckle filter ─────────────────────────────────────────────────────────

    def apply_speckle_filter(self, sigma0_path: Path, out_path: Path) -> Path:
        with rasterio.open(sigma0_path) as src:
            profile = src.profile
            bands = src.read().astype(np.float32)

        filtered = np.stack(
            [lee_filter(bands[i], self.filter_size) for i in range(bands.shape[0])]
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filtered)

        logger.info(f"Speckle filter applied → {out_path.name}")
        return out_path

    # ── Convert to dB ─────────────────────────────────────────────────────────

    def to_db(self, sigma0_path: Path, out_path: Path) -> Path:
        with rasterio.open(sigma0_path) as src:
            profile = src.profile.copy()
            profile.update(dtype="float32", nodata=-9999)
            data = src.read().astype(np.float32)

        db_data = np.where(data > 0, linear_to_db(data), -9999.0)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(db_data)

        logger.info(f"Converted to dB → {out_path.name}")
        return out_path

    # ── Soil moisture ──────────────────────────────────────────────────────────

    def derive_soil_moisture(
        self,
        vv_db_path: Path,
        vh_db_path: Path,
        out_dir: Path,
        scene_id: str,
    ) -> dict[str, Path]:
        """
        Compute SAR-derived products:
          - VV/VH ratio
          - Radar Vegetation Index (RVI)
          - Soil-moisture proxy (0–1)
        """
        with rasterio.open(vv_db_path) as s:
            vv_db = s.read(1).astype(np.float32)
            profile = s.profile

        with rasterio.open(vh_db_path) as s:
            vh_db = s.read(1).astype(np.float32)

        vv_lin = db_to_linear(vv_db)
        vh_lin = db_to_linear(vh_db)

        results = {
            "VV_VH_ratio": vv_vh_ratio(vv_db, vh_db),
            "RVI":         radar_vegetation_index(vv_lin, vh_lin),
            "soil_moisture": sar_soil_moisture_proxy(vv_db, vh_db),
        }

        out_paths: dict[str, Path] = {}
        for name, arr in results.items():
            p = out_dir / f"{scene_id}_{name}.tif"
            write_band(arr, profile, p)
            out_paths[name] = p
            logger.debug(f"  Wrote SAR product {name} → {p.name}")

        return out_paths

    # ── Master pipeline ────────────────────────────────────────────────────────

    def run(self, product_path: Path) -> Path:
        """
        Execute the full SAR preprocessing pipeline.

        Parameters
        ----------
        product_path : Path to a Sentinel-1 .zip, .SAFE, or pre-processed
                       sigma0 GeoTIFF (when SNAP is unavailable).

        Returns
        -------
        Path to the final multi-band GeoTIFF (VV_dB, VH_dB, RVI, SM).
        """
        logger.info(f"Starting SAR preprocessing: {product_path.name}")
        scene_id = product_path.stem
        scene_dir = self.out_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Attempt SNAP-based calibration + terrain correction
        snap_graph = Path(__file__).parent / "snap_graph_s1_sigma0.xml"
        calibrated = scene_dir / f"{scene_id}_sigma0.tif"
        snap_ok = False

        if snap_graph.exists():
            snap_ok = self._run_snap_graph(snap_graph, product_path, calibrated)

        if not snap_ok:
            # Fall back: look for a pre-existing sigma0 GeoTIFF
            fallback = list(self.sat_dir.glob(f"*{scene_id}*sigma0*.tif"))
            if fallback:
                calibrated = fallback[0]
                logger.warning(f"Using pre-processed sigma0: {calibrated.name}")
            else:
                raise FileNotFoundError(
                    f"No calibrated sigma0 product found for {scene_id}. "
                    "Install ESA SNAP or provide a pre-processed GeoTIFF."
                )

        # Speckle filter (lee 5×5)
        filtered = scene_dir / f"{scene_id}_sigma0_lee.tif"
        self.apply_speckle_filter(calibrated, filtered)

        # Convert to dB
        db_path = scene_dir / f"{scene_id}_sigma0_dB.tif"
        self.to_db(filtered, db_path)

        # Reproject
        reproj = scene_dir / f"{scene_id}_sigma0_dB_reproj.tif"
        reproject_raster(db_path, reproj, self.target_crs)

        # Resample
        resampled = scene_dir / f"{scene_id}_sigma0_dB_{self.target_res_m:.0f}m.tif"
        resample_raster(reproj, resampled, self.target_res_m)

        # Clip to boundary
        clipped = scene_dir / f"{scene_id}_sigma0_dB_clipped.tif"
        clip_raster_to_boundary(resampled, self.boundary_shp, clipped)

        # Split into VV and VH (band 1 = VV, band 2 = VH)
        def _extract_band(src_path: Path, band: int, out_path: Path) -> Path:
            with rasterio.open(src_path) as src:
                profile = src.profile.copy()
                profile.update(count=1)
                data = src.read(band).astype(np.float32)
            write_band(data, profile, out_path)
            return out_path

        vv_path = scene_dir / f"{scene_id}_VV_dB.tif"
        vh_path = scene_dir / f"{scene_id}_VH_dB.tif"
        _extract_band(clipped, 1, vv_path)
        _extract_band(clipped, 2, vh_path)

        # Derive soil moisture + RVI
        sar_products = self.derive_soil_moisture(vv_path, vh_path, scene_dir, scene_id)

        # Final stack: VV, VH, RVI, SM
        final_stack = scene_dir / f"{scene_id}_SAR_stack.tif"
        stack_bands(
            [vv_path, vh_path, sar_products["RVI"], sar_products["soil_moisture"]],
            final_stack,
            band_names=["VV_dB", "VH_dB", "RVI", "SoilMoisture"],
        )

        logger.success(f"SAR preprocessing complete → {final_stack}")
        return final_stack
