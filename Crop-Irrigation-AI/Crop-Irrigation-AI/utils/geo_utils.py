"""
utils/geo_utils.py
─────────────────────────────────────────────────────────────────────────────
Shared geospatial helper functions used across preprocessing, modelling,
and dashboard modules.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from rasterio.crs import CRS
from shapely.geometry import mapping, box
from loguru import logger


# ── Reprojection ───────────────────────────────────────────────────────────────

def reproject_raster(
    src_path: Path,
    dst_path: Path,
    target_crs: str = "EPSG:32644",   # UTM Zone 44N — common for peninsular India
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """
    Reproject *src_path* raster to *target_crs* and write to *dst_path*.

    Returns
    -------
    dst_path : Path
        Path to the reprojected file.
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            driver="GTiff",
            compress="lzw",
        )

        with rasterio.open(dst_path, "w", **meta) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )

    logger.info(f"Reprojected {src_path.name} → {target_crs} → {dst_path.name}")
    return dst_path


# ── Clipping ───────────────────────────────────────────────────────────────────

def clip_raster_to_boundary(
    raster_path: Path,
    boundary_shp: Path,
    out_path: Path,
    all_touched: bool = False,
) -> Path:
    """
    Clip *raster_path* to the union of polygons in *boundary_shp*.

    Returns
    -------
    out_path : Path
    """
    boundary = gpd.read_file(boundary_shp)

    with rasterio.open(raster_path) as src:
        # Reproject boundary to raster CRS if needed
        if boundary.crs != src.crs:
            boundary = boundary.to_crs(src.crs)

        geoms = [mapping(geom) for geom in boundary.geometry]
        clipped, clipped_transform = rio_mask(
            src, geoms, crop=True, all_touched=all_touched
        )
        meta = src.meta.copy()
        meta.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            driver="GTiff",
            compress="lzw",
        )

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(clipped)

    logger.info(f"Clipped {raster_path.name} → {out_path.name}")
    return out_path


# ── Resampling ─────────────────────────────────────────────────────────────────

def resample_raster(
    src_path: Path,
    out_path: Path,
    target_res_m: float = 10.0,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """
    Resample *src_path* to *target_res_m* metre pixel spacing.
    """
    with rasterio.open(src_path) as src:
        native_res = src.res[0]
        scale = native_res / target_res_m
        new_width = int(src.width * scale)
        new_height = int(src.height * scale)

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resampling,
        )
        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height,
        )
        meta = src.meta.copy()
        meta.update(
            width=new_width,
            height=new_height,
            transform=new_transform,
            driver="GTiff",
            compress="lzw",
        )

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)

    logger.info(f"Resampled {src_path.name} to {target_res_m}m → {out_path.name}")
    return out_path


# ── Stacking ───────────────────────────────────────────────────────────────────

def stack_bands(
    band_paths: list[Path],
    out_path: Path,
    band_names: Optional[list[str]] = None,
) -> Path:
    """
    Stack multiple single-band GeoTIFFs into one multi-band file.

    Parameters
    ----------
    band_paths  : ordered list of single-band raster paths
    out_path    : destination multi-band GeoTIFF
    band_names  : optional list of names written to band descriptions
    """
    with rasterio.open(band_paths[0]) as ref:
        meta = ref.meta.copy()
        meta.update(count=len(band_paths), driver="GTiff", compress="lzw")

        with rasterio.open(out_path, "w", **meta) as dst:
            for i, bp in enumerate(band_paths, start=1):
                with rasterio.open(bp) as src:
                    dst.write(src.read(1), i)
                if band_names:
                    dst.update_tags(i, name=band_names[i - 1])

    logger.info(
        f"Stacked {len(band_paths)} bands into {out_path.name}"
    )
    return out_path


# ── Raster I/O helpers ─────────────────────────────────────────────────────────

def read_band(raster_path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    """
    Read a single band from a GeoTIFF.

    Returns
    -------
    (array, profile) : ndarray (float32) and rasterio profile dict
    """
    with rasterio.open(raster_path) as src:
        arr = src.read(band).astype(np.float32)
        arr[arr == src.nodata] = np.nan
        return arr, src.profile


def write_band(
    array: np.ndarray,
    profile: dict,
    out_path: Path,
    nodata: float = -9999.0,
) -> Path:
    """
    Write a 2-D float array to a single-band GeoTIFF.
    """
    profile = profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        nodata=nodata,
        driver="GTiff",
        compress="lzw",
    )
    array = np.where(np.isnan(array), nodata, array).astype(np.float32)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)
    return out_path


# ── Bounding-box helpers ───────────────────────────────────────────────────────

def raster_bbox_as_geodataframe(raster_path: Path, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with the raster's bounding box in *crs*."""
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        bbox = box(*bounds)
        gdf = gpd.GeoDataFrame(geometry=[bbox], crs=src.crs)
    return gdf.to_crs(crs)
