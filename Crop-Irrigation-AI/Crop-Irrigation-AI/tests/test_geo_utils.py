"""
tests/test_geo_utils.py
─────────────────────────────────────────────────────────────────────────────
Unit / integration tests for utils/geo_utils.py raster helpers.
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from utils.geo_utils import (
    read_band,
    write_band,
    stack_bands,
    resample_raster,
    reproject_raster,
)


pytestmark = pytest.mark.integration


class TestReadWriteBand:
    def test_write_then_read_roundtrip(self, synthetic_optical_stack: Path, tmp_path: Path):
        arr, profile = read_band(synthetic_optical_stack, band=1)
        out_path = tmp_path / "single_band.tif"
        write_band(arr, profile, out_path)

        arr2, _ = read_band(out_path, band=1)
        np.testing.assert_allclose(arr, arr2, rtol=1e-5, equal_nan=True)

    def test_nan_handled_as_nodata(self, tmp_path: Path):
        from rasterio.transform import from_origin
        profile = {
            "driver": "GTiff", "height": 2, "width": 2, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326",
            "transform": from_origin(0, 0, 1, 1), "nodata": -9999.0,
        }
        arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        out = tmp_path / "nan_test.tif"
        write_band(arr, profile, out)

        with rasterio.open(out) as src:
            data = src.read(1)
            assert data[0, 1] == -9999.0


class TestStackBands:
    def test_stack_produces_correct_band_count(self, tmp_raster_dir: Path, tmp_path: Path):
        from rasterio.transform import from_origin
        profile = {
            "driver": "GTiff", "height": 4, "width": 4, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326",
            "transform": from_origin(0, 0, 1, 1), "nodata": -9999.0,
        }
        paths = []
        for i in range(3):
            p = tmp_raster_dir / f"band_{i}.tif"
            with rasterio.open(p, "w", **profile) as dst:
                dst.write(np.full((4, 4), float(i), dtype=np.float32), 1)
            paths.append(p)

        out = tmp_path / "stacked.tif"
        stack_bands(paths, out, band_names=["A", "B", "C"])

        with rasterio.open(out) as src:
            assert src.count == 3
            band0 = src.read(1)
            band2 = src.read(3)
            np.testing.assert_allclose(band0, np.zeros((4, 4)))
            np.testing.assert_allclose(band2, np.full((4, 4), 2.0))


class TestResample:
    def test_resample_changes_dimensions(self, synthetic_optical_stack: Path, tmp_path: Path):
        out = tmp_path / "resampled.tif"
        # Native res in this fixture ~0.0001 deg; resample coarser
        resample_raster(synthetic_optical_stack, out, target_res_m=0.0002)

        with rasterio.open(synthetic_optical_stack) as src_orig, \
             rasterio.open(out) as src_new:
            assert src_new.width <= src_orig.width
            assert src_new.height <= src_orig.height


class TestReproject:
    def test_reproject_changes_crs(self, synthetic_optical_stack: Path, tmp_path: Path):
        out = tmp_path / "reprojected.tif"
        reproject_raster(synthetic_optical_stack, out, target_crs="EPSG:32644")

        with rasterio.open(out) as src:
            assert src.crs.to_string() == "EPSG:32644"
