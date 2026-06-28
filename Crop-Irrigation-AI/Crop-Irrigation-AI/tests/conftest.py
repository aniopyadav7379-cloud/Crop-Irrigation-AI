"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures: synthetic raster stacks, ground-truth samples,
and temp directories used across the test suite.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def tmp_raster_dir(tmp_path: Path) -> Path:
    d = tmp_path / "rasters"
    d.mkdir()
    return d


def _write_synthetic_raster(
    path: Path,
    n_bands: int,
    height: int = 32,
    width: int = 32,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    data = rng.uniform(0, 1, size=(n_bands, height, width)).astype(np.float32)
    transform = from_origin(79.0, 11.0, 0.0001, 0.0001)  # ~10m pixels near equator

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": n_bands,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


@pytest.fixture
def synthetic_optical_stack(tmp_raster_dir: Path) -> Path:
    """7-band synthetic optical stack: B02,B03,B04,B08,B8A,B11,B12."""
    path = tmp_raster_dir / "optical_stack.tif"
    return _write_synthetic_raster(path, n_bands=7, seed=1)


@pytest.fixture
def synthetic_sar_stack(tmp_raster_dir: Path) -> Path:
    """4-band synthetic SAR stack: VV_dB, VH_dB, RVI, SoilMoisture."""
    path = tmp_raster_dir / "sar_stack.tif"
    return _write_synthetic_raster(path, n_bands=4, seed=2)


@pytest.fixture
def synthetic_index_dir(tmp_raster_dir: Path) -> Path:
    """Directory containing individual single-band index GeoTIFFs."""
    idx_dir = tmp_raster_dir / "indices"
    idx_dir.mkdir()
    for i, name in enumerate(
        ["NDVI", "EVI", "SAVI", "LAI", "NDWI", "NDMI", "LSWI", "MSI", "BSI"]
    ):
        _write_synthetic_raster(idx_dir / f"scene_{name}.tif", n_bands=1, seed=10 + i)
    return idx_dir


@pytest.fixture
def synthetic_feature_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (X, y) for crop classifier unit tests — 7 classes, 20 features."""
    rng = np.random.default_rng(42)
    n_samples = 700
    n_features = 20
    n_classes = 7

    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    X = rng.normal(loc=y[:, None] * 0.5, scale=1.0, size=(len(y), n_features)).astype(np.float32)
    return X.astype(np.float32), y.astype(np.int32)


@pytest.fixture
def synthetic_deficit_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (X, y) for water-deficit regression unit tests."""
    rng = np.random.default_rng(7)
    n_samples = 500
    n_features = 12
    X = rng.uniform(0, 1, size=(n_samples, n_features)).astype(np.float32)
    y = (40 * X[:, 0] + 20 * (1 - X[:, 1]) + rng.normal(0, 3, n_samples)).astype(np.float32)
    y = np.clip(y, 0, None)
    return X, y


@pytest.fixture
def synthetic_weather_df() -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=20, freq="D")
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "temp_mean":    rng.uniform(26, 34, len(dates)),
        "precip_mm":    rng.choice([0, 0, 0, 5, 12, 20], len(dates)),
        "et0_mm":       rng.uniform(3.5, 6.0, len(dates)),
        "wind_speed":   rng.uniform(1.5, 4.5, len(dates)),
        "humidity_pct": rng.uniform(55, 85, len(dates)),
    }, index=dates.astype(str))
    return df


@pytest.fixture
def sample_ground_truth_csv(tmp_path: Path) -> Path:
    """Small synthetic ground-truth CSV for ingestion tests."""
    rng = np.random.default_rng(3)
    n = 60
    crops = ["Paddy", "Maize", "Soybean", "Groundnut"]
    df = pd.DataFrame({
        "lat": 10.85 + rng.uniform(-0.1, 0.1, n),
        "lon": 79.15 + rng.uniform(-0.1, 0.1, n),
        "crop_class": rng.choice(crops, n),
    })
    path = tmp_path / "ground_truth_sample.csv"
    df.to_csv(path, index=False)
    return path
