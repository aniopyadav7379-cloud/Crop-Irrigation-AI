"""
preprocessing/feature_extraction.py
─────────────────────────────────────────────────────────────────────────────
Feature extraction for ML model inputs.

Extracts per-pixel spectral features from the preprocessed optical and SAR
stacks, merges them with weather covariates, and exports flat NumPy arrays
or a labelled pandas DataFrame ready for scikit-learn / PyTorch.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.preprocessing import StandardScaler
import joblib

from config import settings
from utils import logger, read_band


# ── Optical feature names in stack band order ──────────────────────────────────
OPTICAL_BANDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
OPTICAL_INDICES = ["NDVI", "EVI", "SAVI", "LAI", "NDWI", "NDMI", "LSWI", "MSI", "BSI"]
SAR_BANDS = ["VV_dB", "VH_dB", "RVI", "SoilMoisture"]

ALL_FEATURES = OPTICAL_BANDS + OPTICAL_INDICES + SAR_BANDS


class FeatureExtractor:
    """
    Merges optical + SAR + weather data into a pixel-wise feature matrix.
    """

    def __init__(
        self,
        optical_stack: Path,
        sar_stack: Path,
        index_dir: Path,
        boundary_shp: Path | None = None,
        scaler_path: Path | None = None,
    ):
        self.optical_stack = optical_stack
        self.sar_stack = sar_stack
        self.index_dir = index_dir
        self.boundary_shp = boundary_shp or settings.boundary_shp
        self.scaler_path = scaler_path or (
            settings.saved_models_dir / "feature_scaler.pkl"
        )

    # ── Raster → flat array ────────────────────────────────────────────────────

    def _stack_to_matrix(self, stack_path: Path) -> np.ndarray:
        """
        Load a multi-band raster and reshape to (n_pixels, n_bands).
        NaN pixels are preserved.
        """
        with rasterio.open(stack_path) as src:
            data = src.read().astype(np.float32)  # (bands, H, W)
        bands, h, w = data.shape
        return data.reshape(bands, -1).T           # → (H*W, bands)

    def _load_index(self, name: str) -> Optional[np.ndarray]:
        """Load a single-band index GeoTIFF from index_dir, flattened."""
        candidates = list(self.index_dir.glob(f"*{name}.tif"))
        if not candidates:
            logger.warning(f"Index {name} not found in {self.index_dir}")
            return None
        arr, _ = read_band(candidates[0])
        return arr.ravel()

    # ── Reference pixels from ground truth ────────────────────────────────────

    def extract_training_samples(
        self,
        ground_truth_csv: Path | None = None,
        label_column: str = "crop_class",
        n_samples_per_class: int = 5000,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample pixel features at ground-truth locations from the raster stacks.

        Parameters
        ----------
        ground_truth_csv  : CSV with columns lat, lon, <label_column>
        label_column      : target column name
        n_samples_per_class : max samples per class (balancing)
        random_state      : reproducibility seed

        Returns
        -------
        X : (n_samples, n_features) float32
        y : (n_samples,) int32  — encoded class labels
        """
        csv_path = ground_truth_csv or settings.ground_truth_csv
        gdf = self._load_ground_truth(csv_path, label_column)

        # Rasterise label points to pixel coordinates
        with rasterio.open(self.optical_stack) as src:
            profile = src.profile
            transform = src.transform
            h, w = src.height, src.width

        labels_raster = rasterize(
            [(geom, val) for geom, val in zip(gdf.geometry, gdf["label_encoded"])],
            out_shape=(h, w),
            transform=transform,
            fill=255,
            dtype=np.uint8,
        )
        flat_labels = labels_raster.ravel()

        # Build full feature matrix
        X_full = self._build_feature_matrix()
        valid_mask = flat_labels != 255

        X = X_full[valid_mask]
        y = flat_labels[valid_mask]

        # Balance classes
        rng = np.random.default_rng(random_state)
        indices = []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            chosen = rng.choice(idx, size=min(len(idx), n_samples_per_class), replace=False)
            indices.append(chosen)
        indices = np.concatenate(indices)
        rng.shuffle(indices)

        logger.info(
            f"Extracted {len(indices)} training samples "
            f"across {len(np.unique(y))} classes"
        )
        return X[indices].astype(np.float32), y[indices].astype(np.int32)

    def _load_ground_truth(self, csv_path: Path, label_column: str) -> gpd.GeoDataFrame:
        df = pd.read_csv(csv_path)
        required = {"lat", "lon", label_column}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Ground truth CSV missing columns: {missing}")

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )

        # Encode labels
        classes = sorted(df[label_column].unique())
        mapping = {c: i for i, c in enumerate(classes)}
        gdf["label_encoded"] = gdf[label_column].map(mapping).astype(np.uint8)
        gdf.attrs["class_mapping"] = mapping

        # Reproject to raster CRS
        with rasterio.open(self.optical_stack) as src:
            gdf = gdf.to_crs(src.crs)

        return gdf

    # ── Full-scene feature matrix ──────────────────────────────────────────────

    def _build_feature_matrix(self) -> np.ndarray:
        """
        Assemble a (n_pixels, n_features) matrix combining:
        optical bands + spectral indices + SAR bands.
        """
        logger.info("Building feature matrix …")

        # Optical bands (7)
        X_opt = self._stack_to_matrix(self.optical_stack)  # (N, 7)

        # Spectral indices (9)
        index_cols = []
        for name in OPTICAL_INDICES:
            col = self._load_index(name)
            if col is None:
                col = np.full(X_opt.shape[0], np.nan, dtype=np.float32)
            index_cols.append(col[:, None])
        X_idx = np.concatenate(index_cols, axis=1)  # (N, 9)

        # SAR bands (4)
        X_sar = self._stack_to_matrix(self.sar_stack)   # (N, 4)

        X = np.concatenate([X_opt, X_idx, X_sar], axis=1)  # (N, 20)
        logger.info(f"Feature matrix shape: {X.shape}")
        return X.astype(np.float32)

    # ── Inference feature matrix ───────────────────────────────────────────────

    def extract_inference_features(self, scale: bool = True) -> np.ndarray:
        """
        Build and optionally scale the full-scene feature matrix for inference.
        Returns (n_pixels, n_features) float32 array.
        """
        X = self._build_feature_matrix()

        if scale:
            scaler = self._load_scaler()
            X = scaler.transform(X)

        return X

    # ── Scaler ────────────────────────────────────────────────────────────────

    def fit_save_scaler(self, X: np.ndarray) -> StandardScaler:
        """Fit a StandardScaler on training data and persist to disk."""
        scaler = StandardScaler()
        scaler.fit(X)
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, self.scaler_path)
        logger.info(f"Scaler saved → {self.scaler_path}")
        return scaler

    def _load_scaler(self) -> StandardScaler:
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {self.scaler_path}. "
                "Run training pipeline first."
            )
        return joblib.load(self.scaler_path)

    # ── Raster shape metadata ──────────────────────────────────────────────────

    def get_raster_meta(self) -> dict:
        """Return height, width, CRS, and transform from the optical stack."""
        with rasterio.open(self.optical_stack) as src:
            return {
                "height": src.height,
                "width": src.width,
                "crs": str(src.crs),
                "transform": src.transform,
                "profile": src.profile,
            }

    # ── Weather feature appender ───────────────────────────────────────────────

    @staticmethod
    def append_weather_features(
        X: np.ndarray,
        weather_df: pd.DataFrame,
        scene_date: str,
    ) -> np.ndarray:
        """
        Broadcast scalar weather features for a given date to every pixel.

        Parameters
        ----------
        X            : (n_pixels, n_spectral_features)
        weather_df   : DataFrame indexed by date with columns:
                       temp_mean, precip_mm, et0_mm, wind_speed, humidity_pct
        scene_date   : 'YYYY-MM-DD' string

        Returns
        -------
        X_extended : (n_pixels, n_spectral_features + 5)
        """
        if scene_date not in weather_df.index.astype(str):
            logger.warning(f"No weather data for {scene_date} — filling NaN")
            weather_vals = np.full((1, 5), np.nan, dtype=np.float32)
        else:
            row = weather_df.loc[scene_date, ["temp_mean", "precip_mm", "et0_mm",
                                               "wind_speed", "humidity_pct"]]
            weather_vals = row.values.astype(np.float32)[None, :]

        weather_broadcast = np.repeat(weather_vals, X.shape[0], axis=0)
        return np.concatenate([X, weather_broadcast], axis=1)


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from raster stacks")
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--sar",     type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--out",     type=Path, default=Path("features.npz"))
    args = parser.parse_args()

    extractor = FeatureExtractor(args.optical, args.sar, args.index_dir)
    X, y = extractor.extract_training_samples()
    np.savez_compressed(args.out, X=X, y=y)
    logger.success(f"Features saved → {args.out}  (X={X.shape}, y={y.shape})")
