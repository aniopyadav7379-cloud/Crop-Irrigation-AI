"""
models/water_deficit.py
─────────────────────────────────────────────────────────────────────────────
Water-deficit regression model.

Predicts per-pixel water deficit (mm) by fusing:
  • Spectral indices  (NDVI, NDWI, NDMI, …)
  • SAR soil-moisture proxy
  • Weather covariates (ET₀, rainfall, temperature, humidity)
  • Crop coefficient (Kc) lookup from classification map

Model options: XGBoost (default), LightGBM, CatBoost.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import joblib
import rasterio
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import settings
from utils import logger, write_band

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ── Crop-coefficient lookup (FAO-56 midseason Kc values) ──────────────────────
#   Key = crop class integer (matches CropClassifier labels)
CROP_KC = {
    0: 1.20,  # Paddy/Rice
    1: 1.15,  # Maize
    2: 1.10,  # Soybean
    3: 1.25,  # Sugarcane
    4: 0.95,  # Groundnut
    5: 1.10,  # Cotton
    6: 0.40,  # Other/Fallow
}

ModelType = Literal["xgboost", "lightgbm"]


class WaterDeficitModel:
    """
    Regression model estimating water deficit (ETc - ETa) in mm per pixel.

    Water deficit = ETc (crop evapotranspiration demand) − ETa (actual ET)
    ETc = Kc × ET₀        (FAO-56 approach)
    ETa ≈ ET₀ × EF        (evaporative fraction from SEBS or METRIC)
    """

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        model_path: Path | None = None,
    ):
        self.model_type = model_type
        self.model_path = model_path or (
            settings.saved_models_dir / f"water_deficit_{model_type}.pkl"
        )
        self.model = None
        self._load_if_exists()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded water-deficit model from {self.model_path}")

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Water-deficit model saved → {self.model_path}")

    # ── ETc / ETa helpers ──────────────────────────────────────────────────────

    @staticmethod
    def compute_etc(
        et0: np.ndarray,
        crop_class_map: np.ndarray,
        kc_table: dict[int, float] | None = None,
    ) -> np.ndarray:
        """
        Compute crop evapotranspiration demand (ETc = Kc × ET₀).

        Parameters
        ----------
        et0            : scalar or 2-D array of reference ET₀ (mm/day)
        crop_class_map : 2-D uint8 map from CropClassifier
        kc_table       : optional override for CROP_KC defaults

        Returns
        -------
        etc : 2-D float32 array (mm/day)
        """
        kc_map = kc_table or CROP_KC
        kc_arr = np.vectorize(lambda c: kc_map.get(int(c), 0.0))(crop_class_map)
        kc_arr = kc_arr.astype(np.float32)
        etc = et0 * kc_arr
        etc[crop_class_map == 255] = np.nan   # nodata
        return etc.astype(np.float32)

    @staticmethod
    def compute_water_deficit(
        etc: np.ndarray,
        eta: np.ndarray,
        rainfall_mm: float = 0.0,
    ) -> np.ndarray:
        """
        Water deficit = max(ETc − ETa − rainfall, 0).
        Negative values (surplus) are clamped to zero.
        """
        deficit = np.maximum(etc - eta - rainfall_mm, 0.0)
        return deficit.astype(np.float32)

    # ── Feature engineering ────────────────────────────────────────────────────

    @staticmethod
    def build_features(
        index_arrays: dict[str, np.ndarray],
        soil_moisture: np.ndarray,
        weather_row: pd.Series,
        crop_class_flat: np.ndarray,
    ) -> np.ndarray:
        """
        Assemble a (n_pixels, n_features) matrix for regression.

        Parameters
        ----------
        index_arrays    : dict of 1-D index arrays (flattened)
        soil_moisture   : 1-D SAR soil-moisture proxy
        weather_row     : single-row Series with et0_mm, precip_mm, temp_mean, humidity_pct
        crop_class_flat : 1-D crop class array

        Returns
        -------
        X : (n_pixels, n_features) float32
        """
        expected_indices = ["NDVI", "NDWI", "NDMI", "MSI", "EVI", "LSWI"]
        cols = [
            index_arrays.get(name, np.full_like(soil_moisture, np.nan))
            for name in expected_indices
        ]
        cols.append(soil_moisture)

        # Kc from crop class
        kc = np.vectorize(lambda c: CROP_KC.get(int(c), 0.0))(crop_class_flat).astype(np.float32)
        cols.append(kc)

        # Weather scalars broadcast
        for key in ["et0_mm", "precip_mm", "temp_mean", "humidity_pct"]:
            val = float(weather_row.get(key, np.nan))
            cols.append(np.full(len(soil_moisture), val, dtype=np.float32))

        X = np.stack(cols, axis=1)
        return X.astype(np.float32)

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        params: dict | None = None,
    ) -> dict:
        """
        Train a gradient-boosted regression model with k-fold CV.

        Parameters
        ----------
        X        : (n_samples, n_features) float32
        y        : (n_samples,) float32 — water deficit in mm
        cv_folds : number of folds
        params   : model hyper-parameters (uses settings defaults if None)

        Returns
        -------
        metrics dict
        """
        _params = params or {
            "n_estimators":  settings.model.wdr_n_estimators,
            "max_depth":     settings.model.wdr_max_depth,
            "learning_rate": settings.model.wdr_learning_rate,
            "subsample":     0.8,
            "colsample_bytree": 0.8,
            "random_state":  42,
            "n_jobs":        -1,
        }

        if self.model_type == "xgboost":
            if not _HAS_XGB:
                raise ImportError("xgboost not installed")
            from xgboost import XGBRegressor
            self.model = XGBRegressor(**_params, tree_method="hist", verbosity=0)
        elif self.model_type == "lightgbm":
            if not _HAS_LGB:
                raise ImportError("lightgbm not installed")
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(**_params, verbose=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        logger.info(
            f"Training {self.model_type} water-deficit model  "
            f"|  X={X.shape}  |  {cv_folds}-fold CV"
        )

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_maes, cv_r2s = [], []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            self.model.fit(X[tr_idx], y[tr_idx])
            y_pred = self.model.predict(X[val_idx])
            cv_maes.append(mean_absolute_error(y[val_idx], y_pred))
            cv_r2s.append(r2_score(y[val_idx], y_pred))
            logger.debug(
                f"  Fold {fold}  MAE={cv_maes[-1]:.3f}mm  R²={cv_r2s[-1]:.4f}"
            )

        # Final fit on all data
        self.model.fit(X, y)
        self.save()

        metrics = {
            "cv_mae_mean": float(np.mean(cv_maes)),
            "cv_mae_std":  float(np.std(cv_maes)),
            "cv_r2_mean":  float(np.mean(cv_r2s)),
            "cv_r2_std":   float(np.std(cv_r2s)),
        }
        logger.info(
            f"CV MAE: {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f} mm  |  "
            f"R²: {metrics['cv_r2_mean']:.4f}"
        )
        return metrics

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        self._require_model()
        y_pred = self.model.predict(X_test)
        return {
            "mae_mm":  float(mean_absolute_error(y_test, y_pred)),
            "rmse_mm": float(mean_squared_error(y_test, y_pred) ** 0.5),
            "r2":      float(r2_score(y_test, y_pred)),
        }

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict water deficit (mm) for each pixel.

        Returns
        -------
        (n_pixels,) float32 — NaN rows return NaN
        """
        self._require_model()
        result = np.full(len(X), np.nan, dtype=np.float32)
        valid = ~np.any(np.isnan(X), axis=1)
        result[valid] = self.model.predict(X[valid]).astype(np.float32)
        return result

    def predict_to_raster(
        self,
        X: np.ndarray,
        raster_meta: dict,
        out_path: Path | None = None,
    ) -> np.ndarray:
        """
        Predict and write a water-deficit GeoTIFF.

        Returns
        -------
        deficit_map : (H, W) float32 in mm
        """
        deficit_flat = self.predict(X)
        h, w = raster_meta["height"], raster_meta["width"]
        deficit_map = deficit_flat.reshape(h, w)

        out_path = out_path or settings.irrigation_map_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_band(deficit_map, raster_meta["profile"], out_path)

        logger.success(
            f"Water-deficit map written → {out_path}  "
            f"(mean={np.nanmean(deficit_map):.2f} mm)"
        )
        return deficit_map

    # ── Guard ──────────────────────────────────────────────────────────────────

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model not trained or loaded. "
                "Call .train() or ensure model_path exists."
            )
