"""
models/crop_classification.py
─────────────────────────────────────────────────────────────────────────────
Crop type classification using a Random-Forest ensemble with SHAP
explainability and Optuna hyper-parameter tuning.

Supported crop classes (configurable via ground_truth.csv):
    0 – Paddy / Rice
    1 – Maize
    2 – Soybean
    3 – Sugarcane
    4 – Groundnut
    5 – Cotton
    6 – Other / Fallow
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import shap
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import rasterio

from config import settings
from utils import logger, write_band

optuna.logging.set_verbosity(optuna.logging.WARNING)


CROP_LABELS = {
    0: "Paddy/Rice",
    1: "Maize",
    2: "Soybean",
    3: "Sugarcane",
    4: "Groundnut",
    5: "Cotton",
    6: "Other/Fallow",
}


class CropClassifier:
    """
    Random-Forest crop type classifier with:
    - Stratified k-fold cross-validation
    - Optuna hyper-parameter optimisation
    - SHAP feature-importance analysis
    - Full-scene raster prediction
    """

    def __init__(
        self,
        model_path: Path | None = None,
        feature_names: list[str] | None = None,
        n_jobs: int = -1,
    ):
        self.model_path = model_path or (
            settings.saved_models_dir / "crop_classifier.pkl"
        )
        self.feature_names = feature_names
        self.n_jobs = n_jobs
        self.model: Optional[RandomForestClassifier] = None
        self._load_if_exists()

    # ── Model persistence ──────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded crop classifier from {self.model_path}")

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Crop classifier saved → {self.model_path}")

    # ── Hyper-parameter tuning ─────────────────────────────────────────────────

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        cv_folds: int = 5,
    ) -> dict:
        """
        Optuna-driven hyper-parameter search.
        Returns the best parameter dict.
        """
        logger.info(f"Starting Optuna tuning ({n_trials} trials, {cv_folds}-fold CV)")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth":        trial.suggest_int("max_depth", 5, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                "class_weight":     "balanced",
                "n_jobs":           self.n_jobs,
                "random_state":     settings.model.rf_random_state,
            }
            clf = RandomForestClassifier(**params)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                  random_state=settings.model.rf_random_state)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        logger.info(f"Best params: {best}  |  F1-macro: {study.best_value:.4f}")
        return best

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict | None = None,
        cv_folds: int | None = None,
    ) -> dict:
        """
        Train the Random-Forest classifier.

        Parameters
        ----------
        X       : (n_samples, n_features) float32
        y       : (n_samples,) int32 class labels
        params  : RF hyper-parameters (uses defaults from settings if None)
        cv_folds: number of cross-validation folds (None = no CV)

        Returns
        -------
        metrics dict with accuracy, macro-F1, per-class report
        """
        if params is None:
            params = {
                "n_estimators":     settings.model.rf_n_estimators,
                "max_depth":        settings.model.rf_max_depth,
                "min_samples_leaf": settings.model.rf_min_samples_leaf,
                "n_jobs":           self.n_jobs,
                "random_state":     settings.model.rf_random_state,
                "class_weight":     "balanced",
            }

        logger.info(f"Training RandomForest  |  X={X.shape}  |  classes={np.unique(y)}")
        self.model = RandomForestClassifier(**params)

        metrics: dict = {}

        if cv_folds:
            cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True,
                random_state=settings.model.rf_random_state
            )
            cv_scores = cross_val_score(
                self.model, X, y, cv=cv, scoring="f1_macro", n_jobs=self.n_jobs
            )
            metrics["cv_f1_macro_mean"] = float(cv_scores.mean())
            metrics["cv_f1_macro_std"]  = float(cv_scores.std())
            logger.info(
                f"CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )

        self.model.fit(X, y)

        # Train-set metrics
        y_pred = self.model.predict(X)
        metrics["train_accuracy"] = float(accuracy_score(y, y_pred))
        metrics["classification_report"] = classification_report(
            y, y_pred,
            target_names=[CROP_LABELS.get(i, str(i)) for i in np.unique(y)],
        )
        logger.info(f"Train accuracy: {metrics['train_accuracy']:.4f}")
        self.save()
        return metrics

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Full evaluation on held-out test set."""
        self._require_model()
        y_pred = self.model.predict(X_test)
        return {
            "accuracy":              float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=[CROP_LABELS.get(i, str(i)) for i in np.unique(y_test)],
            ),
            "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        }

    # ── SHAP explainability ────────────────────────────────────────────────────

    def explain(
        self,
        X_sample: np.ndarray,
        n_samples: int = 500,
        out_dir: Path | None = None,
    ) -> pd.DataFrame:
        """
        Compute SHAP values for a sample of pixels.

        Returns
        -------
        DataFrame with mean |SHAP| per feature, sorted descending.
        """
        self._require_model()
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_sample), size=min(n_samples, len(X_sample)), replace=False)
        X_sub = X_sample[idx]

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sub)   # list of arrays (one per class)

        # Mean absolute SHAP across classes and samples
        mean_shap = np.mean(np.abs(np.stack(shap_values)), axis=(0, 1))

        names = self.feature_names or [f"f{i}" for i in range(X_sub.shape[1])]
        df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_shap})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / "shap_feature_importance.csv", index=False)
            logger.info(f"SHAP importances saved → {out_dir}")

        return df

    # ── Inference / map generation ─────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict crop classes for a pixel feature matrix.

        Parameters
        ----------
        X : (n_pixels, n_features) float32 — may contain NaN rows

        Returns
        -------
        y_pred : (n_pixels,) int32 — NaN rows get label 255 (nodata)
        """
        self._require_model()
        y_pred = np.full(len(X), 255, dtype=np.uint8)
        valid = ~np.any(np.isnan(X), axis=1)
        y_pred[valid] = self.model.predict(X[valid]).astype(np.uint8)
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (n_pixels, n_classes)."""
        self._require_model()
        proba = np.full((len(X), len(CROP_LABELS)), np.nan, dtype=np.float32)
        valid = ~np.any(np.isnan(X), axis=1)
        proba[valid] = self.model.predict_proba(X[valid]).astype(np.float32)
        return proba

    def predict_to_raster(
        self,
        X: np.ndarray,
        raster_meta: dict,
        out_path: Path | None = None,
    ) -> np.ndarray:
        """
        Run prediction and write a crop classification map GeoTIFF.

        Parameters
        ----------
        X           : (n_pixels, n_features)
        raster_meta : dict with height, width, profile (from FeatureExtractor)
        out_path    : destination GeoTIFF path

        Returns
        -------
        2-D classification map array (H, W)
        """
        y_pred = self.predict(X)
        h, w = raster_meta["height"], raster_meta["width"]
        crop_map = y_pred.reshape(h, w)

        if out_path is None:
            out_path = settings.crop_map_path

        profile = raster_meta["profile"].copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(crop_map, 1)
            dst.update_tags(
                classes=str(CROP_LABELS),
                nodata_value="255",
            )

        logger.success(f"Crop map written → {out_path}")
        return crop_map

    # ── Internal guard ─────────────────────────────────────────────────────────

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Model not trained or loaded. "
                "Call .train() or ensure model_path exists."
            )
