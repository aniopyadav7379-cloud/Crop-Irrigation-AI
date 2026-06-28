"""
tests/test_water_deficit.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for models/water_deficit.py
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pytest

from models.water_deficit import WaterDeficitModel, CROP_KC


pytestmark = pytest.mark.unit


class TestETcComputation:
    def test_etc_scales_with_kc(self):
        et0 = np.array([[5.0, 5.0]])
        crop_map = np.array([[0, 6]], dtype=np.uint8)  # Paddy(Kc=1.20) vs Fallow(Kc=0.40)
        etc = WaterDeficitModel.compute_etc(et0, crop_map)

        assert etc[0, 0] == pytest.approx(5.0 * CROP_KC[0])
        assert etc[0, 1] == pytest.approx(5.0 * CROP_KC[6])

    def test_etc_nodata_pixels_are_nan(self):
        et0 = np.array([[5.0]])
        crop_map = np.array([[255]], dtype=np.uint8)
        etc = WaterDeficitModel.compute_etc(et0, crop_map)
        assert np.isnan(etc[0, 0])


class TestWaterDeficitComputation:
    def test_deficit_is_zero_when_eta_exceeds_etc(self):
        etc = np.array([5.0])
        eta = np.array([8.0])
        deficit = WaterDeficitModel.compute_water_deficit(etc, eta)
        assert deficit[0] == 0.0

    def test_deficit_positive_when_etc_exceeds_eta(self):
        etc = np.array([10.0])
        eta = np.array([4.0])
        deficit = WaterDeficitModel.compute_water_deficit(etc, eta)
        assert deficit[0] == pytest.approx(6.0)

    def test_rainfall_reduces_deficit(self):
        etc = np.array([10.0])
        eta = np.array([4.0])
        deficit_no_rain = WaterDeficitModel.compute_water_deficit(etc, eta, rainfall_mm=0.0)
        deficit_with_rain = WaterDeficitModel.compute_water_deficit(etc, eta, rainfall_mm=3.0)
        assert deficit_with_rain[0] < deficit_no_rain[0]


class TestWaterDeficitModelTraining:
    def test_train_xgboost_produces_metrics(self, synthetic_deficit_data, tmp_path: Path):
        X, y = synthetic_deficit_data
        model = WaterDeficitModel(model_type="xgboost", model_path=tmp_path / "wd.pkl")
        metrics = model.train(X, y, cv_folds=3)

        assert "cv_mae_mean" in metrics
        assert "cv_r2_mean" in metrics
        assert metrics["cv_mae_mean"] >= 0

    def test_predict_returns_correct_shape(self, synthetic_deficit_data, tmp_path: Path):
        X, y = synthetic_deficit_data
        model = WaterDeficitModel(model_type="xgboost", model_path=tmp_path / "wd2.pkl")
        model.train(X, y, cv_folds=3)

        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))

    def test_predict_without_training_raises(self, tmp_path: Path):
        model = WaterDeficitModel(model_type="xgboost", model_path=tmp_path / "missing.pkl")
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 12), dtype=np.float32))

    def test_evaluate_returns_expected_keys(self, synthetic_deficit_data, tmp_path: Path):
        X, y = synthetic_deficit_data
        model = WaterDeficitModel(model_type="xgboost", model_path=tmp_path / "wd3.pkl")
        model.train(X, y, cv_folds=3)

        metrics = model.evaluate(X, y)
        assert set(metrics.keys()) == {"mae_mm", "rmse_mm", "r2"}
