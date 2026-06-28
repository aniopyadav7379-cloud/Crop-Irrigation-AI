"""
tests/test_crop_classification.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for models/crop_classification.py
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pytest

from models.crop_classification import CropClassifier, CROP_LABELS


pytestmark = pytest.mark.unit


class TestCropClassifierTraining:
    def test_train_produces_fitted_model(self, synthetic_feature_matrix, tmp_path: Path):
        X, y = synthetic_feature_matrix
        clf = CropClassifier(model_path=tmp_path / "test_model.pkl")
        metrics = clf.train(X, y, cv_folds=3)

        assert clf.model is not None
        assert "train_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.5  # should fit synthetic separable data reasonably

    def test_train_saves_model_to_disk(self, synthetic_feature_matrix, tmp_path: Path):
        X, y = synthetic_feature_matrix
        model_path = tmp_path / "saved_model.pkl"
        clf = CropClassifier(model_path=model_path)
        clf.train(X, y, cv_folds=None)

        assert model_path.exists()

    def test_loading_saved_model(self, synthetic_feature_matrix, tmp_path: Path):
        X, y = synthetic_feature_matrix
        model_path = tmp_path / "reload_model.pkl"
        clf1 = CropClassifier(model_path=model_path)
        clf1.train(X, y, cv_folds=None)

        clf2 = CropClassifier(model_path=model_path)
        assert clf2.model is not None
        preds = clf2.predict(X)
        assert len(preds) == len(X)


class TestCropClassifierPrediction:
    def test_predict_returns_correct_shape(self, synthetic_feature_matrix, tmp_path: Path):
        X, y = synthetic_feature_matrix
        clf = CropClassifier(model_path=tmp_path / "m.pkl")
        clf.train(X, y, cv_folds=None)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert preds.dtype == np.uint8

    def test_predict_handles_nan_rows(self, synthetic_feature_matrix, tmp_path: Path):
        X, y = synthetic_feature_matrix
        clf = CropClassifier(model_path=tmp_path / "m.pkl")
        clf.train(X, y, cv_folds=None)

        X_with_nan = X.copy()
        X_with_nan[0, :] = np.nan
        preds = clf.predict(X_with_nan)

        assert preds[0] == 255  # nodata sentinel

    def test_predict_proba_sums_to_one(self, synthetic_feature_matrix, tmp_path: Path):
        X, y = synthetic_feature_matrix
        clf = CropClassifier(model_path=tmp_path / "m.pkl")
        clf.train(X, y, cv_folds=None)

        proba = clf.predict_proba(X[:5])
        row_sums = np.nansum(proba, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(5), rtol=1e-4)

    def test_predict_without_training_raises(self, tmp_path: Path):
        clf = CropClassifier(model_path=tmp_path / "nonexistent.pkl")
        with pytest.raises(RuntimeError):
            clf.predict(np.zeros((5, 20), dtype=np.float32))


class TestCropLabels:
    def test_all_labels_are_strings(self):
        assert all(isinstance(v, str) for v in CROP_LABELS.values())

    def test_seven_classes_defined(self):
        assert len(CROP_LABELS) == 7
