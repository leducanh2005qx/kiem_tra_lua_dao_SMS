import numpy as np
import pytest

from src.phase4 import RejectPolicy
from src.prediction_interface import ProductionArtifactError, SklearnPredictor, load_production_registry


class ProbabilityModel:
    def predict(self, messages):
        return np.array([2] * len(messages))

    def predict_proba(self, messages):
        return np.array([[0.05, 0.05, 0.80, 0.10] for _ in messages])


def test_sklearn_interface_distinguishes_uncalibrated_probability():
    metadata = {"model_name": "test", "model_version": "v1", "calibration": {"status": "pending"}, "threshold": {"status": "pending"}}
    result = SklearnPredictor(ProbabilityModel(), metadata).predict("tin nhắn")
    assert result["score_type"] == "posterior_probability"
    assert result["needs_review"] is True
    assert result["predicted_class"] == "PHISHING"


def test_empty_registry_cannot_be_loaded(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text('{"selection_status":"blocked","production_model":null}', encoding="utf-8")
    with pytest.raises(ProductionArtifactError):
        load_production_registry(path)
