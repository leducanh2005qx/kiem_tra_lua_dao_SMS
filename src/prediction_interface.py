"""Unified prediction interface for optional Phase 4 production artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.phase4 import ID2LABEL, RejectPolicy, TemperatureScaler, unified_prediction


class ProductionArtifactError(RuntimeError):
    pass


class SklearnPredictor:
    def __init__(self, model, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.policy = RejectPolicy(**metadata["threshold"]) if metadata.get("threshold", {}).get("status") in {"validated", "production_candidate"} else None

    def predict(self, message: str) -> dict:
        predicted = int(self.model.predict([message])[0])
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba([message])[0], dtype=float)
            calibrated = self.metadata.get("calibration", {}).get("status") in {"fitted", "validated"}
            return unified_prediction(predicted, probabilities, "calibrated_probability" if calibrated else "posterior_probability", self.metadata["model_name"], self.metadata["model_version"], self.policy)
        if hasattr(self.model, "decision_function"):
            decision = np.asarray(self.model.decision_function([message])[0], dtype=float)
            return unified_prediction(predicted, decision, "decision_score", self.metadata["model_name"], self.metadata["model_version"], self.policy)
        raise ProductionArtifactError("Scikit-learn artifact không có predict_proba hoặc decision_function.")


class PhoBERTPredictor:
    def __init__(self, model, tokenizer, metadata: dict, segmenter=None, temperature_scaler: TemperatureScaler | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.segmenter = segmenter
        self.temperature_scaler = temperature_scaler
        self.policy = RejectPolicy(**metadata["threshold"]) if metadata.get("threshold", {}).get("status") in {"validated", "production_candidate"} else None

    def predict(self, message: str) -> dict:
        import torch
        text = self.segmenter.word_segment(message) if self.segmenter else message
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=int(self.metadata.get("preprocessing_config", {}).get("max_length", 256)))
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.detach().cpu().numpy()[0]
        if self.temperature_scaler:
            probabilities = self.temperature_scaler.predict_proba(logits.reshape(1, -1))[0]
            predicted = int(probabilities.argmax())
            return unified_prediction(predicted, probabilities, "calibrated_probability", self.metadata["model_name"], self.metadata["model_version"], self.policy)
        predicted = int(logits.argmax())
        return unified_prediction(predicted, logits, "logit_score", self.metadata["model_name"], self.metadata["model_version"], self.policy)


def load_production_registry(registry_path: str | Path):
    path = Path(registry_path)
    if not path.exists():
        raise ProductionArtifactError("Không tìm thấy model registry.")
    registry = json.loads(path.read_text(encoding="utf-8"))
    selected = registry.get("production_model")
    if not selected:
        raise ProductionArtifactError("Registry chưa đánh dấu production model hợp lệ.")
    if registry.get("selection_status") != "production":
        raise ProductionArtifactError("Registry không ở trạng thái production.")
    return registry, selected
