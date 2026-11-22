"""Runtime loader and reject-option predictor for hierarchical models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.hierarchical import LAYER1_LABELS, SCAM_TYPE_LABELS


def load_hierarchical_models(models_dir):
    model_dir = Path(models_dir) / "hierarchical"
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return {}, {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "ready":
        return {}, metadata
    loaded = {}
    for model_name, artifacts in metadata.get("models", {}).items():
        if all((model_dir / info["artifact"]).exists() for info in artifacts.values()):
            loaded[model_name] = {layer: joblib.load(model_dir / info["artifact"]) for layer, info in artifacts.items()}
    return loaded, metadata


def _score(model, message):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([message])[0]
        index = int(probabilities.argmax())
        return str(model.classes_[index]), float(probabilities[index]), "calibrated_probability"
    prediction = str(model.predict([message])[0])
    return prediction, 1.0, "model_prediction"


def predict_hierarchical(models, metadata, message, model_name="svm", threshold=0.60):
    if model_name not in models:
        raise ValueError("Chưa có mô hình phân cấp đã huấn luyện.")
    model_pair = models[model_name]
    layer1_label, layer1_confidence, score_type = _score(model_pair["layer1"], message)
    result = {
        "layer1_label": layer1_label,
        "layer1_confidence": layer1_confidence,
        "layer1_score_type": score_type,
        "layer2_label": "",
        "layer2_confidence": None,
        "intended_status": "NOT_SCAM" if layer1_label == "NOT_SCAM" else "SCAM",
        "needs_review": layer1_confidence < threshold,
    }
    if layer1_label == "SCAM" and not result["needs_review"]:
        layer2_label, layer2_confidence, layer2_score_type = _score(model_pair["layer2"], message)
        result.update({"layer2_label": layer2_label, "layer2_confidence": layer2_confidence, "layer2_score_type": layer2_score_type, "needs_review": layer2_confidence < threshold})
    if result["needs_review"]:
        result["status"] = "NEEDS_REVIEW"
    elif layer1_label == "NOT_SCAM":
        result["status"] = "NOT_SCAM"
    else:
        result["status"] = result["layer2_label"] or "NEEDS_REVIEW"
    return result
