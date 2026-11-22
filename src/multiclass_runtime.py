"""Runtime loader and reject-option predictor for the verified 4-class models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.dataset_engineering import FOUR_CLASS_LABELS
from src.preprocess import validate_message


LABELS = list(FOUR_CLASS_LABELS)
ID_TO_LABEL = {value: key for key, value in FOUR_CLASS_LABELS.items()}


def load_multiclass_models(models_root: str | Path):
    root = Path(models_root) / "multiclass"
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return {}, {"status": "blocked", "reason": "multiclass_metadata_missing"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("task_type") != "multiclass" or metadata.get("num_classes") != 4 or metadata.get("label_mapping") != FOUR_CLASS_LABELS:
        return {}, {"status": "blocked", "reason": "multiclass_metadata_invalid", "metadata": metadata}
    loaded = {}
    missing = []
    for name, config in metadata.get("models", {}).items():
        artifact = root / config.get("artifact", "")
        if not artifact.exists():
            missing.append(str(artifact))
            continue
        loaded[name] = joblib.load(artifact)
    if missing or set(loaded) != {"nb_baseline", "nb_optimized", "svm"}:
        return {}, {"status": "blocked", "reason": "multiclass_artifact_missing", "missing": missing, "metadata": metadata}
    return loaded, {"status": "ready", **metadata}


def _evidence(model, message, predicted_id, n=8):
    """Return observed TF-IDF features when the classifier exposes weights."""
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["clf"]
        weights = None
        if hasattr(classifier, "coef_"):
            weights = np.asarray(classifier.coef_)[int(predicted_id)]
        elif hasattr(classifier, "feature_log_prob_"):
            weights = np.asarray(classifier.feature_log_prob_)[int(predicted_id)]
        elif hasattr(classifier, "calibrated_classifiers_"):
            estimators = [item.estimator for item in classifier.calibrated_classifiers_ if hasattr(item, "estimator") and hasattr(item.estimator, "coef_")]
            if estimators:
                weights = np.mean([np.asarray(item.coef_)[int(predicted_id)] for item in estimators], axis=0)
        if weights is None:
            return []
        values = vectorizer.transform([message]).tocoo()
        names = vectorizer.get_feature_names_out()
        rows = sorted(((str(names[index]), float(value * weights[index])) for index, value in zip(values.col, values.data)), key=lambda item: item[1], reverse=True)
        return [{"feature": feature, "contribution": round(score, 6), "toward_label": ID_TO_LABEL[predicted_id], "method": "observed_tfidf_contribution"} for feature, score in rows[:n] if score > 0]
    except (AttributeError, IndexError, ValueError):
        return []


def predict_multiclass(model, metadata: dict, message: str, model_name: str | None = None) -> dict:
    clean = validate_message(message)
    probabilities = np.asarray(model.predict_proba([clean])[0], dtype=float)
    predicted_id = int(probabilities.argmax())
    ordered = np.sort(probabilities)
    confidence = float(ordered[-1])
    margin = float(ordered[-1] - ordered[-2])
    policy = metadata.get("models", {}).get(model_name or "", {}).get("threshold", {})
    # The default remains conservative and never treats an uncalibrated score
    # as confidence.
    threshold = policy or {"confidence_threshold": 0.70, "margin_threshold": 0.20, "status": "default_conservative"}
    needs_review = threshold.get("status") not in {"validated", "production_candidate"} or confidence < threshold.get("confidence_threshold", 0.70) or margin < threshold.get("margin_threshold", 0.20)
    label = "NEEDS_REVIEW" if needs_review else ID_TO_LABEL[predicted_id]
    return {
        "predicted_label": ID_TO_LABEL[predicted_id],
        "label": label,
        "class_scores": {ID_TO_LABEL[index]: float(value) for index, value in enumerate(probabilities)},
        "confidence": confidence if threshold.get("status") in {"validated", "production_candidate"} else None,
        "score_type": "calibrated_probability" if threshold.get("status") in {"validated", "production_candidate"} else "posterior_probability_unvalidated",
        "needs_review": bool(needs_review),
        "signals": _evidence(model, clean, predicted_id),
    }
