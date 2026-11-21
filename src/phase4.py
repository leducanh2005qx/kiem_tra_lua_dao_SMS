"""Calibration, rejection, robustness and production-selection primitives.

No model is promoted automatically. Every production candidate needs a valid
artifact manifest, locked dataset evidence and evaluation results.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss


LABEL2ID = {"LEGITIMATE": 0, "PROMOTION": 1, "PHISHING": 2, "FINANCIAL_SCAM": 3}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
HIGH_RISK_IDS = {2, 3}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(y_true, probabilities, bins=10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (confidence > low) & (confidence <= high if index == bins - 1 else confidence < high)
        if mask.any():
            error += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return error


def multiclass_brier_score(y_true, probabilities) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    one_hot = np.eye(probabilities.shape[1])[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


class TemperatureScaler:
    """Temperature scaling fitted only on a calibration/validation split."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)
        self.fitted_on = ""

    def fit(self, logits, labels, split_name: str = "validation"):
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if logits.ndim != 2 or len(logits) != len(labels):
            raise ValueError("logits và labels không cùng số dòng.")
        if split_name.lower() in {"test", "gold_test", "robustness_test"}:
            raise ValueError("Không được fit temperature scaling bằng test/gold/robustness test.")
        candidates = np.geomspace(0.05, 20.0, 300)
        losses = [log_loss(labels, self._softmax(logits / value), labels=list(range(logits.shape[1]))) for value in candidates]
        self.temperature = float(candidates[int(np.argmin(losses))])
        self.fitted_on = split_name
        return self

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, logits):
        return self._softmax(np.asarray(logits, dtype=float) / self.temperature)

    def to_dict(self):
        return {"method": "temperature_scaling", "temperature": self.temperature, "fitted_on": self.fitted_on}


def calibration_metrics(y_true, probabilities) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(range(probabilities.shape[1])))),
    }


@dataclass
class RejectPolicy:
    confidence_threshold: float
    margin_threshold: float
    target_selective_risk: float
    validation_coverage: float
    validation_selective_risk: float
    validation_review_rate: float


def _selective_values(y_true, probabilities, confidence_threshold, margin_threshold):
    predictions = probabilities.argmax(axis=1)
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    review = (confidence < confidence_threshold) | (margin < margin_threshold)
    automatic = ~review
    coverage = float(automatic.mean()) if len(automatic) else 0.0
    risk = float((predictions[automatic] != np.asarray(y_true)[automatic]).mean()) if automatic.any() else 1.0
    return {"coverage": coverage, "selective_risk": risk, "review_rate": float(review.mean()) if len(review) else 0.0, "review_count": int(review.sum()), "predictions": predictions, "review": review}


def select_reject_policy(y_true, probabilities, target_selective_risk: float = 0.10) -> RejectPolicy:
    """Choose the highest validation coverage under a validation risk target."""
    candidates = []
    for confidence in np.arange(0.40, 0.96, 0.05):
        for margin in np.arange(0.00, 0.51, 0.05):
            values = _selective_values(y_true, probabilities, float(confidence), float(margin))
            candidates.append((values["selective_risk"] <= target_selective_risk, values["coverage"], -values["selective_risk"], confidence, margin, values))
    feasible = [row for row in candidates if row[0]] or candidates
    chosen = max(feasible, key=lambda row: (row[1], row[2], -row[3], -row[4]))
    values = chosen[5]
    return RejectPolicy(float(chosen[3]), float(chosen[4]), float(target_selective_risk), values["coverage"], values["selective_risk"], values["review_rate"])


def evaluate_reject_policy(y_true, probabilities, policy: RejectPolicy) -> dict:
    values = _selective_values(y_true, probabilities, policy.confidence_threshold, policy.margin_threshold)
    return {key: value for key, value in values.items() if key not in {"predictions", "review"}}


def status_for_prediction(predicted_id: int, needs_review: bool) -> str:
    if needs_review:
        return "REVIEW"
    return "HIGH_RISK" if int(predicted_id) in HIGH_RISK_IDS else "NORMAL_OR_PROMOTION"


def unified_prediction(predicted_id: int, scores, score_type: str, model_name: str, model_version: str, policy: RejectPolicy | None = None, explanation: list | None = None) -> dict:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    probabilities = scores if score_type == "calibrated_probability" else None
    if probabilities is not None:
        top = np.sort(probabilities)[-2:]
        needs_review = bool(policy and (top[-1] < policy.confidence_threshold or top[-1] - top[-2] < policy.margin_threshold))
        class_scores = {ID2LABEL[index]: float(probabilities[index]) for index in range(len(probabilities))}
    else:
        needs_review = True
        class_scores = {ID2LABEL[index]: float(scores[index]) for index in range(len(scores))}
    return {
        "predicted_class": ID2LABEL[int(predicted_id)],
        "class_scores": class_scores,
        "score_type": score_type,
        "status": status_for_prediction(int(predicted_id), needs_review),
        "needs_review": needs_review,
        "model_name": model_name,
        "model_version": model_version,
        "explanation": explanation or [],
    }


def _strip_diacritics(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFD", text) if not unicodedata.combining(char))


def robustness_variants(message: str) -> dict[str, str]:
    """Semantics-preserving stress variants; transformations are not labels."""
    text = str(message)
    return {
        "original": text,
        "without_diacritics": _strip_diacritics(text),
        "uppercase": text.upper(),
        "extra_whitespace": "  " + "   ".join(text.split()) + "  ",
        "special_char_spacing": "".join((f" {char} " if char in ":/!?.," else char) for char in text),
        "url_obfuscation": text.replace("https://", "hxxps://").replace("http://", "hxxp://"),
        "compact": " ".join(text.split()),
    }


def build_robustness_frame(test_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in test_frame.iterrows():
        for variant, message in robustness_variants(row["message"]).items():
            rows.append({"sample_id": row["sample_id"], "variant": variant, "message": message, "label_id": int(row["label_id"]), "label": row["label"]})
    return pd.DataFrame(rows)


REQUIRED_MANIFEST_FIELDS = {"model_name", "model_version", "dataset_version", "preprocessing_version", "training_config", "validation_metrics", "test_metrics", "calibration", "threshold"}


def validate_model_manifest(manifest: dict, artifact_path: str | Path) -> list[str]:
    errors = []
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        errors.append(f"missing_metadata:{sorted(missing)}")
    if manifest.get("label_mapping") != LABEL2ID:
        errors.append("label_mapping_mismatch")
    if manifest.get("calibration", {}).get("status") not in {"fitted", "validated"}:
        errors.append("calibration_not_complete")
    if manifest.get("threshold", {}).get("status") not in {"validated", "production_candidate"}:
        errors.append("threshold_not_complete")
    if not Path(artifact_path).exists():
        errors.append("artifact_missing")
    return errors


def discover_model_candidates(root: str | Path) -> tuple[list[dict], list[dict]]:
    """Discover only artifacts with complete metadata; retain exclusions for audit."""
    root = Path(root)
    candidates, excluded = [], []
    locations = [root / "models" / "classical_ml", root / "models" / "phobert_4class"]
    for location in locations:
        if not location.exists():
            excluded.append({"path": str(location), "reason": "artifact_directory_missing"})
            continue
        manifest_path = location / "model_manifest.json"
        if not manifest_path.exists():
            excluded.append({"path": str(location), "reason": "model_manifest_missing"})
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            excluded.append({"path": str(location), "reason": f"invalid_manifest:{exc}"})
            continue
        errors = validate_model_manifest(manifest, location)
        if errors:
            excluded.append({"path": str(location), "reason": ";".join(errors)})
        else:
            candidates.append({"artifact_path": str(location), "manifest_path": str(manifest_path), **manifest})
    return candidates, excluded


def create_registry(root: str | Path, registry_path: str | Path) -> dict:
    candidates, excluded = discover_model_candidates(root)
    registry = {
        "schema_version": "phase4_registry_v1",
        "label_mapping": LABEL2ID,
        "selection_status": "blocked" if not candidates else "awaiting_evaluation",
        "production_model": None,
        "models": candidates,
        "excluded_artifacts": excluded,
        "policy": {"production_requires": ["locked_dataset", "validation_and_test_metrics", "calibration", "robustness", "unit_and_integration_tests"]},
    }
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return registry
