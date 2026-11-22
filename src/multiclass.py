"""Data contract and optional trainer for four-class Vietnamese SMS research.

The existing binary labels are deliberately not converted here.  A four-class
model is only trainable after humans provide data with the explicit schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import normalize_message


MULTICLASS_LABELS = {
    "LEGITIMATE": 0,
    "PROMOTION": 1,
    "PHISHING": 2,
    "FINANCIAL_SCAM": 3,
}
REQUIRED_COLUMNS = {"message", "label", "annotation_status"}
VALID_STATUSES = {"certain", "uncertain"}


def validate_multiclass_frame(frame: pd.DataFrame, require_all_classes=False, min_per_class=1) -> dict:
    errors = []
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        errors.append(f"Thiếu cột: {sorted(missing)}")
        return {"valid": False, "ready_for_training": False, "errors": errors, "class_counts": {}}
    data = frame.copy()
    data["message"] = data["message"].fillna("").astype(str)
    data["label"] = data["label"].fillna("").astype(str).str.strip()
    data["annotation_status"] = data["annotation_status"].fillna("").astype(str).str.strip().str.lower()
    if (data["message"].str.strip() == "").any():
        errors.append("Có message rỗng.")
    invalid_labels = sorted(set(data.loc[data["label"] != "", "label"]) - set(MULTICLASS_LABELS))
    if invalid_labels:
        errors.append(f"Label không thuộc schema bốn lớp: {invalid_labels}")
    invalid_status = sorted(set(data["annotation_status"]) - VALID_STATUSES)
    if invalid_status:
        errors.append(f"annotation_status không hợp lệ: {invalid_status}")
    certain_without_label = (data["annotation_status"] == "certain") & (data["label"] == "")
    if certain_without_label.any():
        errors.append("Mẫu certain phải có label.")
    uncertain_with_label = (data["annotation_status"] == "uncertain") & (data["label"] != "")
    if uncertain_with_label.any():
        errors.append("Mẫu uncertain không được tự gán label dùng cho train.")
    trainable = data[(data["annotation_status"] == "certain") & (data["label"] != "")]
    counts = trainable["label"].value_counts().reindex(MULTICLASS_LABELS, fill_value=0).astype(int).to_dict()
    all_classes = all(value >= min_per_class for value in counts.values())
    if require_all_classes and not all_classes:
        errors.append(f"Chưa đủ ít nhất {min_per_class} mẫu certain cho cả bốn lớp.")
    return {
        "valid": not errors,
        "ready_for_training": not errors and all_classes,
        "errors": errors,
        "rows": len(data),
        "trainable_rows": len(trainable),
        "class_counts": counts,
        "classes_present": [label for label, count in counts.items() if count],
    }


def load_multiclass_csv(path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame, validate_multiclass_frame(frame)


def create_annotation_template(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["sample_id", "message", "sms_content", "label", "scam_type", "intent", "reason", "label_reason", "source", "source_name", "source_type", "annotator", "annotator_id", "verified", "annotation_status", "review_status", "confidence", "group_id", "synthetic", "notes"]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def append_annotation(path, message, label="", annotation_status="uncertain", confidence="", annotator="", notes=""):
    path = Path(path)
    if annotation_status not in VALID_STATUSES:
        raise ValueError("annotation_status phải là certain hoặc uncertain.")
    if label and label not in MULTICLASS_LABELS:
        raise ValueError(f"label phải là một trong {list(MULTICLASS_LABELS)}.")
    row = pd.DataFrame([{
        "message": message,
        "label": label if annotation_status == "certain" else "",
        "annotation_status": annotation_status,
        "confidence": confidence,
        "annotator": annotator,
        "notes": notes,
    }])
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig")
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False, encoding="utf-8-sig")
    return validate_multiclass_frame(row)


def train_multiclass(dataset_path, output_dir, seed=42):
    """Train an optional four-class baseline only on explicitly certain data."""
    frame, report = load_multiclass_csv(dataset_path)
    ready = validate_multiclass_frame(frame, require_all_classes=True, min_per_class=2)
    if not ready["ready_for_training"]:
        raise ValueError("Không huấn luyện đa lớp: " + "; ".join(ready["errors"] or ["thiếu dữ liệu bốn lớp"]))
    trainable = frame[(frame["annotation_status"] == "certain") & (frame["label"] != "")].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        trainable["message"].map(normalize_message),
        trainable["label"].map(MULTICLASS_LABELS),
        test_size=0.2,
        random_state=seed,
        stratify=trainable["label"],
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    started = perf_counter()
    model.fit(x_train, y_train)
    elapsed = perf_counter() - started
    predicted = model.predict(x_test)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "weighted_f1": float(f1_score(y_test, predicted, average="weighted")),
        "fit_seconds": elapsed,
        "classification_report": classification_report(y_test, predicted, labels=list(MULTICLASS_LABELS.values()), target_names=list(MULTICLASS_LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=list(MULTICLASS_LABELS.values())).tolist(),
        "label_mapping": MULTICLASS_LABELS,
        "dataset": str(dataset_path),
    }
    (output_dir / "multiclass_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    import joblib
    joblib.dump(model, output_dir / "multiclass_logreg.joblib")
    return metrics


def multiclass_readiness(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"available": False, "ready_for_training": False, "errors": ["Chưa có file annotation đa lớp."], "class_counts": {}}
    _, report = load_multiclass_csv(path)
    ready = validate_multiclass_frame(pd.read_csv(path, encoding="utf-8-sig"), require_all_classes=True, min_per_class=2)
    return {"available": True, **ready}
