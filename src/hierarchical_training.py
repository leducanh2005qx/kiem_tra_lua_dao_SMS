"""Strict trainer for the hierarchical NOT_SCAM/SCAM -> scam-type models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.dataset_engineering import normalize_text
from src.hierarchical import LAYER1_LABELS, SCAM_TYPE_LABELS, VERIFIED_STATUSES, validate_hierarchical_frame


def _verified_frame(queue_path) -> pd.DataFrame:
    frame = pd.read_csv(queue_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in ("message", "layer1_label", "layer2_label", "verified", "review_status", "group_id"):
        if column not in frame.columns:
            frame[column] = ""
    frame["message"] = frame["message"].map(normalize_text)
    verified = frame["verified"].isin({"1", "True", "true", "VERIFIED"}) & frame["review_status"].isin(VERIFIED_STATUSES)
    frame = frame[verified & frame["layer1_label"].isin(LAYER1_LABELS)].copy()
    frame["layer1_id"] = frame["layer1_label"].map(LAYER1_LABELS)
    frame["layer2_id"] = frame["layer2_label"].map(SCAM_TYPE_LABELS)
    return frame


def audit_hierarchical_queue(queue_path, min_per_class=20) -> dict:
    path = Path(queue_path)
    if not path.exists():
        return {"ready_for_training": False, "errors": ["Chưa có hàng đợi gắn nhãn."], "layer1_counts": {}, "layer2_counts": {}}
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    report = validate_hierarchical_frame(frame, min_per_class=min_per_class)
    report["queue_rows"] = len(frame)
    return report


def _split(frame: pd.DataFrame, label_column: str, seed: int):
    if frame[label_column].value_counts().min() < 3:
        raise ValueError(f"Không thể tách dữ liệu {label_column}: mỗi lớp cần ít nhất 3 mẫu.")
    train_valid, test = train_test_split(frame, test_size=0.2, random_state=seed, stratify=frame[label_column])
    train, validation = train_test_split(train_valid, test_size=0.25, random_state=seed, stratify=train_valid[label_column])
    return train, validation, test


def _model_specs():
    return {
        "nb_baseline": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 1), strip_accents="unicode", sublinear_tf=True)), ("clf", MultinomialNB(alpha=1.0))]),
        "nb_optimized": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)), ("clf", ComplementNB(alpha=0.5))]),
        "svm": Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), strip_accents="unicode", sublinear_tf=True)), ("clf", CalibratedClassifierCV(estimator=LinearSVC(C=1.0, class_weight="balanced"), method="sigmoid", cv=3))]),
    }


def _evaluate(model, frame, label_column, labels):
    actual = frame[label_column].to_numpy()
    predicted = model.predict(frame["message"].tolist())
    names = list(labels)
    return {
        "accuracy": float((actual == predicted).mean()),
        "macro_precision": float(precision_score(actual, predicted, labels=names, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, labels=names, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, labels=names, average="macro", zero_division=0)),
        "classification_report": classification_report(actual, predicted, labels=names, target_names=names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=names).tolist(),
    }


def train_hierarchical_models(queue_path, models_dir, results_dir, report_path, seed=42, min_per_class=20):
    queue_path, models_dir, results_dir, report_path = map(Path, [queue_path, models_dir, results_dir, report_path])
    audit = audit_hierarchical_queue(queue_path, min_per_class=min_per_class)
    if not audit["ready_for_training"]:
        payload = {"status": "blocked", "task_type": "hierarchical", "audit": audit, "models_trained": []}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("# Huấn luyện phân cấp\n\n**BLOCKED – chưa đủ dữ liệu xác minh cho hai lớp.**\n\n" + "\n".join(f"- {error}" for error in audit.get("errors", [])) + "\n", encoding="utf-8")
        return payload

    frame = _verified_frame(queue_path)
    layer1 = frame[frame["layer1_label"].isin(LAYER1_LABELS)].copy()
    layer2 = frame[(frame["layer1_label"] == "SCAM") & frame["layer2_label"].isin(SCAM_TYPE_LABELS)].copy()
    splits = {"layer1": _split(layer1, "layer1_label", seed), "layer2": _split(layer2, "layer2_label", seed)}
    model_dir = models_dir / "hierarchical"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison = []
    metadata = {"status": "ready", "task_type": "hierarchical", "layer1_mapping": LAYER1_LABELS, "layer2_mapping": SCAM_TYPE_LABELS, "models": {}, "split_sizes": {layer: {name: len(part) for name, part in zip(("train", "validation", "test"), parts)} for layer, parts in splits.items()}}
    for model_name, _ in _model_specs().items():
        metadata["models"][model_name] = {}
        for layer_name, (train, validation, test) in splits.items():
            labels = LAYER1_LABELS if layer_name == "layer1" else SCAM_TYPE_LABELS
            label_column = "layer1_label" if layer_name == "layer1" else "layer2_label"
            model = _model_specs()[model_name]
            model.fit(train["message"].tolist(), train[label_column].tolist())
            validation_metrics = _evaluate(model, validation, label_column, labels)
            combined = pd.concat([train, validation], ignore_index=True)
            model.fit(combined["message"].tolist(), combined[label_column].tolist())
            test_metrics = _evaluate(model, test, label_column, labels)
            artifact = f"{model_name}_{layer_name}.joblib"
            joblib.dump(model, model_dir / artifact)
            metadata["models"][model_name][layer_name] = {"artifact": artifact, "labels": labels}
            for split_name, metrics in (("validation", validation_metrics), ("test", test_metrics)):
                comparison.append({"model": model_name, "layer": layer_name, "split": split_name, **{key: value for key, value in metrics.items() if key in {"accuracy", "macro_precision", "macro_recall", "macro_f1"}}})
            pd.DataFrame(test_metrics["confusion_matrix"], index=list(labels), columns=list(labels)).to_csv(results_dir / f"confusion_matrix_{model_name}_{layer_name}.csv", encoding="utf-8-sig")
            (results_dir / f"classification_report_{model_name}_{layer_name}.json").write_text(json.dumps(test_metrics["classification_report"], ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).to_csv(results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    report = {"status": "completed", "task_type": "hierarchical", "models_trained": list(metadata["models"]), "audit": audit, "model_dir": str(model_dir)}
    (results_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Huấn luyện phân cấp\n\nTrạng thái: **COMPLETED**\n\nĐã huấn luyện lớp 1 phát hiện lừa đảo và lớp 2 phân loại loại lừa đảo.\n", encoding="utf-8")
    return report
