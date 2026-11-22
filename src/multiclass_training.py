"""Strict four-class trainer for the three production-facing classical models.

It consumes only the locked, human-verified Phase 1 splits. When the lock or
any class is missing it creates truthful blocked reports and no model files.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.classical_ml import (
    LABELS,
    METRIC_COLUMNS,
    evaluate,
    load_locked_dataset,
    make_pipeline,
    model_id,
)
from src.phase4 import select_reject_policy
from src.preprocess import redact_message


LABEL_MAPPING = {label: index for index, label in enumerate(LABELS)}


def _spec(family, features, alpha="", C="", class_weight="none", calibrated=False):
    return {
        "family": family,
        "features": features,
        "min_df": 1,
        "max_df": 1.0,
        "alpha": alpha,
        "C": C,
        "class_weight": class_weight,
        "calibrated": calibrated,
    }


def _three_model_specs():
    baseline = _spec("MultinomialNB", "word_1", alpha=1.0)
    nb_candidates = [_spec("MultinomialNB", feature, alpha=alpha) for feature in ("word_1", "word_12") for alpha in (0.1, 0.3, 0.5, 1.0, 2.0)]
    svm_candidates = [_spec("LinearSVM", "char_35", C=C, class_weight=weight, calibrated=True) for C in (0.1, 0.5, 1.0, 2.0) for weight in ("none", "balanced")]
    return baseline, nb_candidates, svm_candidates


def _selection_key(row):
    return (row["macro_f1"], row["high_risk_recall"], row["macro_precision"])


def _empty_results(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrix").mkdir(exist_ok=True)
    for name, columns in {
        "model_comparison.csv": ["model", "family", "method", "split", "status", "macro_f1", "weighted_f1", "fit_seconds"],
        "per_class_metrics.csv": ["model", "split", "label", "precision", "recall", "f1", "support"],
        "error_analysis.csv": ["model", "sample_id", "message_redacted", "true_label", "predicted_label", "error_type"],
        "test_predictions.csv": ["sample_id", "message_redacted", "true_label", "prediction_nb_baseline", "prediction_nb_optimized", "prediction_svm"],
    }.items():
        pd.DataFrame(columns=columns).to_csv(results_dir / name, index=False, encoding="utf-8-sig")


def _blocked_report(status: dict, results_dir: Path, report_path: Path):
    _empty_results(results_dir)
    payload = {
        "status": "blocked",
        "task_type": "multiclass",
        "num_classes": 4,
        "label_mapping": LABEL_MAPPING,
        "dataset_status": {key: value for key, value in status.items() if key != "frames"},
        "models_trained": [],
        "reason": "No four-class metrics or artifacts are created until verified locked splits exist.",
    }
    (results_dir / "training_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Multiclass Training Report\n\n"
        "**BLOCKED – chưa thể huấn luyện 3 model 4 lớp.**\n\n"
        f"Lý do: `{', '.join(status['reasons'])}`.\n\n"
        "Không chuyển đổi tự động nhãn nhị phân cũ. Cần annotation đã xác minh, chuẩn bị split, kiểm tra leakage và tạo dataset lock trước khi train.\n\n"
        "Các file kết quả đã được tạo với header rỗng; không có metric giả.\n",
        encoding="utf-8",
    )
    return payload


def _metric_rows(model_name, spec, split, metrics, fit_seconds):
    row = {
        "model": model_name,
        "family": spec["family"],
        "method": model_id(spec),
        "split": split,
        "status": "completed",
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "fit_seconds": fit_seconds,
    }
    row.update(metrics)
    return row


def train_three_models(processed_dir, lock_path, models_dir, results_dir, report_path, seed=42):
    processed_dir, lock_path, models_dir, results_dir, report_path = map(Path, [processed_dir, lock_path, models_dir, results_dir, report_path])
    status = load_locked_dataset(processed_dir, lock_path)
    if not status["ready"]:
        return _blocked_report(status, results_dir, report_path)

    train, validation, test = (status["frames"][key] for key in ("train", "validation", "test"))
    baseline_spec, nb_candidates, svm_candidates = _three_model_specs()
    validation_rows = []
    chosen = {"nb_baseline": baseline_spec}

    def fit_validation(name, spec):
        model = make_pipeline(spec)
        started = perf_counter()
        model.fit(train["message"].tolist(), train["label_id"].to_numpy())
        elapsed = perf_counter() - started
        metrics, _, _ = evaluate(model, validation)
        validation_rows.append(_metric_rows(name, spec, "validation", metrics, elapsed))
        return model

    fit_validation("nb_baseline", baseline_spec)
    nb_models = [(spec, fit_validation("nb_candidate", spec)) for spec in nb_candidates]
    nb_rows = [row for row in validation_rows if row["model"] == "nb_candidate"]
    chosen_nb_method = max(nb_rows, key=_selection_key)["method"]
    chosen["nb_optimized"] = next(spec for spec, _ in nb_models if model_id(spec) == chosen_nb_method)
    svm_models = [(spec, fit_validation("svm_candidate", spec)) for spec in svm_candidates]
    svm_rows = [row for row in validation_rows if row["model"] == "svm_candidate"]
    chosen_svm_method = max(svm_rows, key=_selection_key)["method"]
    chosen["svm"] = next(spec for spec, _ in svm_models if model_id(spec) == chosen_svm_method)

    combined = pd.concat([train, validation], ignore_index=True)
    model_dir = models_dir / "multiclass"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrix").mkdir(exist_ok=True)
    test_rows, per_class, errors, prediction_rows, policies = [], [], [], [], {}
    fitted = {}
    for artifact_name, spec in chosen.items():
        model = make_pipeline(spec)
        started = perf_counter()
        model.fit(combined["message"].tolist(), combined["label_id"].to_numpy())
        elapsed = perf_counter() - started
        metrics, predicted, probabilities = evaluate(model, test)
        test_rows.append(_metric_rows(artifact_name, spec, "test", metrics, elapsed))
        fitted[artifact_name] = model
        for index, label in enumerate(LABELS):
            precision, recall, f1, support = precision_recall_fscore_support(test["label_id"], predicted, labels=list(range(4)), zero_division=0)
            per_class.append({"model": artifact_name, "split": "test", "label": label, "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])})
        pd.DataFrame(confusion_matrix(test["label_id"], predicted, labels=list(range(4))), index=LABELS, columns=LABELS).to_csv(results_dir / "confusion_matrix" / f"{artifact_name}.csv", encoding="utf-8")
        if probabilities.ndim == 2:
            validation_model = make_pipeline(spec).fit(train["message"].tolist(), train["label_id"].to_numpy())
            validation_probabilities = validation_model.predict_proba(validation["message"].tolist())
            policies[artifact_name] = {**select_reject_policy(validation["label_id"].to_numpy(), validation_probabilities).__dict__, "status": "validated"}
        joblib.dump(model, model_dir / f"{artifact_name}.joblib")
        for index, row in test.iterrows():
            if int(predicted[index]) != int(row["label_id"]):
                errors.append({"model": artifact_name, "sample_id": row["sample_id"], "message_redacted": redact_message(row["message"]), "true_label": row["label"], "predicted_label": LABELS[int(predicted[index])], "error_type": "false_negative" if int(row["label_id"]) in (2, 3) and int(predicted[index]) in (0, 1) else "misclassified"})

    for index, (_, row) in enumerate(test.iterrows()):
        values = {"sample_id": row["sample_id"], "message_redacted": redact_message(row["message"]), "true_label": row["label"]}
        for artifact_name, model in fitted.items():
            values[f"prediction_{artifact_name}"] = LABELS[int(model.predict([row["message"]])[0])]
        prediction_rows.append(values)
    pd.DataFrame(test_rows).to_csv(results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(results_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["model", "sample_id", "message_redacted", "true_label", "predicted_label", "error_type"]).to_csv(results_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(results_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "task_type": "multiclass",
        "num_classes": 4,
        "label_mapping": LABEL_MAPPING,
        "models": {name: {"artifact": f"{name}.joblib", "method": model_id(spec), "spec": spec, "threshold": policies.get(name)} for name, spec in chosen.items()},
        "dataset_lock": str(lock_path),
        "dataset_lock_sha256": __import__("hashlib").sha256(lock_path.read_bytes()).hexdigest(),
        "split_sizes": {name: len(frame) for name, frame in (("train", train), ("validation", validation), ("test", test))},
        "preprocessing": "serialized normalization + TF-IDF; same pipeline at inference",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = {"status": "completed", "task_type": "multiclass", "num_classes": 4, "label_mapping": LABEL_MAPPING, "models_trained": list(chosen), "model_comparison": test_rows, "selected_nb_method": chosen_nb_method, "selected_svm_method": chosen_svm_method, "model_dir": str(model_dir), "dataset_lock": str(lock_path)}
    (results_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"# Multiclass Training Report\n\nTrạng thái: **COMPLETED**\n\nĐã huấn luyện: `{', '.join(chosen)}`. Kết quả chi tiết nằm trong `results/multiclass/`.\n", encoding="utf-8")
    return report
