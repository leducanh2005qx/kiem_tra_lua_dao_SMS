"""Four-class classical ML benchmark for the locked Phase 1 dataset.

The benchmark is intentionally strict: it refuses to train without a Phase 1
lock manifest and non-empty human-verified train/validation/test splits.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, label_binarize
from sklearn.svm import LinearSVC

from src.dataset_engineering import FOUR_CLASS_LABELS, normalize_text, normalized_hash
from src.preprocess import normalize_messages


LABELS = list(FOUR_CLASS_LABELS)
LABEL_IDS = list(FOUR_CLASS_LABELS.values())
FEATURES = {
    "word_1": {"kind": "word", "ngram_range": (1, 1)},
    "word_12": {"kind": "word", "ngram_range": (1, 2)},
    "word_123": {"kind": "word", "ngram_range": (1, 3)},
    "char_25": {"kind": "char", "ngram_range": (2, 5)},
    "char_35": {"kind": "char", "ngram_range": (3, 5)},
    "hybrid_word12_char35": {"kind": "hybrid"},
}
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
C_VALUES = (0.1, 0.5, 1.0, 2.0)
CLASS_WEIGHTS = (None, "balanced")
MIN_DFS = (1, 2, 5)
MAX_DFS = (1.0, 0.95)
SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
METRIC_COLUMNS = [
    "model_id", "family", "features", "alpha", "C", "class_weight", "calibrated",
    "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
    "high_risk_precision", "high_risk_recall", "high_risk_fpr", "high_risk_fnr",
    "promotion_false_alarm_rate", "multiclass_pr_auc_ovr", "log_loss", "fit_seconds",
    "brier_multiclass", "expected_calibration_error",
    *[f"precision_{label}" for label in LABELS],
    *[f"recall_{label}" for label in LABELS],
    *[f"f1_{label}" for label in LABELS],
]


def _safe(value) -> str:
    return str(value).replace(".", "p").replace(" ", "")


def _vectorizer(kind: str, ngram_range: tuple[int, int], min_df: int, max_df: float) -> TfidfVectorizer:
    args = {
        "analyzer": kind,
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_df": max_df,
        "sublinear_tf": True,
        "strip_accents": None,
    }
    if kind == "word":
        args["token_pattern"] = r"(?u)\b\w[\w@!#$]*\b"
    return TfidfVectorizer(**args)


def _features(name: str, min_df: int, max_df: float):
    spec = FEATURES[name]
    if spec["kind"] != "hybrid":
        return _vectorizer(spec["kind"], spec["ngram_range"], min_df, max_df)
    return FeatureUnion([
        ("word", _vectorizer("word", (1, 2), min_df, max_df)),
        ("char", _vectorizer("char", (3, 5), min_df, max_df)),
    ])


def _estimator(family: str, alpha=None, C=None, class_weight=None, calibrated=False):
    if family == "MultinomialNB":
        return MultinomialNB(alpha=alpha)
    if family == "ComplementNB":
        return ComplementNB(alpha=alpha)
    if family == "LogisticRegression":
        return LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, solver="lbfgs", multi_class="auto", random_state=42)
    base = LinearSVC(C=C, class_weight=class_weight, random_state=42)
    if calibrated:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=-1)
    return base


def candidate_specs() -> list[dict]:
    specs = []
    for family in ("MultinomialNB", "ComplementNB"):
        for features in FEATURES:
            for min_df in MIN_DFS:
                for max_df in MAX_DFS:
                    for alpha in ALPHAS:
                        specs.append({"family": family, "features": features, "min_df": min_df, "max_df": max_df, "alpha": alpha, "C": "", "class_weight": "", "calibrated": False})
    for family in ("LogisticRegression", "LinearSVM"):
        for features in FEATURES:
            for min_df in MIN_DFS:
                for max_df in MAX_DFS:
                    for C in C_VALUES:
                        for class_weight in CLASS_WEIGHTS:
                            specs.append({"family": family, "features": features, "min_df": min_df, "max_df": max_df, "alpha": "", "C": C, "class_weight": class_weight or "none", "calibrated": False})
                            if family == "LinearSVM":
                                specs.append({"family": family, "features": features, "min_df": min_df, "max_df": max_df, "alpha": "", "C": C, "class_weight": class_weight or "none", "calibrated": True})
    return specs


def model_id(spec: dict) -> str:
    calibration = "_cal" if spec["calibrated"] else ""
    alpha = f"_a{_safe(spec['alpha'])}" if spec["alpha"] != "" else ""
    c_value = f"_c{_safe(spec['C'])}" if spec["C"] != "" else ""
    return f"{spec['family']}_{spec['features']}_min{spec['min_df']}_max{_safe(spec['max_df'])}{alpha}{c_value}_w{spec['class_weight']}{calibration}"


def make_pipeline(spec: dict) -> Pipeline:
    estimator = _estimator(spec["family"], alpha=spec["alpha"] or None, C=spec["C"] or None, class_weight=None if spec["class_weight"] in ("", "none") else spec["class_weight"], calibrated=spec["calibrated"])
    return Pipeline([
        ("normalize", FunctionTransformer(normalize_messages, validate=False)),
        ("tfidf", _features(spec["features"], int(spec["min_df"]), float(spec["max_df"]))),
        ("clf", estimator),
    ])


def _scores(model, x):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        return model.predict(x), probabilities, True
    decision = model.decision_function(x)
    return model.predict(x), decision, False


def evaluate(model, frame: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    y_true = frame["label_id"].astype(int).to_numpy()
    predicted, scores, is_probability = _scores(model, frame["message"].tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predicted, labels=LABEL_IDS, zero_division=0)
    high_true = np.isin(y_true, [2, 3])
    high_pred = np.isin(predicted, [2, 3])
    tn, fp, fn, tp = confusion_matrix(high_true, high_pred, labels=[False, True]).ravel()
    binary_precision = precision_score(high_true, high_pred, zero_division=0)
    binary_recall = recall_score(high_true, high_pred, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "high_risk_precision": float(binary_precision),
        "high_risk_recall": float(binary_recall),
        "high_risk_fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "high_risk_fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "promotion_false_alarm_rate": float(((predicted >= 2) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())),
        "multiclass_pr_auc_ovr": float(average_precision_score(label_binarize(y_true, classes=LABEL_IDS), scores, average="macro")),
        "log_loss": float(log_loss(y_true, scores, labels=LABEL_IDS)) if is_probability else np.nan,
        "brier_multiclass": float(np.mean(np.sum((scores - label_binarize(y_true, classes=LABEL_IDS)) ** 2, axis=1))) if is_probability else np.nan,
        "expected_calibration_error": float(_expected_calibration_error(y_true, scores)) if is_probability else np.nan,
    }
    result.update({f"precision_{label}": float(precision[index]) for index, label in enumerate(LABELS)})
    result.update({f"recall_{label}": float(recall[index]) for index, label in enumerate(LABELS)})
    result.update({f"f1_{label}": float(f1[index]) for index, label in enumerate(LABELS)})
    return result, predicted, scores


def _expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == y_true
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper if index == bins - 1 else confidence < upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return error


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {"sample_id", "message", "label", "label_id", "source_type", "group_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
    frame["label_id"] = pd.to_numeric(frame["label_id"], errors="raise").astype(int)
    frame["message"] = frame["message"].map(normalize_text)
    return frame


def load_locked_dataset(processed_dir: str | Path, lock_path: str | Path) -> dict:
    processed_dir = Path(processed_dir)
    lock_path = Path(lock_path)
    reasons = []
    frames = {}
    for split, filename in SPLIT_FILES.items():
        path = processed_dir / filename
        if not path.exists():
            reasons.append(f"missing_{split}")
            continue
        frame = _load_frame(path)
        frames[split] = frame
        if frame.empty:
            reasons.append(f"empty_{split}")
        if not set(frame["label_id"]).issubset(set(LABEL_IDS)) or not set(frame["label"]).issubset(set(LABELS)):
            reasons.append(f"invalid_labels_{split}")
        if frame["source_type"].eq("synthetic").any() and split != "train":
            reasons.append(f"synthetic_outside_train_{split}")
    if not lock_path.exists():
        reasons.append("dataset_lock_missing")
    else:
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            if lock.get("schema_version") != "phase2_dataset_lock_v1":
                reasons.append("unsupported_lock_schema")
            for split, frame in frames.items():
                expected = lock.get("files", {}).get(split, {}).get("sha256")
                actual = _sha256_file(processed_dir / SPLIT_FILES[split])
                if expected != actual:
                    reasons.append(f"lock_hash_mismatch_{split}")
        except Exception as exc:
            reasons.append(f"invalid_lock:{exc}")
    hashes = {}
    groups = {}
    for split, frame in frames.items():
        hashes[split] = set(frame["message"].map(normalized_hash))
        groups[split] = set(frame["group_id"])
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if hashes.get(left, set()) & hashes.get(right, set()):
            reasons.append(f"normalized_text_overlap_{left}_{right}")
        if groups.get(left, set()) & groups.get(right, set()):
            reasons.append(f"group_overlap_{left}_{right}")
    all_classes = set(LABEL_IDS)
    for split, frame in frames.items():
        if set(frame["label_id"]) != all_classes:
            reasons.append(f"missing_class_{split}")
    return {"ready": not reasons, "reasons": sorted(set(reasons)), "frames": frames}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_dataset(processed_dir: str | Path, lock_path: str | Path, seed: int = 42) -> dict:
    """Lock a complete, non-empty, leakage-free four-class dataset."""
    processed_dir = Path(processed_dir)
    lock_path = Path(lock_path)
    frames = {}
    reasons = []
    for split, filename in SPLIT_FILES.items():
        path = processed_dir / filename
        if not path.exists():
            reasons.append(f"missing_{split}")
            continue
        frame = _load_frame(path)
        frames[split] = frame
        if frame.empty:
            reasons.append(f"empty_{split}")
        if set(frame["label_id"]) != set(LABEL_IDS):
            reasons.append(f"missing_class_{split}")
        if split != "train" and frame["source_type"].eq("synthetic").any():
            reasons.append(f"synthetic_outside_train_{split}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if left in frames and right in frames:
            if set(frames[left]["group_id"]) & set(frames[right]["group_id"]):
                reasons.append(f"group_overlap_{left}_{right}")
            if set(frames[left]["message"].map(normalized_hash)) & set(frames[right]["message"].map(normalized_hash)):
                reasons.append(f"normalized_text_overlap_{left}_{right}")
    if reasons:
        return {"locked": False, "reasons": sorted(set(reasons))}
    lock = {
        "schema_version": "phase2_dataset_lock_v1",
        "seed": seed,
        "label_mapping": FOUR_CLASS_LABELS,
        "files": {
            split: {"path": str(processed_dir / SPLIT_FILES[split]), "sha256": _sha256_file(processed_dir / SPLIT_FILES[split]), "rows": len(frame), "class_counts": frame["label"].value_counts().sort_index().to_dict()}
            for split, frame in frames.items()
        },
        "synthetic_policy": "synthetic rows allowed only in train",
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"locked": True, "path": str(lock_path), "files": lock["files"]}


def _empty_outputs(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "confusion_matrices").mkdir(exist_ok=True)
    (results_dir / "charts").mkdir(exist_ok=True)
    for filename, columns in {
        "model_comparison.csv": ["model_id", "family", "features", "macro_f1", "high_risk_recall", "status"],
        "validation_metrics.csv": METRIC_COLUMNS,
        "test_metrics.csv": METRIC_COLUMNS,
        "per_class_metrics.csv": ["split", "model_id", "label", "precision", "recall", "f1", "support"],
        "error_analysis.csv": ["split", "model_id", "sample_id", "message", "true_label", "predicted_label", "error_type"],
        "false_positives.csv": ["split", "model_id", "sample_id", "message", "true_label", "predicted_label", "class"],
        "false_negatives.csv": ["split", "model_id", "sample_id", "message", "true_label", "predicted_label", "class"],
        "class_confusion_analysis.csv": ["split", "model_id", "true_label", "predicted_label", "count"],
        "feature_importance.csv": ["model_id", "family", "features", "class", "feature", "weight", "explanation_type"],
    }.items():
        pd.DataFrame(columns=columns).to_csv(results_dir / filename, index=False, encoding="utf-8-sig")


def _write_blocked_report(report_path: Path, reasons: list[str], dataset_status: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Classical ML Report – Phase 2\n\n"
        "## Trạng thái\n\n"
        "**BLOCKED – chưa có dataset bốn lớp verified và locked để huấn luyện.**\n\n"
        "Pipeline đã được triển khai nhưng không chạy model và không tạo số liệu giả. Phase 1 hiện có candidate nhị phân ở trạng thái `UNMAPPED`, nên không được dùng cho benchmark bốn lớp.\n\n"
        f"Lý do kiểm tra: `{', '.join(reasons)}`.\n\n"
        "## Đã triển khai\n\n"
        "- MultinomialNB, ComplementNB, Logistic Regression và Linear SVM.\n"
        "- Word unigram/bigram/trigram, character 2–5/3–5 và hybrid word-character TF-IDF.\n"
        "- Grid alpha, C, class weight, min_df, max_df; calibrated SVM dùng CV nội bộ trên train.\n"
        "- Metrics multiclass, PR-AUC OvR, high-risk binary, confusion matrix, lỗi và feature importance.\n"
        "- Vectorizer chỉ fit trong pipeline trên train; validation dùng để chọn cấu hình; test chỉ đánh giá sau khóa.\n\n"
        "## Số liệu chưa được công bố\n\n"
        "Chưa có model tốt nhất trên validation, kết quả test, confusion matrix hoặc feature importance hợp lệ vì ba split train/validation/test đang rỗng. Việc ghi các con số lúc này sẽ là bịa số liệu.\n\n"
        "Các file benchmark đã được tạo với header trong `results/classical_ml/`; `model_comparison.csv` chỉ ghi trạng thái BLOCKED, không ghi metric giả.\n\n"
        "## Cách kích hoạt sau khi annotation\n\n"
        "1. Điền annotation và chạy `python scripts/dataset_pipeline.py prepare`.\n"
        "2. Kiểm tra leakage rồi chạy `python scripts/classical_ml_benchmark.py lock`.\n"
        "3. Chạy `python scripts/classical_ml_benchmark.py run`.\n\n"
        "Không xóa hoặc thay thế model nhị phân hiện tại.\n",
        encoding="utf-8",
    )


def _metric_row(spec: dict, metrics: dict, fit_seconds: float) -> dict:
    row = {"model_id": model_id(spec), "family": spec["family"], "features": spec["features"], "alpha": spec["alpha"], "C": spec["C"], "class_weight": spec["class_weight"], "calibrated": spec["calibrated"], "fit_seconds": fit_seconds}
    row.update(metrics)
    return {column: row.get(column, np.nan) for column in METRIC_COLUMNS}


def _save_explanations(model, spec: dict, output_path: Path, top_n: int = 25) -> None:
    clf = model.named_steps["clf"]
    tfidf = model.named_steps["tfidf"]
    rows = []
    if hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_
        explanation_type = "log_likelihood"
    elif hasattr(clf, "coef_"):
        weights = clf.coef_
        explanation_type = "linear_margin_coefficient"
    elif hasattr(clf, "calibrated_classifiers_"):
        estimators = [item.estimator for item in clf.calibrated_classifiers_ if hasattr(item, "estimator") and hasattr(item.estimator, "coef_")]
        if not estimators:
            return
        weights = np.mean([item.coef_ for item in estimators], axis=0)
        explanation_type = "calibrated_linear_margin_coefficient"
    else:
        return
    if hasattr(tfidf, "get_feature_names_out"):
        names = tfidf.get_feature_names_out()
    else:
        names = tfidf.get_feature_names_out()
    for class_index, label in enumerate(LABELS):
        values = weights[class_index]
        for index in np.argsort(values)[::-1][:top_n]:
            rows.append({"model_id": model_id(spec), "family": spec["family"], "features": spec["features"], "class": label, "feature": str(names[index]), "weight": float(values[index]), "explanation_type": explanation_type})
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def _save_errors(frame: pd.DataFrame, predicted: np.ndarray, split: str, current_model_id: str, results_dir: Path) -> None:
    rows, fps, fns = [], [], []
    for index, (_, record) in enumerate(frame.iterrows()):
        true_id, pred_id = int(record["label_id"]), int(predicted[index])
        if true_id == pred_id:
            continue
        true_label, pred_label = LABELS[true_id], LABELS[pred_id]
        rows.append({"split": split, "model_id": current_model_id, "sample_id": record["sample_id"], "message": record["message"], "true_label": true_label, "predicted_label": pred_label, "error_type": "misclassified"})
        fps.append({"split": split, "model_id": current_model_id, "sample_id": record["sample_id"], "message": record["message"], "true_label": true_label, "predicted_label": pred_label, "class": pred_label})
        fns.append({"split": split, "model_id": current_model_id, "sample_id": record["sample_id"], "message": record["message"], "true_label": true_label, "predicted_label": pred_label, "class": true_label})
    pd.DataFrame(rows).to_csv(results_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fps).to_csv(results_dir / "false_positives.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fns).to_csv(results_dir / "false_negatives.csv", index=False, encoding="utf-8-sig")
    confusion = frame.assign(predicted_id=predicted).groupby(["label_id", "predicted_id"]).size().reset_index(name="count")
    confusion["true_label"] = confusion["label_id"].map(dict(enumerate(LABELS)))
    confusion["predicted_label"] = confusion["predicted_id"].map(dict(enumerate(LABELS)))
    confusion.insert(0, "split", split)
    confusion.insert(1, "model_id", current_model_id)
    confusion[["split", "model_id", "true_label", "predicted_label", "count"]].to_csv(results_dir / "class_confusion_analysis.csv", index=False, encoding="utf-8-sig")


def run_benchmark(processed_dir: str | Path, lock_path: str | Path, results_dir: str | Path, report_path: str | Path, seed: int = 42) -> dict:
    results_dir = Path(results_dir)
    _empty_outputs(results_dir)
    status = load_locked_dataset(processed_dir, lock_path)
    (results_dir / "run_config.json").write_text(json.dumps({"seed": seed, "python": sys.version, "platform": platform.platform(), "scikit_learn": sklearn.__version__, "status": "ready" if status["ready"] else "blocked"}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not status["ready"]:
        pd.DataFrame([{"model_id": "__BLOCKED__", "family": "", "features": "", "macro_f1": np.nan, "high_risk_recall": np.nan, "status": "blocked"}]).to_csv(results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
        _write_blocked_report(Path(report_path), status["reasons"], status)
        return {"status": "blocked", "reasons": status["reasons"], "report": str(report_path)}
    train, validation, test = status["frames"]["train"], status["frames"]["validation"], status["frames"]["test"]
    validation_rows, trained = [], {}
    for spec in candidate_specs():
        current_id = model_id(spec)
        model = make_pipeline(spec)
        started = perf_counter()
        model.fit(train["message"].tolist(), train["label_id"].to_numpy())
        elapsed = perf_counter() - started
        metrics, _, _ = evaluate(model, validation)
        validation_rows.append(_metric_row(spec, metrics, elapsed))
        trained[current_id] = (spec, model)
    validation_frame = pd.DataFrame(validation_rows, columns=METRIC_COLUMNS).sort_values(["macro_f1", "high_risk_recall", "balanced_accuracy"], ascending=False)
    validation_frame.to_csv(results_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    validation_frame[["model_id", "family", "features", "macro_f1", "high_risk_recall"]].assign(status="validation_selected_pool").to_csv(results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    winner_id = str(validation_frame.iloc[0]["model_id"])
    winner_spec, _ = trained[winner_id]
    final_model = make_pipeline(winner_spec)
    combined = pd.concat([train, validation], ignore_index=True)
    final_model.fit(combined["message"].tolist(), combined["label_id"].to_numpy())
    test_metrics, predicted, _ = evaluate(final_model, test)
    test_row = _metric_row(winner_spec, test_metrics, 0.0)
    pd.DataFrame([test_row], columns=METRIC_COLUMNS).to_csv(results_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")
    per_class = []
    for label in LABELS:
        per_class.append({"split": "test", "model_id": winner_id, "label": label, "precision": test_row[f"precision_{label}"], "recall": test_row[f"recall_{label}"], "f1": test_row[f"f1_{label}"], "support": int((test["label"] == label).sum())})
    pd.DataFrame(per_class).to_csv(results_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    _save_errors(test, predicted, "test", winner_id, results_dir)
    _save_explanations(final_model, winner_spec, results_dir / "feature_importance.csv")
    joblib.dump(final_model, results_dir / "best_classical_model.joblib")
    (results_dir / "confusion_matrices" / f"{winner_id}.csv").write_text(pd.DataFrame(confusion_matrix(test["label_id"], predicted, labels=LABEL_IDS), index=LABELS, columns=LABELS).to_csv(), encoding="utf-8")
    report_path = Path(report_path)
    report_path.write_text(f"# Classical ML Report – Phase 2\n\nDataset đã khóa và benchmark hoàn tất. Model được chọn bằng validation: `{winner_id}`. Kết quả test nằm tại `results/classical_ml/test_metrics.csv`. Không thay thế model nhị phân đang chạy trên website.\n", encoding="utf-8")
    return {"status": "completed", "winner": winner_id, "validation_models": len(validation_frame), "test_metrics": test_row}
