"""Verified SCAM/NOT_SCAM training pipeline.

The legacy model in ``models/`` is intentionally separate from this module:
its label ``1`` means spam/suspicious and is not evidence of a scam.  This
pipeline refuses to train until human-verified ``SCAM`` and ``NOT_SCAM``
records exist in the annotation queue.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.dataset_engineering import normalize_text, normalized_hash
from src.preprocess import normalize_messages, redact_message


SCAM_LABELS = {"NOT_SCAM": 0, "SCAM": 1}
SCAM_VERIFIED_STATUSES = {"VERIFIED", "GOLD_VERIFIED", "ROBUSTNESS_VERIFIED"}
REVIEW_LABEL = "NEEDS_REVIEW"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty(value) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _load_queue(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = {"sample_id", "message", "final_label", "review_status", "group_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
    frame["message"] = frame["message"].map(normalize_text)
    frame["final_label"] = frame["final_label"].str.upper().str.strip()
    frame["review_status"] = frame["review_status"].str.upper().str.strip()
    frame["normalized_hash"] = frame["message"].map(normalized_hash)
    return frame


def audit_scam_queue(queue_path: str | Path) -> dict:
    """Return readiness evidence without assigning or changing labels."""
    frame = _load_queue(queue_path)
    eligible = frame[frame["final_label"].isin(SCAM_LABELS) & frame["review_status"].isin(SCAM_VERIFIED_STATUSES)].copy()
    conflicts = int(
        eligible.groupby("normalized_hash")["final_label"].nunique().gt(1).sum()
    ) if not eligible.empty else 0
    eligible_without_conflicts = eligible[~eligible["normalized_hash"].isin(
        set(eligible.loc[eligible.groupby("normalized_hash")["final_label"].transform("nunique") > 1, "normalized_hash"])
    )]
    unique = eligible_without_conflicts.drop_duplicates("normalized_hash")
    counts = unique["final_label"].value_counts().to_dict()
    source_counts = unique.groupby("source_name", dropna=False).size().to_dict() if not unique.empty else {}
    status_counts = frame["review_status"].value_counts().to_dict()
    review_count = int((frame["final_label"] == REVIEW_LABEL).sum())
    ready = bool(
        len(unique) >= 40
        and counts.get("SCAM", 0) >= 20
        and counts.get("NOT_SCAM", 0) >= 20
        and unique["group_id"].nunique() >= 6
        and conflicts == 0
    )
    return {
        "queue_path": str(queue_path),
        "queue_rows": int(len(frame)),
        "verified_rows_before_dedup": int(len(eligible)),
        "verified_rows_after_dedup": int(len(unique)),
        "verified_label_counts": {str(key): int(value) for key, value in counts.items()},
        "needs_review_rows": review_count,
        "unverified_or_unmapped_rows": int(len(frame) - len(eligible)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "source_counts_verified": {str(key): int(value) for key, value in source_counts.items()},
        "conflicting_verified_text_groups": conflicts,
        "group_count_verified": int(unique["group_id"].nunique()) if not unique.empty else 0,
        "ready_for_training": ready,
        "blocking_reasons": ([] if ready else [
            reason for reason, condition in [
                ("fewer_than_40_verified_rows", len(unique) < 40),
                ("fewer_than_20_verified_SCAM", counts.get("SCAM", 0) < 20),
                ("fewer_than_20_verified_NOT_SCAM", counts.get("NOT_SCAM", 0) < 20),
                ("fewer_than_6_verified_groups", unique["group_id"].nunique() < 6 if not unique.empty else True),
                ("conflicting_verified_text_groups", conflicts > 0),
            ] if condition
        ]),
    }


def _verified_frame(queue_path: str | Path) -> pd.DataFrame:
    frame = _load_queue(queue_path)
    frame = frame[frame["final_label"].isin(SCAM_LABELS) & frame["review_status"].isin(SCAM_VERIFIED_STATUSES)].copy()
    conflict_hashes = set(frame.groupby("normalized_hash")["final_label"].nunique().loc[lambda s: s > 1].index)
    frame = frame[~frame["normalized_hash"].isin(conflict_hashes)].drop_duplicates("normalized_hash").copy()
    frame["label_id"] = frame["final_label"].map(SCAM_LABELS).astype(int)
    return frame.reset_index(drop=True)


def _calibrated_svm():
    try:
        return CalibratedClassifierCV(estimator=LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid")
    except TypeError:  # scikit-learn < 1.2 compatibility
        return CalibratedClassifierCV(base_estimator=LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid")


def candidate_models() -> dict[str, Pipeline]:
    def pipeline(vectorizer, classifier):
        return Pipeline([
            ("normalize", FunctionTransformer(normalize_messages, validate=False)),
            ("tfidf", vectorizer),
            ("clf", classifier),
        ])

    return {
        "nb_word_12": pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), MultinomialNB(alpha=0.1)),
        "logreg_word_12": pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        "svm_char_35_calibrated": pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True), _calibrated_svm()),
    }


def _metrics(y_true, y_pred, split: str, model_name: str) -> dict:
    return {
        "model": model_name,
        "split": split,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "false_positive": int(((np.asarray(y_true) == 0) & (np.asarray(y_pred) == 1)).sum()),
        "false_negative": int(((np.asarray(y_true) == 1) & (np.asarray(y_pred) == 0)).sum()),
    }


def select_reject_policy(y_true, probabilities, target_risk: float = 0.10) -> dict:
    """Select confidence/margin thresholds using validation only."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)[:, 1]
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    margin = np.abs(2.0 * probabilities - 1.0)
    candidates = []
    for confidence_threshold in np.arange(0.50, 0.96, 0.05):
        for margin_threshold in np.arange(0.00, 0.51, 0.05):
            automatic = (confidence >= confidence_threshold) & (margin >= margin_threshold)
            predictions = (probabilities >= 0.5).astype(int)
            risk = float((predictions[automatic] != y_true[automatic]).mean()) if automatic.any() else 1.0
            candidates.append((risk <= target_risk, int(automatic.sum()), -risk, confidence_threshold, margin_threshold, risk))
    chosen = max([item for item in candidates if item[0]] or candidates, key=lambda item: (item[1], item[2], -item[3], -item[4]))
    return {
        "confidence_threshold": float(chosen[3]),
        "margin_threshold": float(chosen[4]),
        "target_selective_risk": float(target_risk),
        "validation_selective_risk": float(chosen[5]),
        "validation_automatic_count": int(chosen[1]),
        "validation_review_rate": float(1.0 - chosen[1] / max(1, len(y_true))),
        "status": "validated",
    }


def predict_with_policy(model, message: str, policy: dict) -> dict:
    probabilities = np.asarray(model.predict_proba([message])[0], dtype=float)
    scam_probability = float(probabilities[1])
    confidence = float(probabilities.max())
    margin = float(abs(probabilities[1] - probabilities[0]))
    needs_review = confidence < policy["confidence_threshold"] or margin < policy["margin_threshold"]
    label = REVIEW_LABEL if needs_review else ("SCAM" if scam_probability >= 0.5 else "NOT_SCAM")
    return {
        "label": label,
        "confidence": confidence,
        "scam_probability": scam_probability,
        "score_type": "calibrated_probability",
        "needs_review": bool(needs_review),
        "signals": [
            {"name": "scam_probability", "value": round(scam_probability, 6)},
            {"name": "confidence", "value": round(confidence, 6)},
        ],
    }


def train_verified_model(queue_path: str | Path, models_dir: str | Path, results_dir: str | Path, report_path: str | Path, seed: int = 42) -> dict:
    """Train only from verified labels, or return a truthful blocked result."""
    queue_path, models_dir, results_dir, report_path = map(Path, [queue_path, models_dir, results_dir, report_path])
    audit = audit_scam_queue(queue_path)
    report = {"status": "blocked", "task": "SCAM vs NOT_SCAM", "audit": audit, "model_artifact": None, "error_analysis": "not_generated; existing results/error_analysis.csv belongs to the legacy binary model"}
    if not audit["ready_for_training"]:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_markdown_report(report), encoding="utf-8")
        return report

    frame = _verified_frame(queue_path)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(frame, frame["label_id"], frame["group_id"]))
    test_idx, validation_idx = folds[0][1], folds[1][1]
    test = frame.iloc[test_idx].reset_index(drop=True)
    validation = frame.iloc[validation_idx].reset_index(drop=True)
    train_idx = sorted(set(frame.index) - set(test_idx) - set(validation_idx))
    train = frame.iloc[train_idx].reset_index(drop=True)
    x_train, y_train = train["message"].tolist(), train["label_id"].to_numpy()
    validation_rows, fitted = [], {}
    for name, model in candidate_models().items():
        model.fit(x_train, y_train)
        fitted[name] = model
        pred = model.predict(validation["message"].tolist())
        validation_rows.append(_metrics(validation["label_id"], pred, "validation", name))
    validation_frame = pd.DataFrame(validation_rows).sort_values(["macro_f1", "recall", "precision"], ascending=False)
    winner = str(validation_frame.iloc[0]["model"])
    selected = fitted[winner]
    policy = select_reject_policy(validation["label_id"], selected.predict_proba(validation["message"].tolist()))
    combined = pd.concat([train, validation], ignore_index=True)
    final_model = candidate_models()[winner].fit(combined["message"].tolist(), combined["label_id"].to_numpy())
    test_pred = final_model.predict(test["message"].tolist())
    test_metrics = _metrics(test["label_id"], test_pred, "test", winner)
    errors = []
    for _, row in test.iterrows():
        pred = int(final_model.predict([row["message"]])[0])
        if pred != int(row["label_id"]):
            errors.append({
                "sample_id": row["sample_id"],
                "message_redacted": redact_message(row["message"]),
                "true_label": row["final_label"],
                "predicted_label": "SCAM" if pred else "NOT_SCAM",
                "error_type": "false_negative" if int(row["label_id"]) == 1 else "false_positive",
            })
    results_dir.mkdir(parents=True, exist_ok=True)
    validation_frame.to_csv(results_dir / "scam_validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([test_metrics]).to_csv(results_dir / "scam_test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors, columns=["sample_id", "message_redacted", "true_label", "predicted_label", "error_type"]).to_csv(results_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    cm = confusion_matrix(test["label_id"], test_pred, labels=[0, 1]).tolist()
    artifact_dir = models_dir / "scam_verified"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, artifact_dir / "model.joblib")
    manifest = {
        "model_name": winner,
        "model_version": "scam_verified_v1",
        "label_mapping": SCAM_LABELS,
        "dataset_version": _sha256(queue_path),
        "preprocessing_version": "NFC + lowercase + whitespace normalization in serialized pipeline",
        "training_config": {"seed": seed, "group_aware_split": True},
        "validation_metrics": validation_frame.iloc[0].to_dict(),
        "test_metrics": test_metrics,
        "calibration": {"status": "fitted", "method": "model_predict_proba", "fitted_on": "train_only"},
        "threshold": policy,
        "artifact": "model.joblib",
    }
    (artifact_dir / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = {"status": "completed", "task": "SCAM vs NOT_SCAM", "audit": audit, "winner": winner, "validation": validation_frame.to_dict(orient="records"), "test": test_metrics, "confusion_matrix": cm, "threshold": policy, "model_artifact": str(artifact_dir / "model.joblib")}
    (results_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _markdown_report(report: dict) -> str:
    if report["status"] == "blocked":
        audit = report["audit"]
        reasons = ", ".join(audit["blocking_reasons"]) or "unknown"
        return f"""# SCAM model training report\n\n**BLOCKED** — chưa có đủ dữ liệu `SCAM`/`NOT_SCAM` đã được người xác minh.\n\n- Candidate trong queue: **{audit['queue_rows']}**\n- Mẫu verified: **{audit['verified_rows_after_dedup']}**\n- Mẫu `NEEDS_REVIEW`: **{audit['needs_review_rows']}**\n- Lý do chặn: `{reasons}`\n\n`results/error_analysis.csv` chưa được thay thế: file hiện tại là error analysis của model nhị phân legacy, không phải model SCAM.\n\nNhãn nhị phân cũ `0/1` không được chuyển thành nhãn scam. Chạy lại sau khi annotation và xác minh dữ liệu:\n\n```powershell\npython scripts/train_scam_model.py audit\npython scripts/train_scam_model.py train\n```\n"""
    metrics = report["test"]
    return f"""# SCAM model training report\n\nTrạng thái: **COMPLETED**\n\nModel: `{report['winner']}`\n\n| Metric | Test |\n|---|---:|\n| Precision | {metrics['precision']:.4f} |\n| Recall | {metrics['recall']:.4f} |\n| F1 | {metrics['f1']:.4f} |\n| Macro-F1 | {metrics['macro_f1']:.4f} |\n| False positive | {metrics['false_positive']} |\n| False negative | {metrics['false_negative']} |\n\nConfusion matrix (NOT_SCAM, SCAM): `{report['confusion_matrix']}`\n\nArtifact: `{report['model_artifact']}`\n"""
