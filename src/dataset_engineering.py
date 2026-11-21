"""Leakage-aware, provenance-preserving four-class dataset engineering.

This module deliberately does not infer a four-class label from the existing
binary corpus. Imported records remain candidates until a human reviewer fills
``final_label`` and marks the record as verified.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors


FOUR_CLASS_LABELS = {
    "LEGITIMATE": 0,
    "PROMOTION": 1,
    "PHISHING": 2,
    "FINANCIAL_SCAM": 3,
}
VALID_SOURCE_TYPES = {"real", "synthetic", "heuristic-derived", "manually-verified"}
VERIFIED_STATUSES = {"VERIFIED", "GOLD_VERIFIED", "ROBUSTNESS_VERIFIED"}
QUEUE_COLUMNS = [
    "sample_id", "message", "source_name", "source_type", "source_url", "license",
    "original_label", "raw_label", "annotator_1_label", "annotator_2_label",
    "final_label", "review_status", "confidence", "group_id", "pii_status",
    "target_split", "notes",
]
PROCESSED_COLUMNS = [
    "sample_id", "message", "message_anonymized", "label", "label_id",
    "source_name", "source_type", "source_url", "license", "original_label",
    "raw_label", "group_id", "pii_status", "split",
]

URL_RE = re.compile(r"(?i)(?:https?://|www\.)[^\s<>]+")
EMAIL_RE = re.compile(r"(?i)\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\w)")
LONG_NUMBER_RE = re.compile(r"(?<!\w)\d{8,}(?!\w)")
OTP_RE = re.compile(r"(?i)\b(?:otp|mã xác thực|ma xac thuc|mã otp)\b")
MONEY_RE = re.compile(r"(?i)\b(?:vnd|vnđ|đồng|dong|triệu|trieu|chuyển khoản|chuyen khoan|phí|phi|lãi suất|lai suat)\b")


def _text(value) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def normalize_text(value) -> str:
    """Normalize Unicode and whitespace while preserving Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", _text(value)).replace("\u200b", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalized_hash(value) -> str:
    return hashlib.sha256(normalize_text(value).casefold().encode("utf-8")).hexdigest()


def template_group_id(value) -> str:
    """Create a conservative template group key for grouped splitting."""
    text = normalize_text(value).casefold()
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = re.sub(r"\d+", " <NUM> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "grp_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def anonymize_message(value) -> str:
    """Create a derived, safer text view; the imported raw message is retained."""
    text = normalize_text(value)

    def url_replacement(match):
        raw = match.group(0)
        parsed = urlparse(raw if not raw.lower().startswith("www.") else "http://" + raw)
        domain = (parsed.netloc or "").lower().split(":", 1)[0]
        return f"[url_domain:{domain or 'unknown'}]"

    text = URL_RE.sub(url_replacement, text)
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    # Keep short OTPs and money amounts; mask likely account/identifier numbers.
    text = LONG_NUMBER_RE.sub("[account_number]", text)
    return text


def make_sample_id(source_name, message, raw_label, ordinal=0) -> str:
    payload = "|".join([_text(source_name), _text(raw_label), normalize_text(message), str(ordinal)])
    return "sms_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def read_records(path: str | Path) -> pd.DataFrame:
    """Read CSV, JSON, or JSONL with common message/text field aliases."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            for key in ("data", "train", "records", "examples"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        frame = pd.DataFrame(payload)
    else:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    aliases = {"text": "message", "content": "message", "sms": "message"}
    for old, new in aliases.items():
        if "message" not in frame.columns and old in frame.columns:
            frame = frame.rename(columns={old: new})
    if "message" not in frame.columns:
        raise ValueError(f"{path}: cần cột message hoặc text/content/sms")
    if "label" not in frame.columns:
        frame["label"] = ""
    return frame


def import_records(
    input_path: str | Path,
    output_path: str | Path,
    source_name: str,
    source_type: str,
    source_url: str,
    license_name: str,
) -> dict:
    """Import records without assigning a four-class label."""
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"source_type phải thuộc {sorted(VALID_SOURCE_TYPES)}")
    frame = read_records(input_path)
    rows = []
    for ordinal, record in frame.iterrows():
        message = normalize_text(record.get("message", ""))
        raw_label = _text(record.get("label", record.get("raw_label", "")))
        if not message:
            continue
        rows.append({
            "sample_id": make_sample_id(source_name, message, raw_label, ordinal),
            "message": message,
            "source_name": source_name,
            "source_type": source_type,
            "source_url": source_url,
            "license": license_name,
            "original_label": raw_label,
            "raw_label": raw_label,
            "annotator_1_label": "",
            "annotator_2_label": "",
            "final_label": "",
            "review_status": "UNMAPPED",
            "confidence": "",
            "group_id": template_group_id(message),
            "pii_status": "raw_retained_derived_anonymized",
            "target_split": "",
            "notes": "Imported candidate; four-class label intentionally not inferred.",
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows, columns=QUEUE_COLUMNS)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {"input": str(input_path), "output": str(output_path), "rows": len(result), "source_name": source_name}


def merge_candidate_files(paths: list[str | Path], output_path: str | Path) -> pd.DataFrame:
    """Merge candidate files while preserving provenance and duplicate rows."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        missing = set(QUEUE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
        frames.append(frame[QUEUE_COLUMNS])
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QUEUE_COLUMNS)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def _duplicate_report(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["normalized_text"] = work["message"].map(normalize_text).str.casefold()
    work["normalized_hash"] = work["normalized_text"].map(normalized_hash)
    rows = []
    for key, group in work.groupby("normalized_hash", sort=False):
        if len(group) > 1:
            ids = group["sample_id"].tolist()
            for sample_id in ids:
                rows.append({"kind": "exact_normalized", "sample_id": sample_id, "related_sample_id": "", "similarity": 1.0, "group_id": key, "count": len(ids)})
    texts = work["normalized_text"].tolist()
    if len(texts) > 1:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        neighbors = NearestNeighbors(n_neighbors=2, metric="cosine").fit(matrix)
        distances, indices = neighbors.kneighbors(matrix)
        for index, (distance_row, index_row) in enumerate(zip(distances, indices)):
            similarity = float(1.0 - distance_row[1])
            if similarity >= 0.90 and work.iloc[index]["normalized_hash"] != work.iloc[index_row[1]]["normalized_hash"]:
                rows.append({"kind": "near_duplicate_nearest", "sample_id": work.iloc[index]["sample_id"], "related_sample_id": work.iloc[index_row[1]]["sample_id"], "similarity": round(similarity, 6), "group_id": work.iloc[index]["group_id"], "count": 1})
    return pd.DataFrame(rows, columns=["kind", "sample_id", "related_sample_id", "similarity", "group_id", "count"])


def _statistics(frame: pd.DataFrame) -> pd.DataFrame:
    messages = frame["message"].fillna("").map(normalize_text)
    normalized = messages.str.casefold()
    rows = [
        ("rows", len(frame)),
        ("empty_messages", int((messages == "").sum())),
        ("unique_messages", int(normalized.nunique())),
        ("exact_duplicate_rows", int(frame.duplicated("message").sum())),
        ("normalized_duplicate_rows", int(normalized.duplicated(keep=False).sum() - normalized[normalized.duplicated(keep=False)].nunique())),
        ("with_url", int(messages.str.contains(r"(?i)(?:https?://|www\.)", regex=True).sum())),
        ("with_otp_marker", int(messages.str.contains(OTP_RE).sum())),
        ("with_money_marker", int(messages.str.contains(MONEY_RE).sum())),
        ("with_vietnamese_diacritics", int(messages.map(lambda x: any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", x))).sum())),
        ("real_rows", int((frame["source_type"] == "real").sum())),
        ("synthetic_rows", int((frame["source_type"] == "synthetic").sum())),
        ("heuristic_derived_rows", int((frame["source_type"] == "heuristic-derived").sum())),
        ("manually_verified_source_rows", int((frame["source_type"] == "manually-verified").sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _label_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        final_label = _text(row.get("final_label", "")).strip().upper()
        status = _text(row.get("review_status", "")).strip().upper()
        issues = []
        if final_label and final_label not in FOUR_CLASS_LABELS:
            issues.append("invalid_final_label")
        if final_label and status not in VERIFIED_STATUSES:
            issues.append("label_not_verified")
        if status in VERIFIED_STATUSES and not final_label:
            issues.append("verified_without_final_label")
        if _text(row.get("source_type")) == "heuristic-derived" and status in VERIFIED_STATUSES:
            issues.append("heuristic_source_requires_human_reverification")
        rows.append({
            "sample_id": row["sample_id"], "source_name": row["source_name"],
            "original_label": row["original_label"], "final_label": final_label,
            "review_status": status, "issue": ";".join(issues),
            "trainable": bool(final_label in FOUR_CLASS_LABELS and status in VERIFIED_STATUSES and not issues),
        })
    return pd.DataFrame(rows)


def annotation_agreement(frame: pd.DataFrame) -> dict:
    a = frame["annotator_1_label"].fillna("").str.upper().str.strip()
    b = frame["annotator_2_label"].fillna("").str.upper().str.strip()
    mask = a.isin(FOUR_CLASS_LABELS) & b.isin(FOUR_CLASS_LABELS)
    if not mask.any():
        return {"rows_with_two_labels": 0, "agreement": None, "cohen_kappa": None, "note": "Chưa có hai bộ nhãn độc lập hợp lệ."}
    agreement = float((a[mask] == b[mask]).mean())
    kappa = float(cohen_kappa_score(a[mask], b[mask], labels=list(FOUR_CLASS_LABELS)))
    return {"rows_with_two_labels": int(mask.sum()), "agreement": agreement, "cohen_kappa": kappa}


def _empty_processed(path: Path) -> None:
    pd.DataFrame(columns=PROCESSED_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def _assign_splits(frame: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Assign train/validation/test by group with approximate stratification."""
    result = pd.Series("train", index=frame.index, dtype="string")
    if len(frame) < 7 or frame["label"].nunique() < 2 or frame["group_id"].nunique() < 3:
        return result
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = list(splitter.split(frame, frame["label"], frame["group_id"]))
    test_indices = folds[0][1]
    validation_indices = folds[1][1]
    result.iloc[test_indices] = "test"
    result.iloc[validation_indices] = "validation"
    return result


def build_processed(queue_path: str | Path, processed_dir: str | Path, seed: int = 42) -> dict:
    """Materialize only human-verified records into processed splits."""
    queue = pd.read_csv(queue_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    audit = _label_audit(queue)
    eligible = audit["trainable"]
    work = queue.loc[eligible].copy()
    work["label"] = work["final_label"].str.upper().str.strip()
    work["label_id"] = work["label"].map(FOUR_CLASS_LABELS).astype(int)
    # Explicit gold/robustness targets are never reassigned to ordinary splits.
    explicit = work["target_split"].str.lower().map({"gold_test": "gold_test", "robustness_test": "robustness_test"})
    work["split"] = explicit.fillna("")
    regular_mask = work["split"] == ""
    if regular_mask.any():
        work.loc[regular_mask, "split"] = _assign_splits(work.loc[regular_mask], seed=seed).to_numpy()
    # Synthetic records can only be used in the training split.
    work.loc[(work["source_type"] == "synthetic") & (work["split"] != "train"), "split"] = "train"
    work["message_anonymized"] = work["message"].map(anonymize_message)
    output = work.rename(columns={"message": "message"})[
        ["sample_id", "message", "message_anonymized", "label", "label_id", "source_name", "source_type", "source_url", "license", "original_label", "raw_label", "group_id", "pii_status", "split"]
    ]
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in [("train", "train.csv"), ("validation", "validation.csv"), ("test", "test.csv"), ("gold_test", "gold_test.csv"), ("robustness_test", "robustness_test.csv")]:
        subset = output[output["split"] == split] if len(output) else pd.DataFrame(columns=PROCESSED_COLUMNS)
        subset.reindex(columns=PROCESSED_COLUMNS).to_csv(processed_dir / filename, index=False, encoding="utf-8-sig")
    return {"verified_rows": len(output), "class_counts": output["label"].value_counts().sort_index().to_dict() if len(output) else {}, "split_counts": output["split"].value_counts().sort_index().to_dict() if len(output) else {}}


def audit_dataset(queue_path: str | Path, manifests_dir: str | Path, processed_dir: str | Path) -> dict:
    queue = pd.read_csv(queue_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = queue[["sample_id", "source_name", "source_type", "source_url", "license", "original_label", "raw_label", "group_id", "review_status", "final_label", "target_split", "pii_status"]].copy()
    dataset_manifest["message_hash"] = queue["message"].map(lambda value: hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest())
    dataset_manifest["normalized_hash"] = queue["message"].map(normalized_hash)
    dataset_manifest["provenance_status"] = dataset_manifest["source_type"].map({"real": "source_declared", "synthetic": "synthetic_declared", "heuristic-derived": "heuristic_declared", "manually-verified": "manual_source_declared"}).fillna("unknown")
    dataset_manifest.to_csv(manifests_dir / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    _statistics(queue).to_csv(manifests_dir / "dataset_statistics.csv", index=False, encoding="utf-8-sig")
    duplicates = _duplicate_report(queue)
    duplicates.to_csv(manifests_dir / "duplicate_report.csv", index=False, encoding="utf-8-sig")
    labels = _label_audit(queue)
    labels.to_csv(manifests_dir / "label_audit.csv", index=False, encoding="utf-8-sig")
    source_dist = queue.groupby(["source_name", "source_type", "original_label", "review_status"], dropna=False).size().reset_index(name="rows")
    source_dist.to_csv(manifests_dir / "source_distribution.csv", index=False, encoding="utf-8-sig")
    class_rows = []
    for label, label_id in FOUR_CLASS_LABELS.items():
        class_rows.append({"label": label, "label_id": label_id, "candidate_rows": int((labels["final_label"] == label).sum()), "verified_rows": int(labels.loc[(labels["final_label"] == label) & labels["trainable"], "sample_id"].nunique())})
    pd.DataFrame(class_rows).to_csv(manifests_dir / "class_distribution.csv", index=False, encoding="utf-8-sig")
    processed_rows = []
    for path in sorted(Path(processed_dir).glob("*.csv")):
        if path.name not in {"train.csv", "validation.csv", "test.csv", "gold_test.csv", "robustness_test.csv"}:
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        counts = Counter(frame.get("label", pd.Series(dtype=str)))
        for label, label_id in FOUR_CLASS_LABELS.items():
            processed_rows.append({"split": path.stem, "label": label, "label_id": label_id, "rows": int(counts.get(label, 0))})
    pd.DataFrame(processed_rows, columns=["split", "label", "label_id", "rows"]).to_csv(manifests_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    agreement = annotation_agreement(queue)
    (manifests_dir / "annotation_agreement.json").write_text(json.dumps(agreement, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"rows": len(queue), "duplicate_report_rows": len(duplicates), "label_audit_rows": len(labels), "annotation_agreement": agreement}


def write_source_manifest(path: str | Path, records: list[dict]) -> None:
    Path(path).write_text(json.dumps({"schema_version": "phase1", "four_class_labels": FOUR_CLASS_LABELS, "sources": records}, ensure_ascii=False, indent=2), encoding="utf-8")
