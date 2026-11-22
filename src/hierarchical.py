"""Data contract for hierarchical SMS scam classification.

Layer 1 answers whether a message is a scam.  Layer 2 is only evaluated for
messages confirmed as scams.  Non-scam messages keep a small auxiliary type so
legitimate notifications and promotions are not confused with scams.
"""
from __future__ import annotations

import json

import pandas as pd


LAYER1_LABELS = {"NOT_SCAM": 0, "SCAM": 1}
LAYER1_OPTIONS = ["NOT_SCAM", "SCAM", "NEEDS_REVIEW"]
SCAM_TYPE_LABELS = {
    "PHISHING": 0,
    "FINANCIAL_SCAM": 1,
    "OTHER_SCAM": 2,
}
SCAM_TYPE_OPTIONS = list(SCAM_TYPE_LABELS)
NON_SCAM_TYPES = ["LEGITIMATE", "PROMOTION"]
INTENT_OPTIONS = [
    "ASK_OTP",
    "ASK_PASSWORD",
    "ASK_TRANSFER",
    "FAKE_BANK",
    "FAKE_FEE",
    "MALICIOUS_LINK",
    "FAKE_REWARD",
    "FAKE_RECRUITMENT",
    "INVESTMENT_OFFER",
    "IMPERSONATION",
    "OTHER",
]
VERIFIED_STATUSES = {"VERIFIED", "GOLD_VERIFIED", "ROBUSTNESS_VERIFIED"}


def normalize_intents(value) -> list[str]:
    """Normalize a CSV/SQLite intent value to unique known string labels."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            values = parsed if isinstance(parsed, list) else [text]
        except (TypeError, json.JSONDecodeError):
            values = [item.strip() for item in text.split(",")]
    normalized = [str(item).strip().upper() for item in values]
    return list(dict.fromkeys(item for item in normalized if item in INTENT_OPTIONS))


def hierarchy_from_legacy(label: str) -> dict[str, str]:
    """Return a display-only mapping; it must not be used to create verified data."""
    label = str(label or "").strip().upper()
    if label in {"LEGITIMATE", "PROMOTION"}:
        return {"layer1_label": "NOT_SCAM", "layer2_label": "", "non_scam_type": label}
    if label in {"PHISHING", "FINANCIAL_SCAM", "OTHER_SCAM"}:
        return {"layer1_label": "SCAM", "layer2_label": label, "non_scam_type": ""}
    return {"layer1_label": "", "layer2_label": "", "non_scam_type": ""}


def validate_hierarchical_frame(frame: pd.DataFrame, min_per_class: int = 1) -> dict:
    """Validate verified hierarchical rows and report training readiness."""
    required = {"message", "layer1_label", "layer2_label", "verified", "review_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"valid": False, "ready_for_training": False, "errors": [f"Thiếu cột: {missing}"], "layer1_counts": {}, "layer2_counts": {}}
    data = frame.copy()
    for column in required:
        data[column] = data[column].fillna("").astype(str).str.strip()
    errors = []
    if (data["message"].str.strip() == "").any():
        errors.append("Có SMS rỗng.")
    valid_layer1 = set(LAYER1_LABELS)
    valid_layer2 = set(SCAM_TYPE_LABELS)
    invalid_layer1 = sorted(set(data.loc[data["layer1_label"] != "", "layer1_label"]) - valid_layer1)
    invalid_layer2 = sorted(set(data.loc[data["layer2_label"] != "", "layer2_label"]) - valid_layer2)
    if invalid_layer1:
        errors.append(f"Nhãn lớp 1 không hợp lệ: {invalid_layer1}")
    if invalid_layer2:
        errors.append(f"Nhãn lớp 2 không hợp lệ: {invalid_layer2}")
    verified = data["verified"].isin({"1", "True", "true", "VERIFIED"}) & data["review_status"].isin(VERIFIED_STATUSES)
    trainable = data[verified & data["layer1_label"].isin(valid_layer1)].copy()
    missing_layer2 = trainable[trainable["layer1_label"] == "SCAM"]["layer2_label"].eq("")
    if missing_layer2.any():
        errors.append("SMS SCAM đã xác minh phải có loại lừa đảo ở lớp 2.")
    layer1_counts = trainable["layer1_label"].value_counts().reindex(LAYER1_LABELS, fill_value=0).astype(int).to_dict()
    scam_rows = trainable[(trainable["layer1_label"] == "SCAM") & trainable["layer2_label"].isin(valid_layer2)]
    layer2_counts = scam_rows["layer2_label"].value_counts().reindex(SCAM_TYPE_LABELS, fill_value=0).astype(int).to_dict()
    enough_layer1 = all(value >= min_per_class for value in layer1_counts.values())
    enough_layer2 = all(value >= min_per_class for value in layer2_counts.values())
    if not enough_layer1:
        errors.append(f"Lớp 1 cần ít nhất {min_per_class} mẫu cho mỗi nhóm.")
    if not enough_layer2:
        errors.append(f"Lớp 2 cần ít nhất {min_per_class} mẫu cho mỗi loại lừa đảo.")
    return {
        "valid": not errors,
        "ready_for_training": not errors and enough_layer1 and enough_layer2,
        "errors": errors,
        "verified_rows": len(trainable),
        "layer1_counts": layer1_counts,
        "layer2_counts": layer2_counts,
    }
