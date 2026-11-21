"""Dataset loading and leakage-aware quality checks."""
import hashlib
import pandas as pd
from src.preprocess import normalize_message

REQUIRED_COLUMNS = {"message", "label"}


def load_sms_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"message": "string"})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: thiếu cột {sorted(missing)}. Cần message,label.")
    df = df[["message", "label"]].copy()
    original_rows = len(df)
    df = df.dropna(subset=["message", "label"])
    df["message"] = df["message"].astype(str).map(normalize_message)
    df = df.loc[df["message"].str.len() > 0].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: label chỉ được là 0 hoặc 1.")
    # For contradictory duplicate text, keep none: otherwise its true label is ambiguous.
    conflict = df.groupby("message")["label"].nunique()
    conflicting = set(conflict[conflict > 1].index)
    contradictory_rows = int(df["message"].isin(conflicting).sum())
    if conflicting:
        df = df.loc[~df["message"].isin(conflicting)].copy()
    duplicated_rows = int(df.duplicated("message").sum())
    df = df.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
    stats = {
        "original_rows": original_rows,
        "empty_rows": original_rows - len(df) - duplicated_rows - contradictory_rows,
        "duplicate_rows_removed": duplicated_rows,
        "contradictory_rows_removed": contradictory_rows,
        "clean_rows": len(df),
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
    }
    return df, stats


def remove_exact_test_overlap(train_df, test_df):
    """Do not alter the held-out test set; remove only exact overlaps from training."""
    overlap = set(train_df["message"]) & set(test_df["message"])
    clean_train = train_df.loc[~train_df["message"].isin(overlap)].reset_index(drop=True)
    return clean_train, len(overlap), len(train_df) - len(clean_train)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
