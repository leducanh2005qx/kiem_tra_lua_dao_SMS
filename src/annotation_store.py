"""Local, persistent annotation queue storage for the Streamlit labeller.

SMS content is stored only in the local SQLite database and the existing local
annotation queue. No content is emitted to application logs.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.dataset_engineering import QUEUE_COLUMNS, normalize_text, normalized_hash
from src.annotation_assist import SUGGESTION_METHOD, suggest_hierarchical_label, suggest_label
from src.hierarchical import INTENT_OPTIONS, LAYER1_LABELS, NON_SCAM_TYPES, SCAM_TYPE_LABELS, normalize_intents


LABELS = {
    "LEGITIMATE": 0,
    "PROMOTION": 1,
    "PHISHING": 2,
    "FINANCIAL_SCAM": 3,
}
LABEL_OPTIONS = ["LEGITIMATE", "PROMOTION", "PHISHING", "FINANCIAL_SCAM", "NEEDS_REVIEW"]
VERIFIED_STATUSES = {"VERIFIED", "GOLD_VERIFIED", "ROBUSTNESS_VERIFIED"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_items (
    sample_id TEXT PRIMARY KEY,
    message TEXT NOT NULL,
    source_name TEXT,
    source_type TEXT,
    source_url TEXT,
    license TEXT,
    original_label TEXT,
    raw_label TEXT,
    annotator_1_label TEXT,
    annotator_2_label TEXT,
    final_label TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    layer1_label TEXT,
    layer2_label TEXT,
    non_scam_type TEXT,
    intent_labels TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT 'UNMAPPED',
    confidence TEXT,
    group_id TEXT,
    pii_status TEXT,
    scam_type TEXT,
    intent TEXT,
    label_reason TEXT,
    annotator_id TEXT,
    target_split TEXT,
    notes TEXT,
    suggested_label TEXT,
    suggested_layer1 TEXT,
    suggested_layer2 TEXT,
    suggested_non_scam_type TEXT,
    suggested_intents TEXT,
    suggested_confidence TEXT,
    suggestion_reasons TEXT,
    suggestion_method TEXT,
    normalized_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS annotation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL,
    previous_label TEXT,
    new_label TEXT NOT NULL,
    review_status TEXT NOT NULL,
    reason TEXT,
    annotator_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sample_id) REFERENCES annotation_items(sample_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_annotation_label ON annotation_items(final_label);
CREATE INDEX IF NOT EXISTS idx_annotation_status ON annotation_items(review_status);
CREATE INDEX IF NOT EXISTS idx_annotation_source ON annotation_items(source_name);
CREATE INDEX IF NOT EXISTS idx_annotation_hash ON annotation_items(normalized_hash);
"""


def _connect(db_path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(SCHEMA)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(annotation_items)")}
        migrations = {
            "revision": "INTEGER NOT NULL DEFAULT 0",
            "verified": "INTEGER NOT NULL DEFAULT 0",
            "layer1_label": "TEXT",
            "layer2_label": "TEXT",
            "non_scam_type": "TEXT",
            "intent_labels": "TEXT",
            "suggested_label": "TEXT",
            "suggested_layer1": "TEXT",
            "suggested_layer2": "TEXT",
            "suggested_non_scam_type": "TEXT",
            "suggested_intents": "TEXT",
            "suggested_confidence": "TEXT",
            "suggestion_reasons": "TEXT",
            "suggestion_method": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in columns:
                try:
                    connection.execute(f"ALTER TABLE annotation_items ADD COLUMN {column} {definition}")
                except sqlite3.OperationalError as exc:
                    # Another process may have completed this migration while
                    # this connection was waiting for the writer lock.
                    if "duplicate column name" not in str(exc).lower():
                        raise
        connection.commit()
        return connection
    except Exception:
        connection.rollback()
        connection.close()
        raise


@contextmanager
def _connection(db_path):
    """Open a configured SQLite connection and always commit/close it."""
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _text(value) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _queue_frame(queue_path) -> pd.DataFrame:
    path = Path(queue_path)
    if not path.exists():
        return pd.DataFrame(columns=QUEUE_COLUMNS)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in QUEUE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[QUEUE_COLUMNS].copy()
    frame["message"] = frame["message"].map(normalize_text)
    frame["normalized_hash"] = frame["message"].map(normalized_hash)
    return frame


def initialize_annotation_store(db_path, queue_path) -> dict:
    """Create the store and import new queue rows without overwriting saved labels."""
    frame = _queue_frame(queue_path)
    changed = False
    with _connection(db_path) as connection:
        existing = {row[0] for row in connection.execute("SELECT sample_id FROM annotation_items")}
        inserted = 0
        for _, row in frame.iterrows():
            if row["sample_id"] in existing:
                continue
            suggestion = suggest_label(row["message"])
            hierarchical = suggest_hierarchical_label(row["message"])
            row["suggested_label"] = suggestion.label
            row["suggested_layer1"] = hierarchical.layer1_label
            row["suggested_layer2"] = hierarchical.layer2_label
            row["suggested_non_scam_type"] = hierarchical.non_scam_type
            row["suggested_intents"] = json.dumps(list(hierarchical.intents), ensure_ascii=False)
            row["suggested_confidence"] = f"{suggestion.confidence:.3f}"
            row["suggestion_reasons"] = json.dumps(list(suggestion.reasons), ensure_ascii=False)
            row["suggestion_method"] = suggestion.method
            values = [row.get(column, "") for column in QUEUE_COLUMNS] + [row["normalized_hash"]]
            columns = ", ".join(QUEUE_COLUMNS + ["normalized_hash"])
            placeholders = ", ".join("?" for _ in values)
            connection.execute(f"INSERT INTO annotation_items ({columns}) VALUES ({placeholders})", values)
            inserted += 1
            existing.add(row["sample_id"])
            changed = True
        pending_suggestions = connection.execute("SELECT sample_id, message FROM annotation_items WHERE (COALESCE(suggested_layer1, '') = '' OR COALESCE(suggestion_method, '') <> ?) AND review_status NOT IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')", (SUGGESTION_METHOD,)).fetchall()
        for row in pending_suggestions:
            suggestion = suggest_label(row["message"])
            hierarchical = suggest_hierarchical_label(row["message"])
            connection.execute("UPDATE annotation_items SET suggested_label=?, suggested_layer1=?, suggested_layer2=?, suggested_non_scam_type=?, suggested_intents=?, suggested_confidence=?, suggestion_reasons=?, suggestion_method=? WHERE sample_id=?", (suggestion.label, hierarchical.layer1_label, hierarchical.layer2_label, hierarchical.non_scam_type, json.dumps(list(hierarchical.intents), ensure_ascii=False), f"{suggestion.confidence:.3f}", json.dumps(list(suggestion.reasons), ensure_ascii=False), suggestion.method, row["sample_id"]))
            changed = True
    if changed:
        export_annotations(db_path, queue_path)
    return {"queue_rows": len(frame), "inserted_rows": inserted}


def _where(filter_mode="unlabeled", search="", source=""):
    clauses, params = [], []
    if filter_mode == "suggested_unverified":
        clauses.append("suggested_layer1 IN ('NOT_SCAM', 'SCAM') AND review_status NOT IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')")
    elif filter_mode == "suspicious_unverified":
        clauses.append("(original_label IN ('1', '1.0') OR raw_label IN ('1', '1.0') OR suggested_layer1 = 'SCAM') AND review_status NOT IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')")
    elif filter_mode == "unlabeled":
        clauses.append("(final_label = '' OR final_label IS NULL OR review_status = 'UNMAPPED')")
    elif filter_mode == "needs_review":
        clauses.append("final_label = 'NEEDS_REVIEW'")
    elif filter_mode == "verified":
        clauses.append("review_status IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')")
    if search:
        clauses.append("(sample_id LIKE ? OR message LIKE ? OR source_name LIKE ?)")
        pattern = f"%{search}%"
        params.extend([pattern, pattern, pattern])
    if source:
        clauses.append("source_name = ?")
        params.append(source)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def annotation_sources(db_path):
    with _connection(db_path) as connection:
        return [row[0] for row in connection.execute("SELECT DISTINCT source_name FROM annotation_items WHERE source_name <> '' ORDER BY source_name")]


def annotation_stats(db_path) -> dict:
    with _connection(db_path) as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM annotation_items").fetchone()[0])
        verified = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE review_status IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED') AND final_label IN ('LEGITIMATE', 'PROMOTION', 'PHISHING', 'FINANCIAL_SCAM')").fetchone()[0])
        needs_review = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE final_label = 'NEEDS_REVIEW'").fetchone()[0])
        machine_suggestions = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE suggested_label IN ('LEGITIMATE', 'PROMOTION', 'PHISHING', 'FINANCIAL_SCAM') AND review_status NOT IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')").fetchone()[0])
        counts = {label: int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE final_label = ? AND review_status IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')", (label,)).fetchone()[0]) for label in LABELS}
    return {"total": total, "verified": verified, "unlabeled": max(0, total - verified - needs_review), "needs_review": needs_review, "machine_suggestions": machine_suggestions, "verified_by_label": counts}


def hierarchical_stats(db_path) -> dict:
    """Return progress counters for the hierarchical annotation contract."""
    with _connection(db_path) as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM annotation_items").fetchone()[0])
        verified = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE verified = 1 AND review_status IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED') AND layer1_label IN ('NOT_SCAM', 'SCAM')").fetchone()[0])
        needs_review = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE review_status = 'NEEDS_REVIEW'").fetchone()[0])
        layer1 = {label: int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE verified = 1 AND layer1_label = ? AND review_status IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')", (label,)).fetchone()[0]) for label in LAYER1_LABELS}
        layer2 = {label: int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE verified = 1 AND layer1_label = 'SCAM' AND layer2_label = ? AND review_status IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')", (label,)).fetchone()[0]) for label in SCAM_TYPE_LABELS}
        suggested = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE suggested_layer1 IN ('NOT_SCAM', 'SCAM') AND review_status NOT IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED')").fetchone()[0])
    return {"total": total, "verified": verified, "unlabeled": max(0, total - verified - needs_review), "needs_review": needs_review, "machine_suggestions": suggested, "layer1": layer1, "layer2": layer2}


def list_annotations(db_path, filter_mode="unlabeled", search="", source="", page=1, page_size=1):
    where, params = _where(filter_mode, search, source)
    page = max(1, int(page))
    page_size = min(1000, max(1, int(page_size)))
    with _connection(db_path) as connection:
        total = int(connection.execute(f"SELECT COUNT(*) FROM annotation_items{where}", params).fetchone()[0])
        rows = connection.execute(f"SELECT * FROM annotation_items{where} ORDER BY sample_id LIMIT ? OFFSET ?", params + [page_size, (page - 1) * page_size]).fetchall()
    return [dict(row) for row in rows], total


def quality_summary(db_path) -> dict:
    with _connection(db_path) as connection:
        duplicate_groups = int(connection.execute("SELECT COUNT(*) FROM (SELECT normalized_hash FROM annotation_items GROUP BY normalized_hash HAVING COUNT(*) > 1)").fetchone()[0])
        duplicate_rows = int(connection.execute("SELECT COALESCE(SUM(group_count - 1), 0) FROM (SELECT COUNT(*) AS group_count FROM annotation_items GROUP BY normalized_hash HAVING COUNT(*) > 1)").fetchone()[0])
        empty_rows = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE TRIM(message) = ''").fetchone()[0])
        invalid_labels = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE final_label NOT IN ('', 'NEEDS_REVIEW', 'LEGITIMATE', 'PROMOTION', 'PHISHING', 'FINANCIAL_SCAM', 'OTHER_SCAM')").fetchone()[0])
        missing_reason = int(connection.execute("SELECT COUNT(*) FROM annotation_items WHERE review_status IN ('VERIFIED', 'GOLD_VERIFIED', 'ROBUSTNESS_VERIFIED') AND TRIM(COALESCE(label_reason, '')) = ''").fetchone()[0])
    near_duplicate_rows = 0
    manifest = Path(db_path).parent / "manifests" / "duplicate_report.csv"
    if manifest.exists():
        try:
            duplicate_report = pd.read_csv(manifest, encoding="utf-8-sig")
            near_duplicate_rows = int((duplicate_report.get("kind", pd.Series(dtype=str)) == "near_duplicate_nearest").sum())
        except (OSError, ValueError, pd.errors.ParserError):
            near_duplicate_rows = 0
    return {"duplicate_groups": duplicate_groups, "duplicate_rows_beyond_first": duplicate_rows, "near_duplicate_rows": near_duplicate_rows, "empty_messages": empty_rows, "invalid_labels": invalid_labels, "verified_missing_reason": missing_reason}


def save_annotation(db_path, queue_path, sample_id, label, reason, scam_type="", intent="", annotator_id="", confidence="", notes="", expected_revision=None):
    label = str(label or "").strip().upper()
    if label not in LABEL_OPTIONS:
        raise ValueError("Nhãn không hợp lệ.")
    if not str(reason or "").strip():
        raise ValueError("Cần nhập lý do gắn nhãn hoặc lý do NEEDS_REVIEW.")
    if label == "NEEDS_REVIEW":
        status, verified = "NEEDS_REVIEW", 0
    else:
        status, verified = "VERIFIED", 1
    with _connection(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT final_label, annotator_1_label, annotator_2_label, revision FROM annotation_items WHERE sample_id = ?", (sample_id,)).fetchone()
        if not row:
            raise ValueError("Không tìm thấy SMS trong hàng đợi.")
        current_revision = int(row["revision"] or 0)
        if expected_revision is not None and current_revision != int(expected_revision):
            raise ValueError("SMS này vừa được người khác gắn nhãn. Hãy tải lại hàng đợi trước khi sửa tiếp.")
        previous = row["final_label"] or ""
        updated = connection.execute(
            """UPDATE annotation_items SET final_label=?, verified=?, review_status=?, confidence=?, scam_type=?, intent=?, label_reason=?, annotator_id=?, notes=?, revision=revision+1, updated_at=CURRENT_TIMESTAMP WHERE sample_id=? AND revision=?""",
            (label, verified, status, str(confidence or ""), str(scam_type or ""), str(intent or ""), str(reason).strip(), str(annotator_id or ""), str(notes or ""), sample_id, current_revision),
        )
        if updated.rowcount != 1:
            raise ValueError("SMS này vừa được người khác gắn nhãn. Hãy tải lại hàng đợi trước khi sửa tiếp.")
        connection.execute("INSERT INTO annotation_events (sample_id, previous_label, new_label, review_status, reason, annotator_id) VALUES (?, ?, ?, ?, ?, ?)", (sample_id, previous, label, status, str(reason).strip(), str(annotator_id or "")))
    export_annotations(db_path, queue_path)
    return {"sample_id": sample_id, "label": label, "review_status": status, "verified": bool(verified), "previous_label": previous}


def save_hierarchical_annotation(db_path, queue_path, sample_id, layer1_label, layer2_label="", non_scam_type="", intents=None, reason="", annotator_id="", confidence="", notes="", expected_revision=None):
    """Persist a hierarchical decision without inferring any human label."""
    layer1_label = str(layer1_label or "").strip().upper()
    layer2_label = str(layer2_label or "").strip().upper()
    non_scam_type = str(non_scam_type or "").strip().upper()
    if layer1_label not in {"NEEDS_REVIEW", *LAYER1_LABELS}:
        raise ValueError("Nhãn lớp 1 không hợp lệ.")
    if not str(reason or "").strip():
        raise ValueError("Cần nhập lý do gắn nhãn hoặc lý do cần xem lại.")
    intent_values = normalize_intents(intents)
    if layer1_label == "NEEDS_REVIEW":
        final_label, layer2_label, non_scam_type, verified, status = "NEEDS_REVIEW", "", "", 0, "NEEDS_REVIEW"
    elif layer1_label == "NOT_SCAM":
        if non_scam_type not in NON_SCAM_TYPES:
            raise ValueError("SMS không lừa đảo phải chọn Hợp lệ hoặc Quảng cáo.")
        final_label, layer2_label, verified, status = non_scam_type, "", 1, "VERIFIED"
    else:
        if layer2_label not in SCAM_TYPE_LABELS:
            raise ValueError("SMS lừa đảo phải chọn loại lừa đảo ở lớp 2.")
        final_label, non_scam_type, verified, status = layer2_label, "", 1, "VERIFIED"
    with _connection(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT final_label, revision FROM annotation_items WHERE sample_id = ?", (sample_id,)).fetchone()
        if not row:
            raise ValueError("Không tìm thấy SMS trong hàng đợi.")
        current_revision = int(row["revision"] or 0)
        if expected_revision is not None and current_revision != int(expected_revision):
            raise ValueError("SMS này vừa được người khác gắn nhãn. Hãy tải lại hàng đợi trước khi sửa tiếp.")
        previous = row["final_label"] or ""
        updated = connection.execute(
            """UPDATE annotation_items SET final_label=?, layer1_label=?, layer2_label=?, non_scam_type=?, intent_labels=?, verified=?, review_status=?, confidence=?, label_reason=?, annotator_id=?, notes=?, revision=revision+1, updated_at=CURRENT_TIMESTAMP WHERE sample_id=? AND revision=?""",
            (final_label, "" if layer1_label == "NEEDS_REVIEW" else layer1_label, layer2_label, non_scam_type, json.dumps(intent_values, ensure_ascii=False), verified, status, str(confidence or ""), str(reason).strip(), str(annotator_id or ""), str(notes or ""), sample_id, current_revision),
        )
        if updated.rowcount != 1:
            raise ValueError("SMS này vừa được người khác gắn nhãn. Hãy tải lại hàng đợi trước khi sửa tiếp.")
        connection.execute("INSERT INTO annotation_events (sample_id, previous_label, new_label, review_status, reason, annotator_id) VALUES (?, ?, ?, ?, ?, ?)", (sample_id, previous, final_label, status, str(reason).strip(), str(annotator_id or "")))
    export_annotations(db_path, queue_path)
    return {"sample_id": sample_id, "layer1_label": layer1_label, "layer2_label": layer2_label, "non_scam_type": non_scam_type, "verified": bool(verified), "review_status": status, "previous_label": previous}


def export_annotations(db_path, queue_path) -> int:
    """Write the local annotation state back to the derived queue atomically."""
    path = Path(queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connection(db_path) as connection:
        rows = [dict(row) for row in connection.execute("SELECT " + ", ".join(QUEUE_COLUMNS) + " FROM annotation_items ORDER BY sample_id")]
    frame = pd.DataFrame(rows, columns=QUEUE_COLUMNS)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as temporary:
            temporary_name = temporary.name
        frame.to_csv(temporary_name, index=False, encoding="utf-8-sig")
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return len(frame)


def annotation_csv(db_path) -> bytes:
    with _connection(db_path) as connection:
        rows = [dict(row) for row in connection.execute("SELECT " + ", ".join(QUEUE_COLUMNS) + " FROM annotation_items ORDER BY sample_id")]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=QUEUE_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def annotation_history(db_path, sample_id, limit=20):
    with _connection(db_path) as connection:
        rows = connection.execute("SELECT * FROM annotation_events WHERE sample_id = ? ORDER BY id DESC LIMIT ?", (sample_id, int(limit))).fetchall()
    return [dict(row) for row in rows]
