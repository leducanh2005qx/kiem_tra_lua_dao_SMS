"""Privacy-conscious SQLite storage for activity history and user feedback."""
from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.config import APP_HISTORY_DB


SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
    prediction_score REAL,
    score_type TEXT NOT NULL,
    message_length INTEGER NOT NULL CHECK (message_length >= 0),
    message_text TEXT,
    message_hash TEXT,
    content_saved INTEGER NOT NULL DEFAULT 0 CHECK (content_saved IN (0, 1)),
    hybrid_label TEXT,
    risk_level TEXT,
    rule_override INTEGER NOT NULL DEFAULT 0,
    needs_review INTEGER NOT NULL DEFAULT 0,
    risk_reasons TEXT,
    risk_signals TEXT,
    risk_urls TEXT,
    risk_evidence TEXT
);

CREATE TABLE IF NOT EXISTS prediction_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('correct', 'incorrect', 'uncertain')),
    suggested_label INTEGER CHECK (suggested_label IN (0, 1, 2)),
    comment TEXT,
    created_at TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending' CHECK (review_status IN ('pending', 'reviewed', 'rejected')),
    reviewed_at TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analysis_history(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analysis_created_at ON analysis_history(created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_label ON analysis_history(predicted_label);
CREATE INDEX IF NOT EXISTS idx_analysis_model ON analysis_history(model_name);
CREATE INDEX IF NOT EXISTS idx_feedback_status ON prediction_feedback(review_status);
"""


def _path(db_path=None) -> Path:
    return Path(db_path or APP_HISTORY_DB)


def _connect(db_path=None):
    path = _path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(SCHEMA)
        migrations = {
            "analysis_history": {
                "hybrid_label": "TEXT",
                "risk_level": "TEXT",
                "rule_override": "INTEGER NOT NULL DEFAULT 0",
                "needs_review": "INTEGER NOT NULL DEFAULT 0",
                "risk_reasons": "TEXT",
                "risk_signals": "TEXT",
                "risk_urls": "TEXT",
                "risk_evidence": "TEXT",
            },
            "prediction_feedback": {
                "suggested_label_code": "TEXT",
                "review_state": "TEXT NOT NULL DEFAULT 'pending_review'",
                "feedback_reason": "TEXT",
            },
        }
        for table, columns in migrations.items():
            existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for column, definition in columns.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        if int(connection.execute("PRAGMA user_version").fetchone()[0]) < 1:
            connection.execute("UPDATE prediction_feedback SET review_state='verified' WHERE review_status='reviewed' AND COALESCE(review_state, '') IN ('', 'pending_review')")
            connection.execute("UPDATE prediction_feedback SET review_state='rejected' WHERE review_status='rejected' AND COALESCE(review_state, '') IN ('', 'pending_review')")
            connection.execute("PRAGMA user_version=1")
        connection.commit()
        return connection
    except Exception:
        connection.rollback()
        connection.close()
        raise


@contextmanager
def _connection(db_path=None):
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(db_path=None):
    """Create the local database and tables; safe to call repeatedly."""
    with _connection(db_path):
        pass


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_date_range(date_from=None, date_to=None):
    if date_from and date_to and date_from > date_to:
        raise ValueError("Khoảng ngày không hợp lệ.")


def _date_clauses(date_from=None, date_to=None, column="created_at"):
    _validate_date_range(date_from, date_to)
    clauses = []
    params = []
    if date_from:
        clauses.append(f"{column} >= ?")
        params.append(f"{date_from}T00:00:00")
    if date_to:
        next_day = date_to + timedelta(days=1)
        clauses.append(f"{column} < ?")
        params.append(f"{next_day}T00:00:00")
    return clauses, params


def add_analysis(
    model_name,
    predicted_label,
    prediction_score,
    score_type,
    message,
    save_content=False,
    save_hash=False,
    db_path=None,
    hybrid_result=None,
):
    """Store an analysis, omitting message content unless explicitly consented."""
    if int(predicted_label) not in (0, 1):
        raise ValueError("predicted_label phải là 0 hoặc 1.")
    message = str(message or "")
    message_text = message if save_content else None
    message_hash = hashlib.sha256(message.encode("utf-8")).hexdigest() if save_hash else None
    hybrid_result = hybrid_result or {}
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_history
            (created_at, model_name, predicted_label, prediction_score, score_type,
             message_length, message_text, message_hash, content_saved,
             hybrid_label, risk_level, rule_override, needs_review,
             risk_reasons, risk_signals, risk_urls, risk_evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                str(model_name),
                int(predicted_label),
                float(prediction_score) if prediction_score is not None else None,
                str(score_type),
                len(message),
                message_text,
                message_hash,
                int(bool(save_content)),
                str(hybrid_result.get("final_label") or ""),
                str(hybrid_result.get("risk_level") or ""),
                int(bool(hybrid_result.get("rule_override"))),
                int(bool(hybrid_result.get("needs_review"))),
                json.dumps(hybrid_result.get("reasons", []), ensure_ascii=False),
                json.dumps(hybrid_result.get("signals", []), ensure_ascii=False),
                json.dumps(hybrid_result.get("urls", []), ensure_ascii=False),
                json.dumps(hybrid_result.get("evidence", []), ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid)


def get_analysis(analysis_id, db_path=None):
    initialize_database(db_path)
    with _connection(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM analysis_history WHERE id = ?", (int(analysis_id),)
        ).fetchone()
    return dict(row) if row else None


def count_history(search="", label=None, model_name=None, date_from=None, date_to=None, db_path=None):
    clauses, params = _date_clauses(date_from, date_to)
    if search:
        clauses.append("(message_text LIKE ? OR model_name LIKE ? OR message_hash LIKE ?)")
        pattern = f"%{search}%"
        params.extend([pattern, pattern, pattern])
    if label in (0, 1):
        clauses.append("predicted_label = ?")
        params.append(int(label))
    if model_name:
        clauses.append("model_name = ?")
        params.append(model_name)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(db_path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM analysis_history{where}", params).fetchone()[0])


def list_history(
    search="",
    label=None,
    model_name=None,
    date_from=None,
    date_to=None,
    page=1,
    page_size=20,
    db_path=None,
):
    total = count_history(search, label, model_name, date_from, date_to, db_path)
    clauses, params = _date_clauses(date_from, date_to)
    if search:
        clauses.append("(message_text LIKE ? OR model_name LIKE ? OR message_hash LIKE ?)")
        pattern = f"%{search}%"
        params.extend([pattern, pattern, pattern])
    if label in (0, 1):
        clauses.append("predicted_label = ?")
        params.append(int(label))
    if model_name:
        clauses.append("model_name = ?")
        params.append(model_name)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    page = max(1, int(page))
    page_size = min(100, max(1, int(page_size)))
    params.extend([page_size, (page - 1) * page_size])
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT * FROM analysis_history{where} ORDER BY id DESC LIMIT ? OFFSET ?", params
        ).fetchall()
    return [dict(row) for row in rows], total


def delete_analysis(analysis_id, db_path=None):
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM analysis_history WHERE id = ?", (int(analysis_id),))
        return cursor.rowcount > 0


def clear_history(db_path=None):
    with _connection(db_path) as connection:
        connection.execute("DELETE FROM analysis_history")


def activity_summary(date_from=None, date_to=None, db_path=None):
    clauses, params = _date_clauses(date_from, date_to)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(db_path) as connection:
        row = connection.execute(
            f"""
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(CASE WHEN predicted_label = 0 THEN 1 ELSE 0 END), 0) AS valid,
                   COALESCE(SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END), 0) AS suspicious
            FROM analysis_history{where}
            """,
            params,
        ).fetchone()
        feedback_clauses, feedback_params = _date_clauses(date_from, date_to, "f.created_at")
        feedback_where = f" WHERE {' AND '.join(feedback_clauses)}" if feedback_clauses else ""
        feedback = connection.execute(
            f"SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN review_status = 'pending' THEN 1 ELSE 0 END), 0) AS pending FROM prediction_feedback f{feedback_where}",
            feedback_params,
        ).fetchone()
    total = int(row["total"])
    return {
        "total": total,
        "valid": int(row["valid"]),
        "suspicious": int(row["suspicious"]),
        "suspicious_rate": (int(row["suspicious"]) / total) if total else 0.0,
        "feedback_total": int(feedback["total"]),
        "feedback_pending": int(feedback["pending"]),
    }


def activity_series(date_from=None, date_to=None, db_path=None):
    clauses, params = _date_clauses(date_from, date_to)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT substr(created_at, 1, 10) AS day,
                   COUNT(*) AS total,
                   SUM(CASE WHEN predicted_label = 0 THEN 1 ELSE 0 END) AS valid,
                   SUM(CASE WHEN predicted_label = 1 THEN 1 ELSE 0 END) AS suspicious
            FROM analysis_history{where}
            GROUP BY substr(created_at, 1, 10)
            ORDER BY day
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def label_counts(date_from=None, date_to=None, db_path=None):
    clauses, params = _date_clauses(date_from, date_to)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT predicted_label AS label, COUNT(*) AS total FROM analysis_history{where} GROUP BY predicted_label ORDER BY predicted_label",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def model_counts(date_from=None, date_to=None, db_path=None):
    clauses, params = _date_clauses(date_from, date_to)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT model_name, COUNT(*) AS total FROM analysis_history{where} GROUP BY model_name ORDER BY total DESC, model_name",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def add_feedback(analysis_id, feedback_type, suggested_label=None, comment="", db_path=None, suggested_label_code=None, feedback_reason=""):
    if feedback_type not in {"correct", "incorrect", "uncertain"}:
        raise ValueError("feedback_type không hợp lệ.")
    if suggested_label not in (None, 0, 1, 2):
        raise ValueError("suggested_label không hợp lệ.")
    allowed_codes = {"LEGITIMATE", "PROMOTION", "PHISHING", "FINANCIAL_SCAM", "NEEDS_REVIEW"}
    if suggested_label_code is not None and suggested_label_code not in allowed_codes:
        raise ValueError("suggested_label_code không hợp lệ.")
    with _connection(db_path) as connection:
        exists = connection.execute("SELECT 1 FROM analysis_history WHERE id = ?", (int(analysis_id),)).fetchone()
        if not exists:
            raise ValueError("Không tìm thấy bản ghi phân tích.")
        duplicate = connection.execute("SELECT 1 FROM prediction_feedback WHERE analysis_id = ? AND COALESCE(review_state, 'pending_review') NOT IN ('rejected', 'duplicate') LIMIT 1", (int(analysis_id),)).fetchone()
        review_state = "duplicate" if duplicate else "pending_review"
        legacy_status = "rejected" if duplicate else "pending"
        cursor = connection.execute(
            """
            INSERT INTO prediction_feedback
            (analysis_id, feedback_type, suggested_label, suggested_label_code, comment, feedback_reason, created_at, review_status, review_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(analysis_id), feedback_type, suggested_label, suggested_label_code, str(comment or ""), str(feedback_reason or ""), _now(), legacy_status, review_state),
        )
        return int(cursor.lastrowid)


def list_feedback(status=None, search="", page=1, page_size=50, db_path=None):
    clauses = []
    params = []
    if status in {"pending", "reviewed", "rejected"}:
        clauses.append("f.review_status = ?")
        params.append(status)
    elif status in {"pending_review", "verified", "duplicate"}:
        clauses.append("COALESCE(f.review_state, 'pending_review') = ?")
        params.append(status)
    if search:
        clauses.append("(f.comment LIKE ? OR h.model_name LIKE ? OR h.message_text LIKE ?)")
        pattern = f"%{search}%"
        params.extend([pattern, pattern, pattern])
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    page = max(1, int(page))
    page_size = min(100, max(1, int(page_size)))
    count_params = list(params)
    params.extend([page_size, (page - 1) * page_size])
    with _connection(db_path) as connection:
        total = int(connection.execute(
            f"SELECT COUNT(*) FROM prediction_feedback f JOIN analysis_history h ON h.id = f.analysis_id{where}", count_params
        ).fetchone()[0])
        rows = connection.execute(
            f"""
            SELECT f.*, h.created_at AS analysis_created_at, h.model_name,
                   h.predicted_label, h.message_text, h.message_length,
                   h.hybrid_label, h.risk_level, h.rule_override, h.risk_reasons,
                   h.risk_signals, h.risk_urls, h.risk_evidence
            FROM prediction_feedback f
            JOIN analysis_history h ON h.id = f.analysis_id{where}
            ORDER BY f.id DESC LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows], total


def update_feedback_status(feedback_id, status, db_path=None):
    if status not in {"pending", "reviewed", "rejected", "pending_review", "verified", "duplicate"}:
        raise ValueError("Trạng thái feedback không hợp lệ.")
    state = {"pending": "pending_review", "reviewed": "verified", "rejected": "rejected"}.get(status, status)
    legacy_status = {"pending_review": "pending", "verified": "reviewed", "rejected": "rejected", "duplicate": "rejected"}[state]
    reviewed_at = _now() if state not in {"pending_review"} else None
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE prediction_feedback SET review_status = ?, review_state = ?, reviewed_at = ? WHERE id = ?",
            (legacy_status, state, reviewed_at, int(feedback_id)),
        )
        return cursor.rowcount > 0


def history_csv(rows):
    """Export only the rows already selected by the UI; no implicit DB dump."""
    fields = [
        "id", "created_at", "model_name", "predicted_label", "prediction_score",
        "score_type", "message_length", "message_text", "message_hash", "content_saved",
        "hybrid_label", "risk_level", "rule_override", "needs_review", "risk_reasons", "risk_signals", "risk_urls", "risk_evidence",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def feedback_csv(rows):
    fields = [
        "id", "analysis_id", "feedback_type", "suggested_label", "suggested_label_code", "comment", "feedback_reason", "created_at",
        "review_status", "review_state", "reviewed_at", "analysis_created_at", "model_name", "predicted_label",
        "message_text", "message_length",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")
