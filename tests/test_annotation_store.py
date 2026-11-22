import pandas as pd

from src.annotation_store import (
    _connection,
    annotation_stats,
    export_annotations,
    initialize_annotation_store,
    list_annotations,
    quality_summary,
    save_annotation,
)
from src.dataset_engineering import QUEUE_COLUMNS


def _queue(path):
    rows = []
    for index, message in enumerate(["OTP hợp lệ", "Cung cấp OTP để xác minh", "Cung cấp OTP để xác minh"]):
        row = {column: "" for column in QUEUE_COLUMNS}
        row.update({"sample_id": f"sms_{index}", "message": message, "source_name": "test", "source_type": "real", "original_label": str(index % 2), "review_status": "UNMAPPED", "group_id": f"group_{index}"})
        rows.append(row)
    pd.DataFrame(rows, columns=QUEUE_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def test_annotation_persists_labels_reviews_and_queue_export(tmp_path):
    queue = tmp_path / "queue.csv"
    db = tmp_path / "annotations.db"
    _queue(queue)
    assert initialize_annotation_store(db, queue)["inserted_rows"] == 3
    save_annotation(db, queue, "sms_0", "LEGITIMATE", "Thông báo OTP hợp lệ", annotator_id="a1", confidence="high")
    save_annotation(db, queue, "sms_1", "NEEDS_REVIEW", "Thiếu ngữ cảnh xác minh", annotator_id="a1")
    assert annotation_stats(db) == {"total": 3, "verified": 1, "unlabeled": 1, "needs_review": 1, "machine_suggestions": 0, "verified_by_label": {"LEGITIMATE": 1, "PROMOTION": 0, "PHISHING": 0, "FINANCIAL_SCAM": 0}}
    rows, total = list_annotations(db, "verified", page=1, page_size=10)
    assert total == 1 and rows[0]["sample_id"] == "sms_0"
    assert quality_summary(db)["duplicate_groups"] == 1
    initialize_annotation_store(db, queue)
    assert annotation_stats(db)["verified"] == 1
    assert "LEGITIMATE" in queue.read_text(encoding="utf-8-sig")
    assert export_annotations(db, queue) == 3


def test_sqlite_wal_and_optimistic_revision_prevent_stale_overwrite(tmp_path):
    queue = tmp_path / "queue.csv"
    db = tmp_path / "annotations.db"
    _queue(queue)
    initialize_annotation_store(db, queue)
    with _connection(db) as connection:
        pragmas = {name: connection.execute(f"PRAGMA {name}").fetchone()[0] for name in ("journal_mode", "synchronous", "busy_timeout")}
        revision = connection.execute("SELECT revision FROM annotation_items WHERE sample_id = 'sms_0'").fetchone()[0]
    assert str(pragmas["journal_mode"]).lower() == "wal"
    assert int(pragmas["busy_timeout"]) == 30000

    save_annotation(db, queue, "sms_0", "LEGITIMATE", "Người thứ nhất xác nhận", expected_revision=revision)
    try:
        save_annotation(db, queue, "sms_0", "PROMOTION", "Người thứ hai dùng dữ liệu cũ", expected_revision=revision)
    except ValueError as exc:
        assert "người khác" in str(exc)
    else:
        raise AssertionError("Nhãn cũ không được phép ghi đè nhãn mới")
    rows, _ = list_annotations(db, "verified", page=1, page_size=10)
    assert rows[0]["final_label"] == "LEGITIMATE"
