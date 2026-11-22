"""Audit and train the verified SCAM/NOT_SCAM model without touching legacy artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scam_training import audit_scam_queue, train_verified_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "data" / "annotations" / "annotation_queue.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "TRAINING_REPORT.md"
READINESS_REPORT = ROOT / "docs" / "TRAINING_READINESS_AUDIT.md"


def write_audit_reports(result: dict) -> None:
    payload = {"status": "blocked", "task": "SCAM vs NOT_SCAM", "audit": result, "model_artifact": None}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "training_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    reasons = ", ".join(result["blocking_reasons"]) or "unknown"
    READINESS_REPORT.write_text(
        "# Training Readiness Audit\n\n"
        "**BLOCKED – chưa đủ dữ liệu SCAM/NOT_SCAM đã được người xác minh.**\n\n"
        f"- Candidate trong queue: **{result['queue_rows']}**\n"
        f"- Verified trước loại trùng: **{result['verified_rows_before_dedup']}**\n"
        f"- Verified sau loại trùng: **{result['verified_rows_after_dedup']}**\n"
        f"- SCAM verified: **{result['verified_label_counts'].get('SCAM', 0)}**\n"
        f"- NOT_SCAM verified: **{result['verified_label_counts'].get('NOT_SCAM', 0)}**\n"
        f"- NEEDS_REVIEW: **{result['needs_review_rows']}**\n"
        f"- Annotator agreement: **chưa có hai bộ nhãn hợp lệ**\n"
        f"- Lý do chặn: `{reasons}`\n\n"
        "Nhãn nhị phân cũ `0/1` chỉ mô tả hợp lệ/rác-nghi vấn và không được tự động chuyển thành nhãn scam.\n\n"
        "Gán nhãn theo `docs/SCAM_LABELING_GUIDE.md`, sau đó chạy lại `python scripts/train_scam_model.py audit`.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verified SCAM/NOT_SCAM training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Kiểm tra số lượng và chất lượng nhãn đã xác minh")
    audit.add_argument("--queue", type=Path, default=QUEUE)
    train = sub.add_parser("train", help="Huấn luyện nếu dataset verified đủ điều kiện")
    train.add_argument("--queue", type=Path, default=QUEUE)
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "audit":
        result = audit_scam_queue(args.queue)
        write_audit_reports(result)
    else:
        result = train_verified_model(args.queue, MODELS, RESULTS, REPORT, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.command == "train" and result.get("status") == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
