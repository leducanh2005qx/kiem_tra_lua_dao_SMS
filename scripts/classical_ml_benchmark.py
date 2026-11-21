"""Run or lock the Phase 2 four-class classical ML benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classical_ml import lock_dataset, run_benchmark  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
LOCK = ROOT / "data" / "manifests" / "dataset_lock.json"
RESULTS = ROOT / "results" / "classical_ml"
REPORT = ROOT / "docs" / "CLASSICAL_ML_REPORT.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 classical ML benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock", help="Khóa split bốn lớp sau khi verified và kiểm tra leakage")
    lock.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("run", help="Chọn bằng validation và chỉ sau đó đánh giá test")
    run.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "lock":
        result = lock_dataset(PROCESSED, LOCK, seed=args.seed)
    else:
        result = run_benchmark(PROCESSED, LOCK, RESULTS, REPORT, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.command == "lock" and not result.get("locked"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
