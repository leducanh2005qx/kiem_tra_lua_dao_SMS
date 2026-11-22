"""Train the separate hierarchical scam detection artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ANNOTATION_QUEUE, MODEL_DIR, RESULTS_DIR
from src.hierarchical_training import train_hierarchical_models


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình NOT_SCAM/SCAM và loại lừa đảo")
    parser.add_argument("--queue", type=Path, default=ANNOTATION_QUEUE)
    parser.add_argument("--models-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "hierarchical")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "HIERARCHICAL_TRAINING_REPORT.md")
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_hierarchical_models(args.queue, args.models_dir, args.results_dir, args.report, seed=args.seed, min_per_class=args.min_per_class)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("status") == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
