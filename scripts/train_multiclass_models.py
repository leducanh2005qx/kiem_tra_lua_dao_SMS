"""Train the three strict four-class classical model artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multiclass_training import train_three_models  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NB baseline, optimized NB and calibrated Linear SVM on verified four-class data")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--lock", type=Path, default=ROOT / "data" / "manifests" / "dataset_lock.json")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "multiclass")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "MULTICLASS_TRAINING_REPORT.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_three_models(args.processed_dir, args.lock, args.models_dir, args.results_dir, args.report, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("status") == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
