"""Compatibility entry point for the strict three-model four-class trainer."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multiclass_training import train_three_models


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(root / "data" / "processed"))
    parser.add_argument("--lock", default=str(root / "data" / "manifests" / "dataset_lock.json"))
    parser.add_argument("--models-dir", default=str(root / "models"))
    parser.add_argument("--results-dir", default=str(root / "results" / "multiclass"))
    parser.add_argument("--report", default=str(root / "docs" / "MULTICLASS_TRAINING_REPORT.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_three_models(args.processed_dir, args.lock, args.models_dir, args.results_dir, args.report, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("status") == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
