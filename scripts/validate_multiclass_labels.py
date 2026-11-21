"""Validate an explicitly annotated four-class CSV without training a model."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multiclass import load_multiclass_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    _, report = load_multiclass_csv(args.csv_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
