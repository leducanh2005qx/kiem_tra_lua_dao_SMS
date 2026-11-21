"""Train the optional four-class baseline on human-annotated data only."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multiclass import train_multiclass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", default="models/multiclass")
    args = parser.parse_args()
    print(json.dumps(train_multiclass(args.csv_path, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
