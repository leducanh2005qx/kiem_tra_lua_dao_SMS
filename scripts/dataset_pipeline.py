"""Build and audit the Phase 1 four-class dataset artifacts.

Examples:
    python scripts/dataset_pipeline.py init
    python scripts/dataset_pipeline.py audit
    python scripts/dataset_pipeline.py import --input file.jsonl --source-name vnphish ...

The pipeline never maps binary labels to four classes automatically.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_engineering import (  # noqa: E402
    FOUR_CLASS_LABELS,
    QUEUE_COLUMNS,
    audit_dataset,
    build_processed,
    import_records,
    merge_candidate_files,
    write_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ANNOTATIONS = ROOT / "data" / "annotations"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MANIFESTS = ROOT / "data" / "manifests"

SOURCES = {
    "official_binary": {
        "source_name": "trannguyenthaituan/vietnamese_sms_dataset",
        "source_url": "https://huggingface.co/datasets/trannguyenthaituan/vietnamese_sms_dataset",
        "license": "CC BY 4.0",
        "source_type": "real",
        "files": {
            "train.csv": "https://huggingface.co/datasets/trannguyenthaituan/vietnamese_sms_dataset/resolve/main/train.csv",
            "test.csv": "https://huggingface.co/datasets/trannguyenthaituan/vietnamese_sms_dataset/resolve/main/test.csv",
        },
    },
    "fraud_detection": {
        "source_name": "tocong282/vietnamese-fraud-detection",
        "source_url": "https://github.com/tocong282/vietnamese-fraud-detection",
        "license": "UNVERIFIED_DATA_LICENSE",
        "source_type": "heuristic-derived",
        "files": {},
    },
    "vnphish": {
        "source_name": "wikiepeidia/vnphish-dataset",
        "source_url": "https://huggingface.co/datasets/wikiepeidia/vnphish-dataset",
        "license": "CC BY-NC 4.0",
        "source_type": "synthetic",
        "files": {},
    },
}


def _write_empty_templates() -> None:
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame(columns=QUEUE_COLUMNS)
    template.to_csv(ANNOTATIONS / "annotation_template.csv", index=False, encoding="utf-8-sig")
    if not (ANNOTATIONS / "annotation_queue.csv").exists():
        template.to_csv(ANNOTATIONS / "annotation_queue.csv", index=False, encoding="utf-8-sig")
    for directory in [RAW / "official_binary", RAW / "four_class_candidates", RAW / "synthetic", INTERIM, PROCESSED, MANIFESTS]:
        directory.mkdir(parents=True, exist_ok=True)
    for filename in ["train.csv", "validation.csv", "test.csv", "gold_test.csv", "robustness_test.csv"]:
        target = PROCESSED / filename
        if not target.exists():
            pd.DataFrame(columns=["sample_id", "message", "message_anonymized", "label", "label_id", "source_name", "source_type", "source_url", "license", "original_label", "raw_label", "group_id", "pii_status", "split"]).to_csv(target, index=False, encoding="utf-8-sig")


def _snapshot_official_binary() -> list[Path]:
    copied = []
    for filename in ("train.csv", "test.csv"):
        source = RAW / filename
        target = RAW / "official_binary" / filename
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def init_pipeline() -> dict:
    _write_empty_templates()
    copied = _snapshot_official_binary()
    source_manifest = []
    for key, source in SOURCES.items():
        source_manifest.append({"key": key, **{k: v for k, v in source.items() if k != "files"}, "files": source["files"], "status": "local_snapshot" if key == "official_binary" and (RAW / "official_binary" / "train.csv").exists() else "not_imported"})
    write_source_manifest(MANIFESTS / "source_manifest.json", source_manifest)
    return {"created": True, "snapshot_files": [str(p) for p in copied], "source_manifest": str(MANIFESTS / "source_manifest.json")}


def import_source(args) -> dict:
    output = Path(args.output) if args.output else INTERIM / f"{args.source_name.replace('/', '_')}.csv"
    return import_records(args.input, output, args.source_name, args.source_type, args.source_url, args.license)


def rebuild_queue() -> dict:
    candidate_files = sorted(INTERIM.glob("*.csv"))
    if not candidate_files:
        raise SystemExit("Chưa có candidate. Hãy chạy init hoặc import trước.")
    merged = merge_candidate_files(candidate_files, ANNOTATIONS / "annotation_queue.csv")
    return {"candidate_files": [str(p) for p in candidate_files], "rows": len(merged), "queue": str(ANNOTATIONS / "annotation_queue.csv")}


def migrate_annotation_schema() -> dict:
    """Add annotation fields without changing SMS content or existing labels."""
    path = ANNOTATIONS / "annotation_queue.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    added = []
    for column in QUEUE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
            added.append(column)
    frame[QUEUE_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    return {"queue": str(path), "rows": len(frame), "added_columns": added}


def download_source(args) -> dict:
    source = SOURCES[args.source]
    if not source["files"]:
        raise SystemExit("Nguồn này không có URL file ổn định trong pipeline. Hãy tải thủ công/repository rồi dùng lệnh import.")
    target_dir = RAW / args.source
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename, url in source["files"].items():
        target = target_dir / filename
        if target.exists() and target.stat().st_size:
            downloaded.append({"file": str(target), "status": "exists_not_overwritten"})
            continue
        request = Request(url, headers={"User-Agent": "SMS-Shield-VN-Phase1/1.0"})
        with urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
        downloaded.append({"file": str(target), "status": "downloaded"})
    return {"source": args.source, "files": downloaded}


def main() -> None:
    # Windows PowerShell may expose a cp1252 stdout; reports and JSON contain
    # Vietnamese text, so keep the CLI output UTF-8 when possible.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 1 provenance-preserving dataset pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Tạo cấu trúc và snapshot nguồn binary local, không ghi đè")
    sub.add_parser("rebuild-queue", help="Gộp các candidate interim thành annotation queue")
    sub.add_parser("migrate-schema", help="Bổ sung cột annotation mới mà không đổi nội dung")
    sub.add_parser("audit", help="Xuất thống kê, duplicate, label và split reports")
    sub.add_parser("prepare", help="Xuất processed splits chỉ từ dòng VERIFIED")
    download = sub.add_parser("download", help="Tải nguồn có URL file ổn định, không dùng credential")
    download.add_argument("--source", choices=sorted(SOURCES), required=True)
    imp = sub.add_parser("import", help="Nhập CSV/JSON/JSONL mà không suy ra nhãn bốn lớp")
    imp.add_argument("--input", required=True)
    imp.add_argument("--output")
    imp.add_argument("--source-name", required=True)
    imp.add_argument("--source-type", choices=sorted({"real", "synthetic", "heuristic-derived", "manually-verified"}), required=True)
    imp.add_argument("--source-url", required=True)
    imp.add_argument("--license", required=True)
    args = parser.parse_args()

    if args.command == "init":
        result = init_pipeline()
    elif args.command == "download":
        result = download_source(args)
    elif args.command == "import":
        result = import_source(args)
    elif args.command == "rebuild-queue":
        result = rebuild_queue()
    elif args.command == "migrate-schema":
        result = migrate_annotation_schema()
    elif args.command == "prepare":
        result = build_processed(ANNOTATIONS / "annotation_queue.csv", PROCESSED)
    elif args.command == "audit":
        result = audit_dataset(ANNOTATIONS / "annotation_queue.csv", MANIFESTS, PROCESSED)
    else:
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
