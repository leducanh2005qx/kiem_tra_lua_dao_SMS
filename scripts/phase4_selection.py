"""Phase 4 model registry, calibration and production-selection gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase4 import LABEL2ID, build_robustness_frame, create_registry, discover_model_candidates, sha256_file  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry" / "model_registry.json"
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "FINAL_MODEL_SELECTION_REPORT.md"


def initialize_outputs() -> None:
    (RESULTS / "calibration").mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness").mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark").mkdir(parents=True, exist_ok=True)
    tables = {
        RESULTS / "final_model_comparison.csv": ["model_name", "macro_f1", "weighted_f1", "high_risk_recall", "high_risk_precision", "high_risk_fpr", "robustness_score", "brier_score", "expected_calibration_error", "inference_latency_ms", "model_size_bytes", "status"],
        RESULTS / "calibration" / "calibration_metrics.csv": ["model_name", "split", "stage", "brier_score", "expected_calibration_error", "log_loss"],
        RESULTS / "robustness" / "robustness_results.csv": ["model_name", "variant", "rows", "accuracy", "macro_f1", "high_risk_recall", "status"],
        RESULTS / "robustness" / "robustness_errors.csv": ["model_name", "sample_id", "variant", "message", "true_label", "predicted_label"],
        RESULTS / "selective_risk.csv": ["model_name", "split", "confidence_threshold", "margin_threshold", "coverage", "selective_risk", "review_rate", "status"],
    }
    for path, columns in tables.items():
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def write_report(registry: dict, reasons: list[str]) -> None:
    REPORT.write_text(
        "# Final Model Selection Report – Phase 4\n\n"
        "## Trạng thái\n\n"
        "**BLOCKED – chưa có artifact bốn lớp hợp lệ để calibration, robustness hoặc production selection.**\n\n"
        "Không có model nào được đánh dấu production. Model nhị phân hiện tại trên website vẫn được giữ nguyên.\n\n"
        f"Lý do: `{', '.join(reasons)}`.\n\n"
        "## Quy tắc đã triển khai\n\n"
        "- Chỉ artifact có manifest, label mapping, dataset/preprocessing version, training config và metrics mới được registry.\n"
        "- Temperature scaling chỉ được fit trên validation/calibration split, không dùng test.\n"
        "- Confidence và top-1/top-2 margin được chọn trên validation; trạng thái REVIEW không phải lớp thứ năm.\n"
        "- Robustness dùng test biến thể riêng và không được dùng để điều chỉnh model/threshold.\n"
        "- Production ưu tiên Macro-F1, high-risk recall/precision, false alarm, calibration, selective risk, latency và kích thước.\n\n"
        "## Kết quả hiện tại\n\n"
        "Chưa có metric calibration, robustness, latency hoặc model comparison hợp lệ. Không bịa số liệu khi Phase 2/3 chưa tạo artifact bốn lớp.\n\n"
        "## Khi đủ điều kiện\n\n"
        "Chạy `python scripts/phase4_selection.py audit`, sau đó chạy các benchmark Phase 2 và notebook Phase 3 trên cùng dataset lock. Chỉ sau khi toàn bộ unit/integration/model-loading/reject-option tests đạt mới sửa registry để có `production_model`.\n",
        encoding="utf-8",
    )


def audit() -> dict:
    initialize_outputs()
    registry = create_registry(ROOT, REGISTRY)
    reasons = []
    if not registry["models"]:
        reasons.append("no_complete_four_class_model_artifact")
    lock = ROOT / "data" / "manifests" / "dataset_lock.json"
    if not lock.exists():
        reasons.append("dataset_lock_missing")
    classical = RESULTS / "classical_ml" / "test_metrics.csv"
    if not classical.exists() or classical.stat().st_size <= 1:
        reasons.append("classical_test_metrics_missing")
    phobert = ROOT / "models" / "phobert_4class"
    if not phobert.exists():
        reasons.append("phobert_artifact_missing")
    registry["selection_status"] = "blocked" if reasons else registry["selection_status"]
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(registry, reasons)
    return {"status": "blocked" if reasons else "awaiting_evaluation", "reasons": reasons, "registry": str(REGISTRY), "report": str(REPORT)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit"], help="audit artifacts and create the selection gate")
    args = parser.parse_args()
    print(json.dumps(audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
