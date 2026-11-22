import json

import numpy as np
import pandas as pd

from src.scam_training import audit_scam_queue, predict_with_policy, train_verified_model


def test_scam_audit_does_not_infer_legacy_binary_labels(tmp_path):
    path = tmp_path / "queue.csv"
    pd.DataFrame([
        {"sample_id": "1", "message": "OTP hợp lệ", "final_label": "", "review_status": "UNMAPPED", "group_id": "g1"},
        {"sample_id": "2", "message": "Chuyển phí trước", "final_label": "1", "review_status": "UNMAPPED", "group_id": "g2"},
    ]).to_csv(path, index=False, encoding="utf-8-sig")
    result = audit_scam_queue(path)
    assert result["verified_rows_after_dedup"] == 0
    assert result["ready_for_training"] is False


def test_prediction_policy_has_three_explicit_states():
    class Model:
        def predict_proba(self, values):
            return np.array([[0.1, 0.9] for _ in values])

    result = predict_with_policy(Model(), "message", {"confidence_threshold": 0.5, "margin_threshold": 0.0})
    assert result["label"] == "SCAM"
    assert result["score_type"] == "calibrated_probability"
    uncertain = predict_with_policy(Model(), "message", {"confidence_threshold": 0.95, "margin_threshold": 0.0})
    assert uncertain["label"] == "NEEDS_REVIEW"


def test_verified_training_writes_separate_artifact_and_reports(tmp_path):
    rows = []
    for index in range(60):
        scam = index % 2
        rows.append({
            "sample_id": str(index),
            "message": f"{'Chuyen phi xac minh tai khoan' if scam else 'Thong bao giao dich hop le'} mau {index}",
            "final_label": "SCAM" if scam else "NOT_SCAM",
            "review_status": "VERIFIED",
            "group_id": f"group_{index}",
            "source_name": "test",
        })
    queue = tmp_path / "queue.csv"
    pd.DataFrame(rows).to_csv(queue, index=False, encoding="utf-8-sig")
    result = train_verified_model(queue, tmp_path / "models", tmp_path / "results", tmp_path / "TRAINING_REPORT.md")
    assert result["status"] == "completed"
    assert (tmp_path / "models" / "scam_verified" / "model.joblib").exists()
    assert json.loads((tmp_path / "results" / "training_report.json").read_text(encoding="utf-8"))["status"] == "completed"
