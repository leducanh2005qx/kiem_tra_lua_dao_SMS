import pandas as pd

from src.hierarchical import LAYER1_LABELS, SCAM_TYPE_LABELS, validate_hierarchical_frame
from src.hierarchical_runtime import load_hierarchical_models, predict_hierarchical
from src.hierarchical_training import train_hierarchical_models


def _queue(path, repeats=6):
    rows = []
    examples = [
        ("NOT_SCAM", "", "LEGITIMATE", "Thông báo giao dịch thành công hợp lệ"),
        ("NOT_SCAM", "", "PROMOTION", "Khuyến mãi giảm giá sản phẩm hôm nay"),
        ("SCAM", "PHISHING", "", "Cung cấp OTP để xác minh tài khoản"),
        ("SCAM", "FINANCIAL_SCAM", "", "Chuyển phí trước để nhận thưởng"),
        ("SCAM", "OTHER_SCAM", "", "Việc làm tại nhà yêu cầu nộp tiền"),
    ]
    for layer1, layer2, non_scam, text in examples:
        for index in range(repeats):
            rows.append({
                "sample_id": f"{layer1}_{layer2}_{non_scam}_{index}",
                "message": f"{text} mẫu {index}",
                "layer1_label": layer1,
                "layer2_label": layer2,
                "non_scam_type": non_scam,
                "verified": "1",
                "review_status": "VERIFIED",
                "group_id": f"group_{layer1}_{layer2}_{non_scam}_{index}",
            })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_hierarchical_validation_requires_both_layers():
    frame = pd.DataFrame([
        {"message": "a", "layer1_label": "SCAM", "layer2_label": "PHISHING", "verified": "1", "review_status": "VERIFIED"},
        {"message": "b", "layer1_label": "NOT_SCAM", "layer2_label": "", "verified": "1", "review_status": "VERIFIED"},
    ])
    report = validate_hierarchical_frame(frame, min_per_class=1)
    assert report["ready_for_training"] is False
    assert set(report["layer1_counts"]) == set(LAYER1_LABELS)
    assert set(report["layer2_counts"]) == set(SCAM_TYPE_LABELS)


def test_hierarchical_training_creates_separate_artifacts(tmp_path):
    queue = tmp_path / "queue.csv"
    _queue(queue)
    result = train_hierarchical_models(queue, tmp_path / "models", tmp_path / "results", tmp_path / "report.md", min_per_class=4)
    assert result["status"] == "completed"
    loaded, metadata = load_hierarchical_models(tmp_path / "models")
    assert set(loaded) == {"nb_baseline", "nb_optimized", "svm"}
    prediction = predict_hierarchical(loaded, metadata, "Cung cấp OTP để xác minh tài khoản", "svm", threshold=0.0)
    assert prediction["layer1_label"] in LAYER1_LABELS
    assert prediction["layer2_label"] in {"", *SCAM_TYPE_LABELS}
