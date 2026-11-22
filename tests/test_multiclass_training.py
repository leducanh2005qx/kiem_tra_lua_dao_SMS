import json

import pandas as pd

from src.classical_ml import lock_dataset
from src.dataset_engineering import FOUR_CLASS_LABELS
from src.multiclass_runtime import load_multiclass_models, predict_multiclass
from src.multiclass_training import train_three_models


def _write_split(path, split, offset):
    rows = []
    examples = {
        "LEGITIMATE": "thong bao giao dich hop le",
        "PROMOTION": "uu dai giam gia san pham",
        "PHISHING": "cung cap otp de xac minh tai khoan",
        "FINANCIAL_SCAM": "chuyen phi truoc de nhan thuong",
    }
    for class_index, label in enumerate(FOUR_CLASS_LABELS):
        for index in range(3):
            rows.append({
                "sample_id": f"{split}_{class_index}_{index}",
                "message": f"{examples[label]} {split} {offset + index}",
                "message_anonymized": "",
                "label": label,
                "label_id": FOUR_CLASS_LABELS[label],
                "source_name": "test_source",
                "source_type": "real",
                "source_url": "test",
                "license": "test",
                "original_label": "",
                "raw_label": "",
                "group_id": f"{split}_group_{class_index}_{index}",
                "pii_status": "safe",
                "split": split,
            })
    pd.DataFrame(rows).to_csv(path / f"{split}.csv", index=False, encoding="utf-8-sig")


def test_multiclass_trainer_creates_three_models_and_reject_output(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_split(processed, "train", 0)
    _write_split(processed, "validation", 100)
    _write_split(processed, "test", 200)
    lock = tmp_path / "dataset_lock.json"
    assert lock_dataset(processed, lock)["locked"] is True
    result = train_three_models(processed, lock, tmp_path / "models", tmp_path / "results", tmp_path / "report.md")
    assert result["status"] == "completed"
    loaded, metadata = load_multiclass_models(tmp_path / "models")
    assert set(loaded) == {"nb_baseline", "nb_optimized", "svm"}
    assert metadata["label_mapping"] == FOUR_CLASS_LABELS
    prediction = predict_multiclass(loaded["svm"], metadata, "Cung cap OTP de xac minh tai khoan", "svm")
    assert prediction["predicted_label"] in FOUR_CLASS_LABELS
    assert prediction["label"] in {*FOUR_CLASS_LABELS, "NEEDS_REVIEW"}
    assert json.loads((tmp_path / "results" / "training_report.json").read_text(encoding="utf-8"))["num_classes"] == 4
