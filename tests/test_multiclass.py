import pandas as pd

from src.multiclass import MULTICLASS_LABELS, create_annotation_template, validate_multiclass_frame


def test_multiclass_schema_does_not_convert_binary_labels():
    frame = pd.DataFrame([
        {"message": "a", "label": "LEGITIMATE", "annotation_status": "certain"},
        {"message": "b", "label": "", "annotation_status": "uncertain"},
    ])
    report = validate_multiclass_frame(frame)
    assert report["valid"] is True
    assert validate_multiclass_frame(frame, require_all_classes=True)["ready_for_training"] is False
    assert set(MULTICLASS_LABELS) == {"LEGITIMATE", "PROMOTION", "PHISHING", "FINANCIAL_SCAM"}


def test_multiclass_template_has_no_fake_rows(tmp_path):
    path = create_annotation_template(tmp_path / "annotations.csv")
    loaded = pd.read_csv(path)
    assert len(loaded) == 0
    assert {"message", "label", "annotation_status"}.issubset(loaded.columns)
