import numpy as np

from src.phase4 import (
    LABEL2ID,
    TemperatureScaler,
    expected_calibration_error,
    robustness_variants,
    select_reject_policy,
    unified_prediction,
)


def test_temperature_scaling_is_validation_only_and_outputs_probabilities():
    logits = np.array([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    labels = np.array([0, 1, 2, 3])
    scaler = TemperatureScaler().fit(logits, labels, split_name="validation")
    probabilities = scaler.predict_proba(logits)
    assert probabilities.shape == (4, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    try:
        TemperatureScaler().fit(logits, labels, split_name="test")
    except ValueError:
        pass
    else:
        raise AssertionError("test must not fit temperature scaling")


def test_reject_policy_is_selected_from_validation_and_interface_is_explicit():
    probabilities = np.array([[.8, .1, .05, .05], [.25, .25, .4, .1], [.1, .1, .1, .7]])
    labels = np.array([0, 1, 3])
    policy = select_reject_policy(labels, probabilities)
    result = unified_prediction(2, probabilities[1], "calibrated_probability", "test", "v1", policy)
    assert result["predicted_class"] == "PHISHING"
    assert result["score_type"] == "calibrated_probability"
    assert result["status"] in {"HIGH_RISK", "REVIEW"}
    assert set(result["class_scores"]) == set(LABEL2ID)


def test_robustness_variants_do_not_change_label_schema():
    variants = robustness_variants("Ngân hàng: https://example.com")
    assert "without_diacritics" in variants
    assert variants["original"] == "Ngân hàng: https://example.com"
    assert expected_calibration_error(np.array([0]), np.array([[1.0, 0.0, 0.0, 0.0]])) == 0.0
