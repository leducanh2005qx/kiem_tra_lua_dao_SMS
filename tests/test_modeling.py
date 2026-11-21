import joblib

from src.modeling import NB_CANDIDATES, SVM_CANDIDATES, get_model_spec, make_pipeline
from src.preprocess import redact_message


def test_nb_grid_contains_requested_search_space():
    assert len(NB_CANDIDATES) == 30
    spec = get_model_spec("nb_word_12_a0p1_prior_false")
    assert spec["vectorizer"] == "word_12"
    assert spec["alpha"] == 0.1
    assert spec["fit_prior"] is False


def test_serialized_pipeline_owns_normalization():
    model = make_pipeline("nb_word_1")
    assert "normalize" in model.named_steps
    assert "tfidf" in model.named_steps
    assert "clf" in model.named_steps


def test_svm_grid_contains_requested_c_and_class_weight_values():
    assert len(SVM_CANDIDATES) == 24
    assert any(get_model_spec(name)["C"] == 0.1 for name in SVM_CANDIDATES)
    assert any(get_model_spec(name)["C"] == 2.0 for name in SVM_CANDIDATES)
    assert any(get_model_spec(name)["class_weight"] == "balanced" for name in SVM_CANDIDATES)


def test_pipeline_can_be_saved_and_loaded(tmp_path):
    model = make_pipeline("svm_word_1").fit(["tin nhắn bình thường", "cung cấp otp ngay"], [0, 1])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert restored.predict(["cung cap otp ngay"])[0] == model.predict(["cung cap otp ngay"])[0]


def test_report_redaction_masks_urls_and_identifiers():
    redacted = redact_message("Gọi +84 912 345 678 hoặc xem https://example.test/a/1234")
    assert "https://" not in redacted
    assert "+84" not in redacted
    assert "[url]" in redacted
    assert "[number]" in redacted
