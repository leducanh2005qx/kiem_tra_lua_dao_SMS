from src.phobert import dependency_status


def test_phobert_is_optional_and_reports_runtime_status():
    status = dependency_status()
    assert status["model_id"] == "vinai/phobert-base-v2"
    assert status["device"] in {"cpu", "cuda"}
    assert "license" in status
