import pandas as pd

from src.classical_ml import LABELS, candidate_specs, load_locked_dataset, model_id


def test_candidate_grid_contains_required_families_and_features():
    specs = candidate_specs()
    families = {spec["family"] for spec in specs}
    features = {spec["features"] for spec in specs}
    assert {"MultinomialNB", "ComplementNB", "LogisticRegression", "LinearSVM"}.issubset(families)
    assert {"word_1", "word_12", "word_123", "char_25", "char_35", "hybrid_word12_char35"}.issubset(features)
    assert len({model_id(spec) for spec in specs}) == len(specs)


def test_locked_dataset_rejects_empty_phase1_splits(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    columns = ["sample_id", "message", "message_anonymized", "label", "label_id", "source_name", "source_type", "source_url", "license", "original_label", "raw_label", "group_id", "pii_status", "split"]
    for split in ("train", "validation", "test"):
        pd.DataFrame(columns=columns).to_csv(processed / f"{split}.csv", index=False)
    status = load_locked_dataset(processed, tmp_path / "lock.json")
    assert status["ready"] is False
    assert "dataset_lock_missing" in status["reasons"]
    assert "empty_train" in status["reasons"]
