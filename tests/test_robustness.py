from src.robustness import robustness_frame


def test_robustness_fixture_covers_required_variants_and_is_not_training_data():
    cases = robustness_frame()
    assert len(cases) >= 8
    required = {
        "Vietnamese diacritics",
        "Vietnamese without diacritics",
        "letters and numbers",
        "teencode / leetspeak",
        "all uppercase",
        "URL",
        "special characters",
        "irregular whitespace",
    }
    assert required.issubset(set(cases["transformation_type"]))
    assert {"original", "transformed", "reference_label"}.issubset(cases.columns)
