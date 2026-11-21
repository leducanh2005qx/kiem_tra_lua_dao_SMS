from src.dataset_engineering import (
    FOUR_CLASS_LABELS,
    anonymize_message,
    normalize_text,
    normalized_hash,
)


def test_four_class_mapping_is_explicit():
    assert FOUR_CLASS_LABELS == {
        "LEGITIMATE": 0,
        "PROMOTION": 1,
        "PHISHING": 2,
        "FINANCIAL_SCAM": 3,
    }


def test_normalization_preserves_diacritics_and_hashes_equivalent_spacing():
    assert normalize_text("  Ngân   hàng\n hợp lệ ") == "Ngân hàng hợp lệ"
    assert normalized_hash("Ngân hàng hợp lệ") == normalized_hash(" ngân hàng   hợp lệ ")


def test_anonymization_keeps_url_domain_and_masks_contact_values():
    output = anonymize_message("Truy cập https://example.com/login, gọi 0912345678 hoặc email a@b.vn")
    assert "[url_domain:example.com]" in output
    assert "[phone]" in output
    assert "[email]" in output
