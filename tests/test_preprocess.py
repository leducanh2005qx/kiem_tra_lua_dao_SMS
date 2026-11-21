import pytest

from src.preprocess import normalize_message, validate_message
from src.modeling import make_pipeline, explain_message, get_evidence


def test_unicode_and_whitespace():
    assert normalize_message("  TÀI   KHOẢN  \n") == "tài khoản"


def test_keep_negation_and_url():
    value = normalize_message("Không chia sẻ OTP tại https://example.org!")
    assert "không" in value
    assert "https://example.org" in value


def test_teencode_is_normalized_without_rewriting_bare_numbers():
    value = normalize_message("T@i kh0an cua b4n b! kh0a OTP 066595")
    assert "tai khoan cua ban bi khoa" in value
    assert "066595" in value


def test_input_validation_rejects_empty_and_long_messages():
    with pytest.raises(ValueError):
        validate_message("   ")
    with pytest.raises(ValueError):
        validate_message("x" * 11, max_chars=10)


def test_pipeline_predicts_and_explains():
    x = [
        "chuyển tiền trước nhận giải thưởng", "nhập mật khẩu ngay để mở khóa",
        "truy cập đường link lạ để nhận tiền", "yêu cầu cung cấp mã otp bí mật",
        "đơn hàng đã giao thành công", "hẹn gặp tại thư viện",
        "cảm ơn bạn đã tham gia", "bạn đã đăng ký thành công",
    ]
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    for method in ["nb_word_1", "svm_word_1", "svm_word_12", "svm_char_35"]:
        model = make_pipeline(method).fit(x, y)
        prediction = model.predict(["đã đăng ký thành công"])
        assert int(prediction[0]) in [0, 1]
        assert isinstance(get_evidence(model, "đã đăng ký thành công"), list)
        details = explain_message(model, "đã đăng ký thành công")
        assert all("toward_label" in row for row in details)
