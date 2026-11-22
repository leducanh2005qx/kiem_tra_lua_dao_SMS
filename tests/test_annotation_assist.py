from src.annotation_assist import suggest_label


def test_context_suggestions_are_provisional_and_not_keyword_only():
    assert suggest_label("Không chia sẻ mã OTP cho bất kỳ ai.").label == "LEGITIMATE"
    assert suggest_label("Hãy gửi mã OTP để nhân viên xác minh tài khoản qua đường dẫn https://fake.example").label == "PHISHING"
    assert suggest_label("Bạn trúng thưởng, hãy chuyển phí trước để nhận thưởng.").label == "FINANCIAL_SCAM"
    suggestion = suggest_label("OTP ngân hàng")
    assert suggestion.label == "NEEDS_REVIEW"
    assert suggestion.confidence < 0.5
    assert suggestion.method == "context_triage_v2"
