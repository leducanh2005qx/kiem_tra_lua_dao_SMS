from src.hybrid_decision import decide_hybrid
from src.risk_analyzer import analyze_risk
from src.url_analyzer import analyze_url, analyze_sms_urls, extract_urls


def _hybrid(message, ml_label=0):
    risk = analyze_risk(message, ml_label, 0.2, "posterior_label_1")
    return decide_hybrid(ml_label, 0.2, "posterior_label_1", risk)


def test_url_forms_and_domain_boundaries_are_safe_offline():
    values = extract_urls("hxxps://example[.]com/a, example(.)com/b portal.gov.vn/x gov.vn.example.com/y")
    assert "hxxps://example[.]com/a" in values
    assert "example(.)com/b" in values
    assert "portal.gov.vn/x" in values
    assert "gov.vn.example.com/y" in values
    assert analyze_url("portal.gov.vn/x")["status"] == "OFFICIAL_URL"
    assert analyze_url("gov.vn.example.com/y")["status"] != "OFFICIAL_URL"
    assert analyze_url("fake-vietcombank.com/login")["status"] == "SUSPICIOUS_URL"
    assert analyze_url("https://bit.ly/x")["status"] == "SHORTENED_URL"


def test_malformed_ipv6_does_not_stop_other_urls():
    report = analyze_sms_urls("http://[2001:db8:::1]/x và https://example.com/a")
    assert len(report["analyses"]) == 2
    assert any(item["status"] == "MALFORMED_URL" for item in report["analyses"])
    assert report["risk_level"] == "cao"


def test_context_rules_handle_negation_and_requested_actions():
    assert _hybrid("Không cung cấp mã OTP cho bất kỳ ai.")["final_label"] == "LEGITIMATE"
    assert _hybrid("Cung cấp mã OTP để xác minh tài khoản.")["final_label"] == "PHISHING"
    assert _hybrid("Truy cập link giả mạo để xác minh tài khoản ngân hàng.")["final_label"] == "PHISHING"
    assert _hybrid("Siêu thị giảm giá 30%.")["final_label"] == "PROMOTION"
    assert _hybrid("Trúng thưởng và đóng phí trước để nhận tiền.")["final_label"] == "FINANCIAL_SCAM"
    assert _hybrid("Tuyển cộng tác viên.")["final_label"] == "LEGITIMATE"
    assert _hybrid("Tuyển cộng tác viên, nạp tiền trước để nhận hoa hồng.")["final_label"] == "FINANCIAL_SCAM"


def test_ml_legitimate_can_be_overridden_by_strong_rule():
    result = _hybrid("Cung cấp mã OTP để mở khóa tài khoản ngân hàng.", ml_label=0)
    assert result["final_label"] == "PHISHING"
    assert result["rule_override"] is True
    assert result["evidence"]


def test_suspicious_without_specific_type_is_review():
    result = _hybrid("Tin nhắn có thông báo lạ, vui lòng kiểm tra.", ml_label=1)
    assert result["final_label"] == "NEEDS_REVIEW"
    assert result["needs_review"] is True
