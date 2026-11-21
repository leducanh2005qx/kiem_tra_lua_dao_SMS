from src.url_analyzer import analyze_sms_urls, analyze_url, extract_urls


def test_extract_and_parse_urls_offline():
    urls = extract_urls("Xem https://bit.ly/abc?otp=1 hoặc http://192.168.1.1/login.")
    assert urls == ["https://bit.ly/abc?otp=1", "http://192.168.1.1/login"]
    result = analyze_url(urls[1])
    assert result["is_ip_literal"] is True
    assert "Dùng địa chỉ IP literal thay cho tên miền" in result["reasons"]


def test_url_analysis_never_claims_safety_and_does_not_fetch():
    report = analyze_sms_urls("Truy cập https://xn--example-d5a.test/login?password=x")
    assert report["contains_urls"] is True
    assert report["risk_score"] > 0
    assert "Không truy cập" in report["recommendation"]
