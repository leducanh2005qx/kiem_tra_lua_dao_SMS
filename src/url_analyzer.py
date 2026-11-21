"""Offline URL extraction and explainable heuristic risk analysis.

This module never resolves DNS, opens a URL, follows redirects or sends text to
an external service.  Its output is a separate heuristic signal from the SMS
machine-learning classifier.
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qs, unquote, urlsplit


URL_RE = re.compile(
    r"(?i)(?:https?://|www\.)[^\s<>]+|(?<![@\w])(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>]*)?"
)
TRAILING_PUNCTUATION = ".,;:!?)]}>\"'"
SHORTENER_HOSTS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "lnkd.in"}
BRANDS = {"vietcombank", "vcb", "viettel", "momo", "zalopay", "zalo", "bidv", "techcombank", "mbbank", "agribank"}
SUSPICIOUS_PATH_WORDS = {
    "login", "signin", "verify", "verification", "xacminh", "xác minh", "otp",
    "account", "taikhoan", "tài khoản", "unlock", "restore", "reset", "gift",
    "reward", "nhanqua", "trúng thưởng", "payment", "thanhtoan", "chuyen-tien",
}
SUSPICIOUS_QUERY_WORDS = {"password", "passwd", "otp", "token", "verify", "account", "card", "cccd"}


def _clean_url(value: str) -> str:
    return value.rstrip(TRAILING_PUNCTUATION)


def extract_urls(text: str) -> list[str]:
    """Extract display URLs without attempting any network access."""
    return list(dict.fromkeys(_clean_url(match.group(0)) for match in URL_RE.finditer(str(text or ""))))


def _host_parts(hostname: str):
    hostname = (hostname or "").lower().rstrip(".")
    parts = hostname.split(".") if hostname else []
    domain = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
    return hostname, domain, parts


def analyze_url(url: str) -> dict:
    """Return transparent heuristic signals for one URL."""
    original = _clean_url(url)
    parse_target = original if re.match(r"(?i)^https?://", original) else f"http://{original}"
    parsed = urlsplit(parse_target)
    hostname, domain, parts = _host_parts(parsed.hostname)
    reasons = []
    score = 0

    def flag(points, reason):
        nonlocal score
        score += points
        reasons.append(reason)

    if len(original) > 120:
        flag(20, "URL dài bất thường")
    if len(parts) >= 4:
        flag(15, "Có nhiều tầng subdomain")
    if "@" in parsed.netloc:
        flag(30, "Có ký tự @ trong phần định danh host")
    try:
        ipaddress.ip_address(hostname)
        flag(25, "Dùng địa chỉ IP literal thay cho tên miền")
    except ValueError:
        pass
    if "xn--" in hostname:
        flag(20, "Tên miền dùng Punycode")
    if hostname in SHORTENER_HOSTS or domain in SHORTENER_HOSTS:
        flag(20, "Dùng dịch vụ rút gọn URL")
    decoded_path = unquote(parsed.path).lower()
    decoded_query = unquote(parsed.query).lower()
    if any(word in decoded_path for word in SUSPICIOUS_PATH_WORDS):
        flag(20, "Path có từ ngữ thường gặp trong luồng đăng nhập/xác minh")
    if any(word in decoded_query for word in SUSPICIOUS_QUERY_WORDS) or any(key.lower() in SUSPICIOUS_QUERY_WORDS for key in parse_qs(parsed.query)):
        flag(20, "Query chứa tham số nhạy cảm hoặc xác minh")
    if "%" in parsed.path or "%" in parsed.query:
        flag(10, "URL chứa chuỗi mã hóa percent")
    if "//" in parsed.path:
        flag(10, "Path có cấu trúc slash bất thường")
    brand_hits = [brand for brand in BRANDS if brand in hostname or brand in decoded_path]
    if brand_hits and not any(hostname == official or hostname.endswith(f".{official}") for official in {
        "vietcombank.com.vn", "viettel.vn", "momo.vn", "zalopay.vn", "zalo.me", "bidv.com.vn",
        "techcombank.com", "mbbank.com.vn", "agribank.com.vn",
    }):
        flag(25, f"Có dấu hiệu mạo danh thương hiệu: {', '.join(sorted(brand_hits))}")

    score = min(100, score)
    level = "cao" if score >= 60 else "trung bình" if score >= 30 else "thấp"
    return {
        "url": original,
        "hostname": hostname,
        "domain": domain,
        "path": parsed.path,
        "query": parsed.query,
        "is_ip_literal": _is_ip(hostname),
        "risk_score": score,
        "risk_level": level,
        "suspicious": bool(reasons),
        "reasons": reasons,
        "recommendation": "Không truy cập khi chưa xác minh độc lập." if reasons else "Không phát hiện tín hiệu heuristic nổi bật; vẫn không thể kết luận URL an toàn.",
    }


def _is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def analyze_sms_urls(text: str) -> dict:
    """Analyze every URL in SMS text and aggregate only heuristic results."""
    urls = extract_urls(text)
    analyses = [analyze_url(url) for url in urls]
    max_score = max((item["risk_score"] for item in analyses), default=0)
    return {
        "contains_urls": bool(urls),
        "urls": urls,
        "analyses": analyses,
        "risk_score": max_score,
        "risk_level": "cao" if max_score >= 60 else "trung bình" if max_score >= 30 else "thấp",
        "recommendation": "Không truy cập link khi chưa xác minh độc lập." if analyses else "SMS không chứa URL được nhận diện.",
    }
