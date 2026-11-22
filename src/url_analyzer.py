"""Offline URL extraction and explainable risk analysis.

The analyzer never resolves DNS, follows redirects, opens a URL or sends SMS
content to an external service. Defanging is applied only to a private copy
used for parsing; the original URL is always retained in the result.
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qs, unquote, urlsplit

from src.risk_config import BRAND_DOMAINS, OFFICIAL_DOMAINS, SHORTENER_HOSTS


_DOMAIN_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_DOMAIN_SEPARATOR = r"(?:\.|\[\.\]|\(\.\))"
URL_RE = re.compile(
    r"(?i)(?:(?:https?|hxxps?)://|www\.)[^\s<>]+|(?<![@\w.])"
    rf"{_DOMAIN_LABEL}(?:{_DOMAIN_SEPARATOR}{_DOMAIN_LABEL})+"
    r"(?::\d+)?(?:[/?#][^\s<>]*)?"
)
TRAILING_PUNCTUATION = ".,;:!?}>\"'"
SUSPICIOUS_PATH_WORDS = {
    "login", "signin", "verify", "verification", "xacminh", "xác minh", "otp",
    "account", "taikhoan", "tài khoản", "unlock", "restore", "reset", "gift",
    "reward", "nhanqua", "trúng thưởng", "payment", "thanhtoan", "chuyen-tien",
}
SUSPICIOUS_QUERY_WORDS = {"password", "passwd", "otp", "token", "verify", "account", "card", "cccd"}


def _clean_url(value: str) -> str:
    """Remove sentence punctuation without stripping balanced URL characters."""
    value = str(value or "").strip()
    while value and value[-1] in TRAILING_PUNCTUATION:
        value = value[:-1]
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    while value.endswith("]") and value.count("]") > value.count("["):
        value = value[:-1]
    return value


def _defang(value: str) -> str:
    value = re.sub(
        r"(?i)^hxxps?://",
        lambda match: "https://" if match.group(0).lower().startswith("hxxps") else "http://",
        value,
    )
    return value.replace("[.]", ".").replace("(.)", ".")


def extract_urls(text: str) -> list[str]:
    """Extract display URLs, including common defanged forms, offline."""
    values = [_clean_url(match.group(0)) for match in URL_RE.finditer(str(text or ""))]
    return list(dict.fromkeys(value for value in values if value))


def _idna_hostname(hostname: str) -> str:
    hostname = (hostname or "").rstrip(".").lower()
    try:
        return hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return hostname


def _host_parts(hostname: str):
    hostname = _idna_hostname(hostname)
    parts = hostname.split(".") if hostname else []
    domain = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
    return hostname, domain, parts


def _official_domain(hostname: str) -> str | None:
    hostname = _idna_hostname(hostname)
    for official in OFFICIAL_DOMAINS:
        if hostname == official or hostname.endswith(f".{official}"):
            return official
    if hostname == "gov.vn" or hostname.endswith(".gov.vn"):
        return "gov.vn"
    return None


def _brand_hits(hostname: str) -> list[str]:
    labels = re.split(r"[.-]", _idna_hostname(hostname))
    return sorted({brand for brand in BRAND_DOMAINS if any(brand in label for label in labels)})


def _is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except (ValueError, TypeError):
        return False


def analyze_url(url: str) -> dict:
    """Return transparent signals for one URL; malformed URLs never crash."""
    original = _clean_url(url)
    normalized = _defang(original)
    parse_target = normalized if re.match(r"(?i)^https?://", normalized) else f"http://{normalized}"
    reasons: list[str] = []
    evidence: list[str] = []
    score = 0

    def flag(points, reason, item=None):
        nonlocal score
        score += points
        reasons.append(reason)
        if item and item not in evidence:
            evidence.append(item)

    try:
        parsed = urlsplit(parse_target)
        hostname_raw = parsed.hostname
        hostname, domain, parts = _host_parts(hostname_raw)
        if not hostname:
            raise ValueError("hostname rỗng")
        decoded_path = unquote(parsed.path).lower()
        decoded_query = unquote(parsed.query).lower()
        official = _official_domain(hostname)
        brands = _brand_hits(hostname)
        shortener = hostname in SHORTENER_HOSTS or domain in SHORTENER_HOSTS
        if len(original) > 120:
            flag(20, "URL dài bất thường", original)
        if len(parts) >= 5:
            flag(15, "Có nhiều tầng subdomain", hostname)
        if "@" in parsed.netloc:
            flag(30, "Có ký tự @ trong phần định danh host", parsed.netloc)
        if _is_ip(hostname):
            flag(25, "Dùng địa chỉ IP literal thay cho tên miền", hostname)
        if "xn--" in hostname:
            flag(20, "Tên miền dùng Punycode", hostname)
        if shortener:
            flag(20, "Dùng dịch vụ rút gọn URL", hostname)
        if any(word in decoded_path for word in SUSPICIOUS_PATH_WORDS):
            flag(20, "Path có từ ngữ thường gặp trong luồng đăng nhập/xác minh", parsed.path)
        if any(word in decoded_query for word in SUSPICIOUS_QUERY_WORDS) or any(key.lower() in SUSPICIOUS_QUERY_WORDS for key in parse_qs(parsed.query)):
            flag(20, "Query chứa tham số nhạy cảm hoặc xác minh", parsed.query)
        if "%" in parsed.path or "%" in parsed.query:
            flag(10, "URL chứa chuỗi mã hóa percent", original)
        if "//" in parsed.path:
            flag(10, "Path có cấu trúc slash bất thường", parsed.path)
        if brands and not official:
            flag(35, f"Có dấu hiệu mạo danh thương hiệu: {', '.join(brands)}", hostname)
        status = "OFFICIAL_URL" if official else "SHORTENED_URL" if shortener else "SUSPICIOUS_URL" if reasons else "UNKNOWN_URL"
        if official:
            reasons.append("Hostname khớp domain tham chiếu; điều này không đảm bảo toàn bộ SMS an toàn.")
        level = "cao" if score >= 60 else "trung bình" if score >= 30 else "thấp"
        return {
            "url": original,
            "original_url": original,
            "normalized_url": normalized,
            "hostname": hostname,
            "domain": domain,
            "path": parsed.path,
            "query": parsed.query,
            "is_ip_literal": _is_ip(hostname),
            "status": status,
            "brand_hits": brands,
            "official_domain": official,
            "risk_score": min(100, score),
            "risk_level": level,
            "suspicious": bool(score or status == "MALFORMED_URL"),
            "reasons": reasons,
            "evidence": evidence,
            "malformed": False,
            "recommendation": "Không truy cập khi chưa xác minh độc lập." if reasons and not official else "Domain khớp danh sách tham chiếu; vẫn cần kiểm tra nội dung SMS và người gửi." if official else "Không phát hiện tín hiệu heuristic nổi bật; vẫn không thể kết luận URL an toàn.",
        }
    except (ValueError, IndexError, UnicodeError) as exc:
        return {
            "url": original,
            "original_url": original,
            "normalized_url": normalized,
            "hostname": "",
            "domain": "",
            "path": "",
            "query": "",
            "is_ip_literal": False,
            "status": "MALFORMED_URL",
            "brand_hits": [],
            "official_domain": None,
            "risk_score": 80,
            "risk_level": "cao",
            "suspicious": True,
            "reasons": [f"URL không thể phân tích an toàn: {type(exc).__name__}"],
            "evidence": [original] if original else [],
            "malformed": True,
            "recommendation": "Không truy cập URL này; cần kiểm tra bằng nguồn độc lập.",
        }


def analyze_sms_urls(text: str) -> dict:
    """Analyze every URL independently and aggregate the strongest signal."""
    urls = extract_urls(text)
    analyses = [analyze_url(url) for url in urls]
    max_score = max((item["risk_score"] for item in analyses), default=0)
    if any(item["status"] == "MALFORMED_URL" for item in analyses):
        level = "cao"
    else:
        level = "cao" if max_score >= 60 else "trung bình" if max_score >= 30 else "thấp"
    return {
        "contains_urls": bool(urls),
        "urls": urls,
        "analyses": analyses,
        "risk_score": max_score,
        "risk_level": level,
        "recommendation": "Không truy cập link khi chưa xác minh độc lập." if analyses else "SMS không chứa URL được nhận diện.",
    }
