"""Context-aware, offline rule analysis for Vietnamese SMS.

This module is deliberately independent from the serialized ML pipelines.  It
produces transparent signals and original-text evidence; the final label is
chosen by :mod:`src.hybrid_decision`.
"""
from __future__ import annotations

import re
import unicodedata

from src.preprocess import normalize_message
from src.risk_config import (
    CREDENTIAL_ACTION_PATTERNS,
    FINANCIAL_LURE_PATTERNS,
    IMPERSONATION_PATTERNS,
    MONEY_ACTION_PATTERNS,
    NEGATION_PATTERNS,
    OTP_PATTERNS,
    PROMOTION_PATTERNS,
)
from src.url_analyzer import analyze_sms_urls


def _fold_with_map(text: str):
    folded, indexes = [], []
    for index, character in enumerate(str(text or "").casefold()):
        value = unicodedata.normalize("NFD", character)
        for item in value:
            if not unicodedata.combining(item):
                folded.append("d" if item == "đ" else item)
                indexes.append(index)
    return "".join(folded), indexes


def _patterns_folded(patterns):
    output = []
    for pattern in patterns:
        folded, _ = _fold_with_map(pattern)
        output.append(folded)
    return tuple(output)


def _matches(text: str, patterns) -> list[re.Match]:
    return [match for pattern in _patterns_folded(patterns) for match in re.finditer(pattern, text, re.IGNORECASE)]


def _has(text: str, patterns) -> bool:
    return bool(_matches(text, patterns))


def _evidence(original: str, folded: str, indexes: list[int], patterns) -> list[str]:
    values = []
    for match in _matches(folded, patterns):
        if not indexes or match.start() >= len(indexes):
            continue
        start = indexes[match.start()]
        end_index = min(len(indexes) - 1, max(match.start(), match.end() - 1))
        end = indexes[end_index] + 1
        context_start = max(0, start - 28)
        context_end = min(len(original), end + 48)
        snippet = original[context_start:context_end].strip()
        if snippet and snippet not in values:
            values.append(snippet)
    return values[:8]


def _negated_target(folded: str, target_patterns) -> bool:
    negations = _patterns_folded(NEGATION_PATTERNS)
    for match in _matches(folded, target_patterns):
        window = folded[max(0, match.start() - 48):match.end()]
        if any(re.search(pattern, window, re.IGNORECASE) for pattern in negations):
            return True
    return False


def _signal(code, label, severity, explanation, evidence, action=""):
    return {
        "code": code,
        "label": label,
        "severity": severity,
        "explanation": explanation,
        "evidence": evidence,
        "requested_action": action,
    }


def analyze_risk(sms_text: str, ml_label, ml_confidence=None, ml_score_type="unavailable") -> dict:
    """Analyze one SMS without changing the ML model or making network calls."""
    original = str(sms_text or "")
    normalized = normalize_message(original)
    folded, indexes = _fold_with_map(original)
    urls = analyze_sms_urls(original)
    signals = []
    evidence = []

    otp = _has(folded, OTP_PATTERNS)
    credentials = _has(folded, (r"mật\s+khẩu", r"mat\s+khau", r"số\s+thẻ", r"so\s+the", r"cccd", r"thông\s+tin\s+cá\s+nhân", r"thong\s+tin\s+ca\s+nhan"))
    credential_action = _has(folded, CREDENTIAL_ACTION_PATTERNS)
    negated_credentials = _negated_target(folded, OTP_PATTERNS + (r"mật\s+khẩu", r"mat\s+khau", r"số\s+thẻ", r"so\s+the"))
    suspicious_urls = [item for item in urls["analyses"] if item["status"] in {"SUSPICIOUS_URL", "MALFORMED_URL"}]
    shortened_urls = [item for item in urls["analyses"] if item["status"] == "SHORTENED_URL"]
    url_evidence = [item.get("original_url", item.get("url", "")) for item in suspicious_urls + shortened_urls]
    pressure = _has(folded, (r"khóa\s+tài\s+khoản", r"khoa\s+tai\s+khoan", r"hạn\s+cuối", r"han\s+cuoi", r"ngay", r"phạt\s+nguội", r"phat\s+nguoi", r"cắt\s+điện", r"cat\s+dien"))
    impersonation = _has(folded, IMPERSONATION_PATTERNS)
    fake_link = _has(folded, (r"link\s+giả\s+mạo", r"link\s+gia\s+mao", r"đường\s+dẫn\s+giả\s+mạo", r"duong\s+dan\s+gia\s+mao", r"link\s+fake"))
    money_action = _has(folded, MONEY_ACTION_PATTERNS)
    financial_lure = _has(folded, FINANCIAL_LURE_PATTERNS)
    promotion = _has(folded, PROMOTION_PATTERNS)

    phishing_context = ((otp or credentials) or fake_link) and credential_action and not negated_credentials
    phishing_context = phishing_context and (impersonation or bool(urls["contains_urls"]) or pressure or otp or fake_link)
    if phishing_context:
        matches = OTP_PATTERNS if otp else (r"mật\s+khẩu", r"mat\s+khau", r"số\s+thẻ", r"so\s+the", r"link\s+giả\s+mạo", r"link\s+gia\s+mao")
        item_evidence = _evidence(original, folded, indexes, matches + CREDENTIAL_ACTION_PATTERNS)
        item_evidence.extend(url_evidence)
        signal = _signal("PHISHING_CREDENTIAL_REQUEST", "PHISHING", "high", "SMS yêu cầu cung cấp thông tin xác thực hoặc OTP trong ngữ cảnh đáng ngờ.", list(dict.fromkeys(item_evidence)), "Cung cấp OTP/mật khẩu/thông tin tài khoản")
        signals.append(signal)
        evidence.extend(signal["evidence"])

    financial_context = money_action and (financial_lure or pressure or impersonation or bool(suspicious_urls))
    if financial_context:
        item_evidence = _evidence(original, folded, indexes, MONEY_ACTION_PATTERNS + FINANCIAL_LURE_PATTERNS)
        item_evidence.extend(url_evidence)
        signal = _signal("FINANCIAL_PAYMENT_REQUEST", "FINANCIAL_SCAM", "high", "SMS gắn lợi ích/áp lực với yêu cầu chuyển tiền, đóng phí, đặt cọc hoặc nạp tiền.", list(dict.fromkeys(item_evidence)), "Chuyển tiền/đóng phí/nạp tiền/đặt cọc")
        signals.append(signal)
        evidence.extend(signal["evidence"])

    if promotion and not phishing_context and not financial_context and not credential_action and not money_action:
        item_evidence = _evidence(original, folded, indexes, PROMOTION_PATTERNS)
        signal = _signal("PROMOTION_CONTENT", "PROMOTION", "low", "SMS có các dấu hiệu quảng cáo hoặc khuyến mãi và chưa thấy yêu cầu nhạy cảm.", item_evidence, "Xem ưu đãi/sản phẩm")
        signals.append(signal)
        evidence.extend(item_evidence)

    for item in urls["analyses"]:
        if item["status"] in {"SUSPICIOUS_URL", "MALFORMED_URL"}:
            signal_label = "NEEDS_REVIEW" if item["status"] == "MALFORMED_URL" else "PHISHING"
            signal = _signal("SUSPICIOUS_URL", signal_label, "high" if item["status"] == "MALFORMED_URL" else "medium", "URL có dấu hiệu bất thường hoặc giả mạo thương hiệu.", [item.get("original_url", item.get("url", ""))] + item.get("reasons", []), "Truy cập đường dẫn")
            signals.append(signal)
            evidence.extend(signal["evidence"])
        elif item["status"] == "SHORTENED_URL":
            signal = _signal("SHORTENED_URL", "NEEDS_REVIEW", "medium", "SMS chứa URL rút gọn nên chưa xác định được đích đến ngoại tuyến.", [item.get("original_url", item.get("url", ""))], "Truy cập đường dẫn rút gọn")
            signals.append(signal)
            evidence.extend(signal["evidence"])

    risk_level = "cao" if any(item["severity"] == "high" for item in signals) else "trung bình" if signals or urls["risk_score"] >= 20 else "thấp"
    labels = {item["label"] for item in signals if item["label"] in {"PROMOTION", "PHISHING", "FINANCIAL_SCAM"}}
    return {
        "ml_label": int(ml_label) if str(ml_label).strip() in {"0", "1"} else ml_label,
        "ml_score": ml_confidence,
        "ml_score_type": ml_score_type or "unavailable",
        "final_label": "NEEDS_REVIEW",
        "risk_level": risk_level,
        "rule_override": False,
        "needs_review": True,
        "reasons": list(dict.fromkeys(item["explanation"] for item in signals)),
        "signals": signals,
        "urls": urls["analyses"],
        "evidence": list(dict.fromkeys(evidence)),
        "candidate_labels": sorted(labels),
        "normalized_text": normalized,
    }
