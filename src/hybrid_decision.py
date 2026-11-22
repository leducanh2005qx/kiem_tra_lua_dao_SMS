"""Deterministic decision layer combining the legacy binary ML result and rules."""
from __future__ import annotations


FINAL_LABELS = {
    "LEGITIMATE": "Tin nhắn hợp lệ",
    "PROMOTION": "Quảng cáo / Spam",
    "PHISHING": "Giả mạo / Đánh cắp thông tin",
    "FINANCIAL_SCAM": "Lừa đảo tài chính",
    "NEEDS_REVIEW": "Chưa đủ căn cứ để kết luận",
}


def decide_hybrid(ml_label, ml_score, ml_score_type, risk_report: dict) -> dict:
    """Choose one final label with explicit rule precedence and audit fields."""
    ml_label = int(ml_label) if str(ml_label).strip() in {"0", "1"} else ml_label
    signals = list(risk_report.get("signals", []))
    high_phishing = [item for item in signals if item.get("label") == "PHISHING" and item.get("severity") == "high"]
    high_financial = [item for item in signals if item.get("label") == "FINANCIAL_SCAM" and item.get("severity") == "high"]
    promotion = [item for item in signals if item.get("label") == "PROMOTION"]
    suspicious_urls = [item for item in signals if item.get("code") in {"SUSPICIOUS_URL", "SHORTENED_URL"}]

    if high_phishing and high_financial:
        phishing_evidence = sum(len(item.get("evidence", [])) for item in high_phishing)
        financial_evidence = sum(len(item.get("evidence", [])) for item in high_financial)
        if phishing_evidence == financial_evidence:
            final_label = "NEEDS_REVIEW"
            reason = "Có đồng thời dấu hiệu lấy thông tin và yêu cầu tài chính nhưng chưa xác định được mục tiêu chính."
        else:
            final_label = "PHISHING" if phishing_evidence > financial_evidence else "FINANCIAL_SCAM"
            reason = "Ưu tiên nhóm có nhiều bằng chứng trực tiếp hơn trong SMS."
    elif high_phishing:
        final_label = "PHISHING"
        reason = "Rule phát hiện yêu cầu lấy OTP/mật khẩu/thông tin tài khoản trong ngữ cảnh đáng ngờ."
    elif high_financial:
        final_label = "FINANCIAL_SCAM"
        reason = "Rule phát hiện yêu cầu chuyển tiền, đóng phí, nạp tiền hoặc đặt cọc gắn với lời dụ dỗ/áp lực."
    elif promotion and not suspicious_urls:
        final_label = "PROMOTION"
        reason = "Rule nhận diện nội dung quảng cáo/khuyến mãi mà chưa thấy hành vi nhạy cảm."
    elif ml_label == 0 and not signals:
        final_label = "LEGITIMATE"
        reason = "Model dự đoán hợp lệ và chưa phát hiện tín hiệu rủi ro rõ ràng."
    elif ml_label == 0 and not any(item.get("severity") == "high" for item in signals):
        final_label = "NEEDS_REVIEW"
        reason = "Model dự đoán hợp lệ nhưng vẫn có tín hiệu cần kiểm tra thêm."
    else:
        final_label = "NEEDS_REVIEW"
        reason = "Model dự đoán SMS nghi vấn nhưng rule chưa đủ căn cứ để chọn loại lừa đảo cụ thể."

    reasons = list(dict.fromkeys([reason] + list(risk_report.get("reasons", []))))
    result = dict(risk_report)
    result.update({
        "ml_label": ml_label,
        "ml_score": ml_score,
        "ml_score_type": ml_score_type or "unavailable",
        "final_label": final_label,
        "risk_level": "cao" if final_label in {"PHISHING", "FINANCIAL_SCAM"} else risk_report.get("risk_level", "thấp"),
        "rule_override": bool(ml_label == 0 and final_label in {"PHISHING", "FINANCIAL_SCAM", "PROMOTION"}),
        "needs_review": final_label == "NEEDS_REVIEW",
        "reasons": reasons,
        "decision_reason": reason,
        "ml_label_name": "LEGITIMATE" if ml_label == 0 else "SUSPICIOUS_BINARY" if ml_label == 1 else "UNKNOWN",
    })
    return result
