"""Conservative machine-assisted label suggestions for the annotation queue.

These are triage suggestions, not verified labels. They deliberately use
context relations (requested action + object + negation), cap confidence, and
must never be consumed by the training pipeline until a human confirms them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.preprocess import normalize_message


SUGGESTION_METHOD = "context_triage_v2"


@dataclass(frozen=True)
class Suggestion:
    label: str
    confidence: float
    reasons: tuple[str, ...]
    method: str = SUGGESTION_METHOD


@dataclass(frozen=True)
class HierarchicalSuggestion:
    layer1_label: str
    layer2_label: str
    non_scam_type: str
    intents: tuple[str, ...]
    confidence: float
    reasons: tuple[str, ...]
    method: str = SUGGESTION_METHOD


def _has(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def suggest_label(message: str) -> Suggestion:
    """Suggest one label or NEEDS_REVIEW without asserting ground truth."""
    text = normalize_message(message)
    if not text:
        return Suggestion("NEEDS_REVIEW", 0.0, ("SMS rỗng",))

    negation = _has(text, (r"\bkhông\b", r"\bđừng\b", r"\bkhông bao giờ\b", r"tuyệt đối không"))
    otp_or_credential = _has(text, (r"\botp\b", r"mã xác thực", r"mật khẩu", r"mã pin", r"số thẻ", r"đăng nhập"))
    credential_request = _has(text, (r"cung cấp", r"cung cap", r"gửi", r"gui", r"đọc", r"doc", r"nhập", r"nhap", r"điền", r"dien", r"xác minh ngay", r"xac minh ngay", r"bấm vào", r"bam vao", r"truy cập", r"truy cap", r"link", r"đường dẫn", r"duong dan"))
    money_action = _has(text, (r"chuyển khoản", r"chuyen khoan", r"chuyển tiền", r"chuyen tien", r"chuyển phí", r"chuyen phi", r"nộp phí", r"nop phi", r"đóng phí", r"dong phi", r"đặt cọc", r"dat coc", r"thanh toán trước", r"thanh toan truoc", r"phí trước", r"phi truoc", r"phí nhận", r"phi nhan", r"phí hồ sơ", r"phi ho so"))
    financial_offer = _has(text, (r"trúng thưởng", r"trung thuong", r"nhận thưởng", r"nhan thuong", r"đầu tư", r"dau tu", r"lợi nhuận", r"loi nhuan", r"khoản vay", r"khoan vay", r"giải ngân", r"giai ngan", r"hoàn tiền", r"hoan tien", r"tuyển dụng", r"tuyen dung"))
    impersonation_or_pressure = _has(text, (r"ngân hàng", r"ngan hang", r"cơ quan", r"co quan", r"công an", r"cong an", r"viettel", r"vinaphone", r"tài khoản.*khóa", r"tai khoan.*khoa", r"xác minh.*ngay", r"xac minh.*ngay", r"hạn cuối", r"han cuoi"))
    url = _has(text, (r"https?://", r"www\.", r"hxxp", r"\.vn/", r"\.com/"))
    promotion = _has(text, (r"khuyến mãi", r"khuyen mai", r"ưu đãi", r"uu dai", r"giảm giá", r"giam gia", r"sale", r"mã giảm", r"ma giam", r"tích điểm", r"tich diem", r"mua hàng", r"mua hang"))
    legitimate_notice = _has(text, (r"giao dịch.*thành công", r"giao dich.*thanh cong", r"số dư", r"so du", r"đơn hàng.*đã giao", r"don hang.*da giao", r"mã otp.*là", r"ma otp.*la", r"otp xác thực", r"otp xac thuc", r"thông báo", r"thong bao", r"xác nhận", r"xac nhan", r"cảm ơn quý khách", r"cam on quy khach"))

    if money_action and (financial_offer or impersonation_or_pressure or url or credential_request):
        reasons = ["Có hành động liên quan đến chuyển tiền/phí/đặt cọc", "Có ngữ cảnh lợi ích tài chính hoặc thúc ép"]
        if financial_offer:
            reasons.append("Có lời mời thưởng/đầu tư/vay/tuyển dụng gắn với hành động tài chính")
        return Suggestion("FINANCIAL_SCAM", 0.86 if not negation else 0.58, tuple(reasons))

    if otp_or_credential and credential_request and (impersonation_or_pressure or url) and not negation:
        reasons = ["Có yêu cầu cung cấp/nhập thông tin xác thực", "Có dấu hiệu giả mạo, áp lực hoặc liên kết"]
        return Suggestion("PHISHING", 0.86 if url or impersonation_or_pressure else 0.75, tuple(reasons))

    if promotion and not (money_action or credential_request):
        return Suggestion("PROMOTION", 0.80, ("Mục tiêu chính có vẻ là quảng cáo/khuyến mãi", "Chưa thấy yêu cầu credential hoặc chuyển tiền"))

    if legitimate_notice or (negation and otp_or_credential and not credential_request):
        return Suggestion("LEGITIMATE", 0.78, ("Có dạng thông báo/cảnh báo hợp lệ hoặc phủ định hành vi cung cấp thông tin",))

    return Suggestion("NEEDS_REVIEW", 0.35, ("Chưa đủ quan hệ ngữ cảnh để phân biệt chắc chắn 4 lớp",))


def suggest_hierarchical_label(message: str) -> HierarchicalSuggestion:
    """Convert the conservative triage result into layer-1/layer-2 fields."""
    suggestion = suggest_label(message)
    text = normalize_message(message)
    intents = []
    if _has(text, (r"\botp\b", r"mã xác thực", r"ma xac thuc")):
        intents.append("ASK_OTP")
    if _has(text, (r"mật khẩu", r"mat khau", r"mã pin", r"ma pin", r"đăng nhập", r"dang nhap")):
        intents.append("ASK_PASSWORD")
    if _has(text, (r"chuyển khoản", r"chuyen khoan", r"chuyển tiền", r"chuyen tien", r"thanh toán", r"thanh toan")):
        intents.append("ASK_TRANSFER")
    if _has(text, (r"ngân hàng", r"ngan hang", r"tài khoản", r"tai khoan")):
        intents.append("FAKE_BANK")
    if _has(text, (r"đóng phí", r"dong phi", r"nộp phí", r"nop phi", r"lệ phí", r"le phi")):
        intents.append("FAKE_FEE")
    if _has(text, (r"https?://", r"www\.", r"hxxp", r"đường dẫn", r"duong dan", r"link")):
        intents.append("MALICIOUS_LINK")
    if _has(text, (r"trúng thưởng", r"trung thuong", r"nhận thưởng", r"nhan thuong")):
        intents.append("FAKE_REWARD")
    if _has(text, (r"tuyển dụng", r"tuyen dung", r"việc làm", r"viec lam")):
        intents.append("FAKE_RECRUITMENT")
    if _has(text, (r"đầu tư", r"dau tu", r"lợi nhuận", r"loi nhuan")):
        intents.append("INVESTMENT_OFFER")
    if suggestion.label == "LEGITIMATE":
        return HierarchicalSuggestion("NOT_SCAM", "", "LEGITIMATE", tuple(dict.fromkeys(intents)), suggestion.confidence, suggestion.reasons)
    if suggestion.label == "PROMOTION":
        return HierarchicalSuggestion("NOT_SCAM", "", "PROMOTION", tuple(dict.fromkeys(intents)), suggestion.confidence, suggestion.reasons)
    if suggestion.label == "PHISHING":
        return HierarchicalSuggestion("SCAM", "PHISHING", "", tuple(dict.fromkeys(intents)), suggestion.confidence, suggestion.reasons)
    if suggestion.label == "FINANCIAL_SCAM":
        return HierarchicalSuggestion("SCAM", "FINANCIAL_SCAM", "", tuple(dict.fromkeys(intents)), suggestion.confidence, suggestion.reasons)
    return HierarchicalSuggestion("", "", "", tuple(dict.fromkeys(intents)), suggestion.confidence, suggestion.reasons)
