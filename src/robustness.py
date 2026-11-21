"""Fixed, out-of-training-distribution checks for Vietnamese SMS variants."""

import pandas as pd


# These cases are evaluation fixtures only.  They are never passed to train.py
# as training data and are intentionally free of personal identifiers.
ROBUSTNESS_CASES = [
    {
        "case_id": "accent_01",
        "original": "Tai khoan cua ban bi khoa. Vui long xac minh ngay.",
        "transformed": "Tài khoản của bạn bị khóa. Vui lòng xác minh ngay.",
        "reference_label": 1,
        "transformation_type": "Vietnamese diacritics",
    },
    {
        "case_id": "no_accent_01",
        "original": "Tài khoản của bạn bị khóa. Vui lòng xác minh ngay.",
        "transformed": "Tai khoan cua ban bi khoa. Vui long xac minh ngay.",
        "reference_label": 1,
        "transformation_type": "Vietnamese without diacritics",
    },
    {
        "case_id": "mixed_alnum_01",
        "original": "Mã OTP để xác minh tài khoản.",
        "transformed": "Ma OTP 123456 de xac minh tai khoan.",
        "reference_label": 1,
        "transformation_type": "letters and numbers",
    },
    {
        "case_id": "teencode_01",
        "original": "Tài khoản của bạn bị khóa.",
        "transformed": "T@i kh0an cua b4n b! kh0a.",
        "reference_label": 1,
        "transformation_type": "teencode / leetspeak",
    },
    {
        "case_id": "uppercase_01",
        "original": "Vui lòng cung cấp OTP để xác minh.",
        "transformed": "VUI LÒNG CUNG CẤP OTP ĐỂ XÁC MINH.",
        "reference_label": 1,
        "transformation_type": "all uppercase",
    },
    {
        "case_id": "url_01",
        "original": "Bạn đã trúng thưởng 50 triệu. Xem https://example.test/nhanqua.",
        "transformed": "Ban da trung thuong 50 trieu. Xem https://example.test/nhanqua.",
        "reference_label": 1,
        "transformation_type": "URL",
    },
    {
        "case_id": "special_01",
        "original": "Chúc mừng bạn đã nhận quà.",
        "transformed": "!!! Chúc mừng bạn đã nhận quà !!! $$$",
        "reference_label": 1,
        "transformation_type": "special characters",
    },
    {
        "case_id": "spacing_01",
        "original": "Không chia sẻ OTP cho người khác.",
        "transformed": "  Không   chia sẻ\tOTP   cho người khác.  ",
        "reference_label": 1,
        "transformation_type": "irregular whitespace",
    },
    {
        "case_id": "legitimate_01",
        "original": "Đơn hàng của bạn đã được giao thành công.",
        "transformed": "DON HANG CUA BAN DA DUOC GIAO THANH CONG.",
        "reference_label": 0,
        "transformation_type": "uppercase and no diacritics",
    },
]


def robustness_frame():
    """Return a fresh DataFrame so evaluation cannot mutate the fixtures."""
    return pd.DataFrame(ROBUSTNESS_CASES).copy()
