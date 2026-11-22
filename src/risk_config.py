"""Configuration for offline risk rules and domain references."""

OFFICIAL_DOMAINS = {
    "vietcombank.com.vn",
    "mbbank.com.vn",
    "bidv.com.vn",
    "techcombank.com",
    "acb.com.vn",
    "vietinbank.vn",
    "agribank.com.vn",
    "vpbank.vn",
    "tpbank.vn",
    "momo.vn",
    "zalopay.vn",
    "vnpay.vn",
    "viettel.vn",
    "vinaphone.com.vn",
    "mobifone.vn",
    "evn.com.vn",
    "evnhanoi.vn",
    "shopee.vn",
    "lazada.vn",
    "tiki.vn",
    "ghn.vn",
    "ghtk.vn",
    "viettelpost.com.vn",
}

SHORTENER_HOSTS = {
    "bit.ly", "tinyurl.com", "bom.so", "t.co", "goo.gl", "ow.ly",
    "is.gd", "cutt.ly", "lnkd.in",
}

BRAND_DOMAINS = {
    "vietcombank": {"vietcombank.com.vn"},
    "vcb": {"vietcombank.com.vn"},
    "mbbank": {"mbbank.com.vn"},
    "bidv": {"bidv.com.vn"},
    "techcombank": {"techcombank.com"},
    "acb": {"acb.com.vn"},
    "vietinbank": {"vietinbank.vn"},
    "agribank": {"agribank.com.vn"},
    "vpbank": {"vpbank.vn"},
    "tpbank": {"tpbank.vn"},
    "viettel": {"viettel.vn", "viettelpost.com.vn"},
    "vinaphone": {"vinaphone.com.vn"},
    "mobifone": {"mobifone.vn"},
    "momo": {"momo.vn"},
    "zalopay": {"zalopay.vn"},
    "vnpay": {"vnpay.vn"},
    "evn": {"evn.com.vn", "evnhanoi.vn"},
    "shopee": {"shopee.vn"},
    "lazada": {"lazada.vn"},
    "tiki": {"tiki.vn"},
    "ghn": {"ghn.vn"},
    "ghtk": {"ghtk.vn"},
    "vneid": {"gov.vn"},
}

PROMOTION_PATTERNS = (
    r"\[\s*qc\s*\]", r"soạn\s+tc\s+gửi", r"tu\s+choi\s+qc",
    r"giảm\s+giá", r"voucher", r"tặng\s+data", r"mua\s*1\s+tặng\s*1",
    r"ưu\s+đãi", r"khuyến\s+mãi", r"chương\s+trình\s+khách\s+hàng",
    r"sale", r"mã\s+giảm", r"tích\s+điểm",
)

NEGATION_PATTERNS = (
    r"\bkhông\b", r"\bđừng\b", r"\bkhong\b", r"\bdung\b",
    r"tuyệt\s+đối\s+không", r"tuyet\s+doi\s+khong",
)

OTP_PATTERNS = (r"\botp\b", r"mã\s+xác\s+thực", r"ma\s+xac\s+thuc", r"mã\s+pin")
CREDENTIAL_ACTION_PATTERNS = (
    r"cung\s+cấp", r"cung\s+cap", r"gửi", r"gui", r"nhập", r"nhap",
    r"điền", r"dien", r"đăng\s+nhập", r"dang\s+nhap", r"xác\s+minh",
    r"xac\s+minh", r"truy\s+cập", r"truy\s+cap", r"bấm\s+vào", r"bam\s+vao",
    r"click", r"cài\s+đặt", r"cai\s+dat",
)
MONEY_ACTION_PATTERNS = (
    r"chuyển\s+khoản", r"chuyen\s+khoan", r"chuyển\s+tiền", r"chuyen\s+tien",
    r"đóng\s+phí", r"dong\s+phi", r"nộp\s+phí", r"nop\s+phi", r"phí\s+hồ\s+sơ",
    r"phi\s+ho\s+so", r"đặt\s+cọc", r"dat\s+coc", r"nạp\s+tiền", r"nap\s+tien",
    r"thanh\s+toán\s+trước", r"thanh\s+toan\s+truoc", r"chuyển\s+phí",
)
FINANCIAL_LURE_PATTERNS = (
    r"trúng\s+thưởng", r"trung\s+thuong", r"nhận\s+thưởng", r"nhan\s+thuong",
    r"lợi\s+nhuận", r"loi\s+nhuan", r"đầu\s+tư", r"dau\s+tu", r"giải\s+ngân",
    r"giai\s+ngan", r"hoàn\s+tiền", r"hoan\s+tien", r"hoa\s+hồng", r"hoa\s+hong",
    r"tuyển\s+cộng\s+tác\s+viên", r"tuyen\s+cong\s+tac\s+vien", r"việc\s+nhẹ\s+lương\s+cao",
    r"viec\s+nhe\s+luong\s+cao", r"bao\s+nợ\s+xấu", r"bao\s+no\s+xau",
)
IMPERSONATION_PATTERNS = (
    r"ngân\s+hàng", r"ngan\s+hang", r"cơ\s+quan", r"co\s+quan", r"công\s+an",
    r"cong\s+an", r"vietcombank", r"mbbank", r"bidv", r"vneid", r"tài\s+khoản",
    r"tai\s+khoan", r"khóa\s+tài\s+khoản", r"khoa\s+tai\s+khoan", r"phạt\s+nguội",
    r"phat\s+nguoi", r"cắt\s+điện", r"cat\s+dien", r"sinh\s+trắc\s+học",
)
