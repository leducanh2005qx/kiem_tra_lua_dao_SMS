# Quy tắc gán nhãn 4 lớp SMS Shield VN

## Nhãn cố định

| ID | Nhãn | Mục tiêu chính |
|---:|---|---|
| 0 | `LEGITIMATE` | Thông báo hợp lệ, OTP/giao dịch/dịch vụ thật, tin cá nhân hoặc cảnh báo an toàn. |
| 1 | `PROMOTION` | Quảng cáo, khuyến mại, giới thiệu sản phẩm/dịch vụ thông thường, không có dấu hiệu chiếm đoạt. |
| 2 | `PHISHING` | Giả mạo hoặc dụ lấy OTP, mật khẩu, thông tin thẻ, thông tin tài khoản/cá nhân, thường qua link hoặc trang đăng nhập giả. |
| 3 | `FINANCIAL_SCAM` | Dụ chuyển tiền, đóng phí, đặt cọc, đầu tư/vay giả, hoàn tiền giả, trúng thưởng giả hoặc chiếm đoạt tài sản. |

`NEEDS_REVIEW` là trạng thái kiểm duyệt, không phải lớp thứ 5 và không được đưa vào train.

## Quy tắc phân biệt theo ngữ cảnh

- “Không chia sẻ mã OTP cho bất kỳ ai” → `LEGITIMATE`; “gửi OTP cho nhân viên để xác minh” → `PHISHING`.
- “Mã OTP của bạn là …” hoặc thông báo giao dịch chỉ để thông báo → `LEGITIMATE`, không gán phishing chỉ vì có từ OTP.
- “Siêu thị giảm giá 50%” → `PROMOTION`; “trúng thưởng, chuyển phí trước để nhận thưởng” → `FINANCIAL_SCAM`.
- Thông báo ngân hàng/số dư/giao dịch hợp lệ → `LEGITIMATE`; giả ngân hàng yêu cầu đăng nhập, OTP hoặc thông tin → `PHISHING`.
- Tuyển dụng thông thường → `LEGITIMATE` hoặc `PROMOTION` theo mục tiêu; yêu cầu nộp phí/đặt cọc bất thường để nhận việc → `FINANCIAL_SCAM`.
- Nếu vừa lấy thông tin vừa yêu cầu tiền, chọn mục tiêu hành động trực tiếp: lấy credential/OTP → `PHISHING`; chuyển tiền/phí/đặt cọc → `FINANCIAL_SCAM`.

Không dùng một từ khóa, URL heuristic hoặc nhãn nhị phân cũ để quyết định. Phải xét chủ thể, động từ yêu cầu, phủ định, áp lực/thời hạn, đích đến URL và hậu quả. Không đủ ngữ cảnh thì để `NEEDS_REVIEW`.

## Trường annotation bắt buộc

`sample_id`, `sms_content`/`message`, `label`, `scam_type`, `intent`, `reason`/`label_reason`, `source`, `annotator`, `verified`, `review_status`, `confidence`, `group_id`, `notes`.

Mỗi mẫu cần hai người gán nhãn độc lập khi có thể. Người adjudicate ghi nhãn cuối, lý do và trạng thái `VERIFIED`/`GOLD_VERIFIED`. Mẫu tổng hợp phải đánh dấu `synthetic=true`, không được mô tả là SMS thực tế và không được đưa vào gold test nếu chưa kiểm duyệt.

## Bắt buộc kiểm tra hard-negative/hard-positive

Tạo các cặp cùng từ khóa nhưng khác ý định: phủ định OTP đối lập với yêu cầu OTP; giao dịch hợp lệ đối lập với link xác minh giả; quảng cáo giảm giá đối lập với phí nhận thưởng; tuyển dụng hợp lệ đối lập với nộp phí trước. Giữ nguyên cặp trong cùng nhóm để tránh leakage giữa train/test.
