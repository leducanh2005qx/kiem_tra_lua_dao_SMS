# Hướng dẫn gán nhãn SCAM cho SMS Shield VN

Tài liệu này dùng cho người gán nhãn thủ công. Mục tiêu là phân biệt ý định và ngữ cảnh của toàn bộ SMS, không phải đếm từ khóa.

## Nhãn

| Nhãn | Dùng khi |
|---|---|
| `SCAM` | Tin nhắn có căn cứ cho thấy giả mạo, chiếm đoạt thông tin/tài sản, dụ chuyển tiền, đóng phí trước, đầu tư/vay giả, tuyển dụng giả, trúng thưởng giả hoặc dẫn tới hành động nguy hiểm. |
| `NOT_SCAM` | OTP hợp lệ, thông báo giao dịch/dịch vụ, cảnh báo an toàn, tin cá nhân hoặc quảng cáo thông thường không có mục tiêu chiếm đoạt. |
| `NEEDS_REVIEW` | Thiếu ngữ cảnh, không đủ căn cứ, hoặc có dấu hiệu xung đột cần người khác xác minh. Không được đưa nhãn này vào train. |

`SCAM` là nhãn hành vi đáng ngờ dựa trên nội dung SMS; không phải kết luận pháp lý về người gửi hay tổ chức.

## Quy tắc theo ngữ cảnh

- “Không chia sẻ mã OTP cho bất kỳ ai” → `NOT_SCAM`.
- “Cung cấp mã OTP cho nhân viên để xác minh” → `SCAM`.
- Thông báo số dư/giao dịch chỉ để thông báo → `NOT_SCAM`.
- “Tài khoản bị khóa, truy cập link để xác minh ngay” → `SCAM` nếu toàn bộ ngữ cảnh cho thấy link/yêu cầu là giả mạo hoặc chiếm thông tin.
- Khuyến mãi giảm giá thông thường → `NOT_SCAM`.
- Trúng thưởng nhưng yêu cầu chuyển phí trước → `SCAM`.
- Chỉ có từ “OTP”, “ngân hàng”, “tài khoản”, “chuyển tiền”, “nhận thưởng” hoặc URL → chưa đủ căn cứ; cân nhắc `NEEDS_REVIEW`.

Luôn kiểm tra quan hệ giữa chủ thể, hành động được yêu cầu, áp lực/thời hạn, đích đến của URL và hậu quả được nêu. Phủ định (“không”, “đừng”, “không bao giờ”) phải được giữ nguyên vì có thể đảo ngược ý nghĩa.

## Trường cần ghi cho mỗi mẫu

`message`, `final_label`, `scam_type`, `intent`, `label_reason`, `source_name`, `source_type`, `annotator_id`, `review_status`, `confidence`, `group_id`, `notes`.

`scam_type` có thể là `phishing`, `payment`, `fee`, `investment`, `recruitment`, `prize`, `impersonation` hoặc `other`. Với `NOT_SCAM`, ghi `none`. `intent` mô tả hành động mà SMS muốn người nhận thực hiện; `label_reason` phải là lý do ngắn dựa trên nội dung, không phải tên từ khóa.

## Quy trình xác minh

1. Hai người gán nhãn độc lập khi có thể.
2. Bất đồng hoặc thiếu ngữ cảnh để trống `final_label` hoặc dùng `NEEDS_REVIEW`.
3. Người adjudicate ghi `final_label`, `review_status=VERIFIED` hoặc `GOLD_VERIFIED`, confidence và lý do.
4. Chỉ các mẫu `SCAM`/`NOT_SCAM` đã verified mới được huấn luyện. Feedback ứng dụng không tự động trở thành nhãn.
5. Mẫu tổng hợp phải ghi rõ `source_type=synthetic`, không được trình bày như SMS thực tế và không được đưa vào test gold.

Không sửa/xóa bản raw. Không dùng dự đoán của model hoặc heuristic URL để tự động gán nhãn.
