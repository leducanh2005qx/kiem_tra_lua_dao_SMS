# Human-in-the-loop

## Phản hồi trong ứng dụng

Sau mỗi lần kiểm tra SMS, mở **Phản hồi kết quả chưa chính xác?** và chọn:

- Kết quả đúng.
- Kết quả chưa đúng, sau đó chọn một trong bốn nhóm hoặc `NEEDS_REVIEW`.
- Chưa chắc chắn.

Phản hồi mặc định có trạng thái `pending_review`. Người quản trị có thể xem ở
trang **Quản lý phản hồi**, sau đó chuyển sang `verified`, `rejected` hoặc
`duplicate`.

SQLite lưu liên kết đến SMS, nhãn ML, nhãn Hybrid, điểm và loại điểm, rule,
URL, bằng chứng, lý do phản hồi và thời gian. Nội dung SMS chỉ tồn tại trong
lịch sử khi người dùng đã đồng ý lưu.

## Dùng cho huấn luyện

Phản hồi chưa xác minh không được đưa vào dataset. Có thể xuất từ trang quản lý
phản hồi hoặc lịch sử để kiểm tra thủ công trước khi nhập vào hàng đợi gắn nhãn.
Không train tự động ngay sau khi gửi phản hồi.

## Quyền riêng tư

Database nằm cục bộ trong `data/app_history.db`. Không gửi SMS ra ngoài, không
truy cập URL và không ghi OTP/mật khẩu vào log kỹ thuật. Có thể xóa lịch sử từ
trang **Lịch sử phân tích**.
