# Tham chiếu rule rủi ro

Rule được đặt trong `src/risk_config.py`; logic chạy trong
`src/risk_analyzer.py`.

## Nhóm rule

- `PHISHING_CREDENTIAL_REQUEST`: OTP, mật khẩu, thông tin thẻ hoặc thông tin cá nhân đi kèm hành động cung cấp/nhập/xác minh đáng ngờ.
- `FINANCIAL_PAYMENT_REQUEST`: chuyển tiền, đóng phí, nạp tiền, đặt cọc đi kèm thưởng, đầu tư, hoa hồng hoặc áp lực.
- `PROMOTION_CONTENT`: giảm giá, voucher, ưu đãi, tặng data, chương trình khách hàng khi không có tín hiệu nhạy cảm.
- `SUSPICIOUS_URL`: URL giả mạo thương hiệu, path/query nhạy cảm, IP literal, Punycode hoặc cấu trúc bất thường.
- `SHORTENED_URL`: URL rút gọn, chỉ là tín hiệu cần xem lại nếu chưa có ngữ cảnh khác.

Rule xét quan hệ hành động và đối tượng. Câu phủ định như “Không cung cấp mã
OTP cho bất kỳ ai” không tạo rule yêu cầu OTP.

## Trạng thái URL

- `OFFICIAL_URL`: hostname khớp domain tham chiếu theo ranh giới chính xác.
- `SUSPICIOUS_URL`: có dấu hiệu nhái thương hiệu hoặc cấu trúc nguy hiểm.
- `SHORTENED_URL`: shortener chưa được mở rộng qua mạng.
- `MALFORMED_URL`: URL lỗi cú pháp, bao gồm IPv6 lỗi; không được coi là an toàn.
- `UNKNOWN_URL`: parse được nhưng không thuộc danh sách tham chiếu.

`OFFICIAL_URL` không chứng minh toàn bộ SMS an toàn. Domain tham chiếu chỉ là
một tín hiệu, không được dùng để hạ cảnh báo của nội dung yêu cầu OTP hoặc tiền.
