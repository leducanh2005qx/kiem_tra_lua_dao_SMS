# Kiến trúc Hybrid của SMS Shield VN

## Luồng xử lý

```text
SMS gốc
  -> model ML nhị phân hiện tại
  -> Risk Analyzer ngoại tuyến
  -> Hybrid Decision Engine
  -> kết quả 4 nhóm hoặc NEEDS_REVIEW
  -> lưu lịch sử và nhận phản hồi người dùng
```

Model cũ vẫn trả về hai nhãn: `0` hợp lệ và `1` rác/nghi vấn. Kết quả đó
được giữ nguyên trong `ml_label`, `ml_score` và `ml_score_type`. Risk Analyzer
không thay đổi trọng số hoặc pipeline của model.

Kết quả Hybrid có bốn nhóm: `LEGITIMATE`, `PROMOTION`, `PHISHING` và
`FINANCIAL_SCAM`. `NEEDS_REVIEW` là trạng thái từ chối kết luận khi thiếu căn
cứ, không phải lớp huấn luyện thứ năm.

## Ưu tiên quyết định

1. Yêu cầu OTP/mật khẩu/thông tin tài khoản đáng ngờ: `PHISHING`.
2. Yêu cầu chuyển tiền, đóng phí, nạp tiền hoặc đặt cọc gắn với lời dụ dỗ/áp lực: `FINANCIAL_SCAM`.
3. Quảng cáo rõ ràng nhưng không có hành vi nhạy cảm: `PROMOTION`.
4. ML dự đoán hợp lệ và không có tín hiệu rủi ro: `LEGITIMATE`.
5. Tín hiệu xung đột, URL rút gọn đơn độc hoặc ML nghi vấn nhưng thiếu loại cụ thể: `NEEDS_REVIEW`.

Mọi override đều ghi lại `rule_override`, lý do, rule, URL và bằng chứng trong
SQLite. Không có request mạng, DNS lookup hoặc mở rộng short URL.
