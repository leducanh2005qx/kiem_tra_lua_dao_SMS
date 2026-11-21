# Hướng dẫn gán nhãn SMS đa lớp

## Nhãn được phép

- `NORMAL`: nội dung giao dịch/thông báo bình thường, không có mục đích quảng cáo rác hoặc chiếm đoạt.
- `AD_SPAM`: quảng cáo không mong muốn, khuyến mại hàng loạt, mời chào dịch vụ nhưng chưa có dấu hiệu chiếm đoạt.
- `PHISHING`: mạo danh hoặc dụ người nhận cung cấp OTP, mật khẩu, thông tin tài khoản, CCCD hoặc đăng nhập vào trang giả.
- `FINANCIAL_SCAM`: yêu cầu chuyển tiền, đóng phí, nộp phạt hoặc hứa lợi ích tài chính theo cách đáng ngờ.

## Quy tắc

1. Đọc toàn bộ SMS, không gán nhãn chỉ dựa trên một từ khóa.
2. Nếu không đủ ngữ cảnh, đặt `annotation_status=uncertain`, để trống `label` và ghi lý do trong `notes`.
3. Chỉ dòng `certain` mới được đưa vào huấn luyện.
4. Không tự chuyển nhãn `0/1` của dataset nhị phân hiện tại thành bốn nhãn.
5. Hai người gán nhãn nên xem độc lập; trường `annotator` dùng để kiểm tra bất đồng.
6. Dữ liệu có thông tin cá nhân cần được ẩn danh trước khi đưa vào file annotation.

File mẫu: `data/multiclass/annotations_template.csv`.
