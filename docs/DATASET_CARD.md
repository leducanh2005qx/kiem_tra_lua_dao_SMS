# SMS Shield VN – Dataset Card Phase 1

## Trạng thái

Đây là card cho pipeline dữ liệu bốn lớp, không phải tuyên bố rằng repository đã có đủ dữ liệu bốn lớp. Phase 1 hiện tạo candidate và annotation queue; dữ liệu processed chỉ chứa mẫu đã được người kiểm duyệt xác nhận.

## Nhãn

`0=LEGITIMATE`, `1=PROMOTION`, `2=PHISHING`, `3=FINANCIAL_SCAM`. Nhãn nhị phân cũ không được tự động ánh xạ.

## Nguồn và provenance

- Vietnamese Quality-Assured SMS Dataset: nguồn binary, `real`, CC BY 4.0; giữ URL và nhãn gốc.
- Vietnamese Fraud Detection: nguồn phái sinh/heuristic-derived; license dữ liệu cần xác minh trước khi nhập.
- VNPhish: nguồn synthetic theo dataset card, CC BY-NC 4.0; không mặc định dùng cho production hoặc commercial.

Các nguồn có quan hệ nguồn gốc không được coi là test độc lập. Mỗi mẫu có `source_name`, `source_url`, `license`, `source_type`, `original_label`, `raw_label` và `group_id`.

## Riêng tư và tiền xử lý

Bản raw được giữ ở khu vực raw/interim theo quyền truy cập. Bản processed thêm `message_anonymized`: domain URL được giữ như đặc trưng, email/số điện thoại/số nhận dạng dài được ẩn danh; OTP ngắn và số tiền không bị xóa máy móc. Unicode dùng NFC, giữ dấu tiếng Việt và giữ từ phủ định.

## Split và giới hạn

Split dùng group-aware stratification khi có đủ dữ liệu verified. Synthetic chỉ được đưa vào train. Gold test và robustness test là các vai trò riêng, không dùng để điều chỉnh mô hình. Nếu chưa có mẫu verified, các file split được tạo với header rỗng và báo cáo ghi rõ thiếu dữ liệu.

## Không được suy diễn

Số lượng mục tiêu gold test 100–200 mẫu mỗi lớp là mục tiêu dự án, không phải số liệu hiện có. Không tự sinh hoặc tái sử dụng mẫu để tuyên bố đã đạt mục tiêu.
