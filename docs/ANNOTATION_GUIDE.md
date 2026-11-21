# Hướng dẫn annotation bốn lớp – Phase 1

## Nguyên tắc bắt buộc

Pipeline không chuyển nhãn nhị phân `0/1` thành bốn lớp bằng từ khóa hoặc heuristic. Mọi dòng nhập từ nguồn ngoài vẫn giữ `original_label` và `raw_label`, nhưng `final_label` phải để trống cho đến khi có người kiểm duyệt.

Bốn nhãn chính thức:

| ID | Nhãn | Ý nghĩa |
|---:|---|---|
| 0 | `LEGITIMATE` | Tin hợp lệ, dịch vụ, OTP hợp lệ, công việc, học tập, giao hàng hoặc cảnh báo an toàn. |
| 1 | `PROMOTION` | Quảng cáo/ưu đãi hợp lệ, không có mục đích chiếm đoạt rõ ràng. |
| 2 | `PHISHING` | Mạo danh để lấy OTP, mật khẩu, thông tin tài khoản/cá nhân hoặc dụ đăng nhập giả. |
| 3 | `FINANCIAL_SCAM` | Dụ chuyển tiền, đóng phí trước, vay/đầu tư giả, giả thưởng hoặc chiếm đoạt tài sản. |

## Quy trình gán nhãn

1. Đọc toàn bộ nội dung, không gán theo một từ khóa đơn lẻ.
2. Gán độc lập vào `annotator_1_label` và `annotator_2_label` nếu có hai người.
3. Nếu chưa đủ căn cứ, để trống các nhãn cuối và đặt `review_status=REVIEW`.
4. Người kiểm duyệt đặt `final_label` và `review_status=VERIFIED`, `GOLD_VERIFIED` hoặc `ROBUSTNESS_VERIFIED`.
5. Ghi lý do bất đồng, nguồn, tình trạng PII và độ tin cậy vào các cột tương ứng.

Chỉ các dòng có `final_label` hợp lệ và trạng thái verified mới được đưa vào `data/processed/`. Output của AI không được coi là nhãn chuẩn nếu chưa được người duyệt xác nhận.

## Trường hợp dễ nhầm

- “Ngân hàng không yêu cầu cung cấp OTP.”, “Không truy cập liên kết lạ.” và “Nhân viên không bao giờ yêu cầu mật khẩu.” là `LEGITIMATE` khi là cảnh báo an toàn.
- OTP hoặc URL không tự làm mẫu thành lừa đảo; phải đọc mục đích và ngữ cảnh.
- Giả danh ngân hàng để lấy credential là `PHISHING`.
- Yêu cầu chuyển tiền, đóng phí trước, vay/đầu tư hoặc nhận thưởng theo cách đáng ngờ là `FINANCIAL_SCAM`.
- Quảng cáo không có dấu hiệu chiếm đoạt là `PROMOTION`.
- Nếu một tin vừa quảng cáo vừa yêu cầu chuyển tiền/credential, phân loại theo mục đích gây hại chính; nếu không xác định được thì `REVIEW`.

## Hard negative và hard positive

Ưu tiên tạo hàng đợi kiểm tra gồm OTP ngân hàng hợp lệ, cảnh báo chống lừa đảo, giao hàng, tuyển dụng, học tập, quảng cáo hợp lệ và thông báo thanh toán hợp lệ; đối lập với giả ngân hàng lấy OTP, giả giao hàng lấy thông tin, vay yêu cầu phí trước, giả thưởng và đầu tư giả. Mẫu synthetic phải có `source_type=synthetic`, không được mô tả là SMS thực tế, và chỉ được dùng ở train sau khi kiểm duyệt.

## Schema

Schema chính nằm tại `data/annotations/annotation_template.csv` và `annotation_queue.csv`, gồm `sample_id`, `message`, provenance, nhãn gốc, hai nhãn độc lập, `final_label`, `review_status`, `confidence`, `group_id`, `pii_status`, `target_split` và `notes`.

Kappa chỉ được tính trên các dòng có đủ hai nhãn hợp lệ. Nếu chưa có hai bộ nhãn, báo cáo phải ghi rõ là chưa tính được, không thay bằng số giả.
