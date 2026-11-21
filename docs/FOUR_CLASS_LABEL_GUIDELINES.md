# Quy chuẩn gán nhãn bốn lớp

## Mục tiêu

Quy chuẩn này dùng cho human annotation sau Phase 0. Không dùng để tự động chuyển nhãn nhị phân hiện tại. Mỗi dòng phải có provenance và trạng thái chắc chắn.

| Mã | Nhãn | Định nghĩa ngắn |
|---:|---|---|
| 0 | `LEGITIMATE` | SMS hợp lệ hoặc cảnh báo an toàn/dịch vụ thật |
| 1 | `PROMOTION` | Quảng cáo, ưu đãi, giới thiệu sản phẩm hợp lệ; chưa có mục đích chiếm đoạt rõ |
| 2 | `PHISHING` | Mạo danh/dụ lấy OTP, mật khẩu, thông tin tài khoản hoặc thông tin cá nhân |
| 3 | `FINANCIAL_SCAM` | Dụ chuyển tiền, đóng phí, đầu tư, vay, giả thưởng hoặc chiếm đoạt tài sản |

## 0 – LEGITIMATE

Bao gồm:

- OTP hoặc thông báo giao dịch hợp lệ từ dịch vụ thật, khi nội dung chỉ thông báo và không yêu cầu gửi OTP/mật khẩu cho người khác.
- Giao hàng, trường học, công việc, lịch hẹn, thông báo dịch vụ.
- Cảnh báo an toàn/chống lừa đảo.
- URL chính thức hoặc hướng dẫn liên hệ qua kênh chính thức, nếu không có dấu hiệu chiếm đoạt.

Các câu sau là `LEGITIMATE` nếu đứng độc lập và không có ngữ cảnh giả mạo:

- “Ngân hàng không yêu cầu cung cấp OTP.”
- “Không truy cập liên kết lạ.”
- “Nhân viên không bao giờ yêu cầu mật khẩu.”

Không gán `PHISHING` chỉ vì SMS có từ “OTP”, “ngân hàng” hoặc có URL.

## 1 – PROMOTION

Bao gồm quảng cáo, khuyến mại, giới thiệu gói cước/sản phẩm, mã giảm giá hoặc lời mời dịch vụ. Một tin quảng cáo chưa được xem là scam chỉ vì không mong muốn.

Gán `PROMOTION` khi mục tiêu chính là tiếp thị và không có yêu cầu bất thường như chuyển tiền trước, cung cấp credential hoặc đe dọa khóa tài khoản.

Nếu tin quảng cáo gắn với link giả, yêu cầu OTP/mật khẩu hoặc khoản phí để nhận lợi ích, chuyển sang `PHISHING` hoặc `FINANCIAL_SCAM` theo mục đích chính.

## 2 – PHISHING

Mục tiêu chính là thu thập thông tin để chiếm quyền truy cập hoặc mạo danh:

- Giả danh ngân hàng, cơ quan, nhà mạng, sàn thương mại hoặc dịch vụ.
- Yêu cầu OTP, mật khẩu, mã xác thực, số thẻ, CCCD hoặc đăng nhập.
- Dùng URL không chính thức, gây áp lực “xác minh ngay”, “tài khoản bị khóa”.
- Dụ người nhận mở trang đăng nhập giả.

Một URL xấu hoặc một từ khóa riêng lẻ không đủ để gán nhãn; phải xét mối quan hệ giữa chủ thể, yêu cầu hành động và hậu quả được nêu.

## 3 – FINANCIAL_SCAM

Mục tiêu chính là chiếm đoạt tiền/tài sản hoặc bắt người nhận thực hiện giao dịch đáng ngờ:

- Yêu cầu chuyển khoản, nộp phí trước, đóng phạt, “mở khóa” bằng tiền.
- Vay/đầu tư với cam kết lợi nhuận bất thường và yêu cầu tiền trước.
- Giả thưởng nhưng yêu cầu phí nhận thưởng hoặc phí xác minh.
- Mạo danh đơn vị tài chính để ép giao dịch.

Nếu SMS vừa lấy OTP vừa yêu cầu chuyển tiền, chọn `FINANCIAL_SCAM` khi hành động chuyển tiền là mục tiêu trực tiếp; chọn `PHISHING` khi trọng tâm là lấy credential/OTP để chiếm tài khoản và chưa có yêu cầu chuyển tiền cụ thể.

## Quy tắc trường hợp giao nhau

1. Xác định mục đích chính và hành động mà người nhận bị thúc ép thực hiện.
2. Phân biệt “thông báo/cảnh báo” với “yêu cầu cung cấp/thực hiện”.
3. Không dùng từ khóa riêng lẻ làm nhãn.
4. Nếu thiếu ngữ cảnh, ghi `annotation_status=uncertain`, để trống label và đưa vào `REVIEW_QUEUE`.
5. Không tự suy đoán nhãn thật từ URL heuristic hoặc dự đoán của model nhị phân.
6. Ghi `notes` nêu bằng chứng và điểm còn mơ hồ.

## Các mẫu phải được kiểm tra cẩn thận

- Câu phủ định: “Không cung cấp OTP cho bất kỳ ai.” thường là `LEGITIMATE`.
- OTP hợp lệ: “Mã OTP của bạn là …” không tự động là phishing.
- URL hợp lệ: URL chính thức không tự động là an toàn tuyệt đối, nhưng cần xem toàn bộ ngữ cảnh.
- Ngân hàng: thông báo giao dịch không yêu cầu hành động khác với cảnh báo đăng nhập giả.
- Công việc/học tập: thông báo lịch, điểm, ca làm không phải promotion/scam chỉ vì có link.
- Quảng cáo hợp lệ: ưu đãi sản phẩm không phải financial scam nếu không có hành vi chiếm đoạt.
- Tin tài chính: “đóng phí trước”, “chuyển tiền để nhận thưởng”, “nộp phạt ngay” là tín hiệu cần xem xét `FINANCIAL_SCAM`.

## Trường dữ liệu annotation tối thiểu

```text
message,label,annotation_status,confidence,annotator,notes
```

- `label`: một trong bốn mã tên nêu trên hoặc để trống khi uncertain.
- `annotation_status`: `certain` hoặc `uncertain`.
- `confidence`: mức tin cậy do annotator ghi rõ.
- `annotator`: mã người gán nhãn, không ghi thông tin cá nhân không cần thiết.
- `notes`: lý do, nguồn và bất đồng cần adjudicate.

Chỉ mẫu `certain` đã qua kiểm tra chất lượng mới được xem xét cho train/validation. `REVIEW_QUEUE` và feedback website không tự động trở thành nhãn chuẩn.
