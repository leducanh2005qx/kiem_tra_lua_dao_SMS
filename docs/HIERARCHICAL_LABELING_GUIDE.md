# Hướng dẫn gắn nhãn phân cấp

SMS Shield VN dùng hai lớp nhãn độc lập:

- **Lớp 1:** `NOT_SCAM` hoặc `SCAM`.
- **Lớp 2:** chỉ áp dụng khi lớp 1 là `SCAM`: `PHISHING`, `FINANCIAL_SCAM` hoặc `OTHER_SCAM`.
- **Nhãn phụ:** có thể chọn nhiều ý định như yêu cầu OTP, yêu cầu chuyển tiền, giả mạo ngân hàng, dụ đóng phí và đường dẫn đáng ngờ.

`LEGITIMATE` và `PROMOTION` được lưu ở nhánh `NOT_SCAM`. `NEEDS_REVIEW` không được huấn luyện.

## Gắn nhãn nhanh

1. Mở trang **GẮN NHÃN DỮ LIỆU HUẤN LUYỆN**.
2. Xem gợi ý của máy.
3. Chọn **Đúng – xác nhận nhanh**, **Sai – sửa lại** hoặc **Chưa chắc chắn**.
4. Bấm **Xác nhận và chuyển SMS tiếp theo**.

Gợi ý máy không trở thành nhãn xác minh nếu người dùng chưa bấm xác nhận.

## Kiểm tra điều kiện huấn luyện

Chạy:

```powershell
python scripts/train_hierarchical_models.py --min-per-class 20
```

Lệnh chỉ tạo model khi có đủ mẫu đã xác minh cho cả lớp 1 và các loại lừa đảo lớp 2. Model mới được lưu trong `models/hierarchical/`, không ghi đè model nhị phân cũ.
