# Multiclass Data Migration Report

## Trạng thái

**BLOCKED – chưa thể chuyển dữ liệu nhị phân cũ thành dataset 4 lớp đã xác minh.**

Nguồn nhị phân hiện tại chỉ công bố `0 = hợp lệ` và `1 = rác/nghi vấn lừa đảo`. Nhãn `1` trộn spam, quảng cáo và nghi vấn nên không có căn cứ ánh xạ tự động sang `PROMOTION`, `PHISHING` hoặc `FINANCIAL_SCAM`.

## Số liệu audit hiện tại

Các con số được sinh từ `data/annotations/annotation_queue.csv` và lưu trong `results/multiclass/training_report.json`.

| Hạng mục | Số lượng |
|---|---:|
| Candidate nhị phân đã nhập | 2.991 |
| Mẫu có thể ánh xạ tự động | 0 |
| Mẫu đã verified 4 lớp | 0 |
| Mẫu cần gán nhãn thủ công/review | 2.991 |
| Mẫu bị loại do lỗi chất lượng | 0 được tự động loại khỏi queue |
| LEGITIMATE verified | 0 |
| PROMOTION verified | 0 |
| PHISHING verified | 0 |
| FINANCIAL_SCAM verified | 0 |

Queue hiện giữ nguyên provenance, raw label và các duplicate để người gán nhãn xử lý có kiểm soát. Báo cáo duplicate hiện có 978 dòng cảnh báo; không dùng các bản sao gần nhau đồng thời ở train/test.

## Hạn chế

- Chưa có hai bộ annotation độc lập nên chưa tính được agreement/kappa.
- Chưa có phân bố đủ bốn lớp, hard negatives/hard positives và gold test độc lập.
- Chưa thể đánh giá 3 model 4 lớp hoặc chọn model production.
- Không được dùng các ví dụ tổng hợp chưa verified để thay thế dữ liệu thực.

## Bước tiếp theo

Gán nhãn theo `docs/FOUR_CLASS_LABELING_GUIDE.md`, chạy `python scripts/dataset_pipeline.py prepare`, kiểm tra lock/leakage, rồi chạy `python scripts/train_multiclass_models.py`. Model chỉ được lưu vào `models/multiclass/` sau khi toàn bộ điều kiện verified đạt.
