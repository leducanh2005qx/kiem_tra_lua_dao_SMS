# Final Model Selection Report – Phase 4

## Trạng thái

**BLOCKED – chưa có artifact bốn lớp hợp lệ để calibration, robustness hoặc production selection.**

Không có model nào được đánh dấu production. Model nhị phân hiện tại trên website vẫn được giữ nguyên.

Lý do: `no_complete_four_class_model_artifact, dataset_lock_missing, phobert_artifact_missing`.

## Quy tắc đã triển khai

- Chỉ artifact có manifest, label mapping, dataset/preprocessing version, training config và metrics mới được registry.
- Temperature scaling chỉ được fit trên validation/calibration split, không dùng test.
- Confidence và top-1/top-2 margin được chọn trên validation; trạng thái REVIEW không phải lớp thứ năm.
- Robustness dùng test biến thể riêng và không được dùng để điều chỉnh model/threshold.
- Production ưu tiên Macro-F1, high-risk recall/precision, false alarm, calibration, selective risk, latency và kích thước.

## Kết quả hiện tại

Chưa có metric calibration, robustness, latency hoặc model comparison hợp lệ. Không bịa số liệu khi Phase 2/3 chưa tạo artifact bốn lớp.

## Khi đủ điều kiện

Chạy `python scripts/phase4_selection.py audit`, sau đó chạy các benchmark Phase 2 và notebook Phase 3 trên cùng dataset lock. Chỉ sau khi toàn bộ unit/integration/model-loading/reject-option tests đạt mới sửa registry để có `production_model`.
