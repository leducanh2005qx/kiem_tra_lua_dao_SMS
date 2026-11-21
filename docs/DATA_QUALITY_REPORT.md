# Data Quality Report – Phase 1

Báo cáo này được tạo lại bằng:

```text
python scripts/dataset_pipeline.py audit
```

Các bảng số liệu ở `data/manifests/` gồm:

- `dataset_manifest.csv`: manifest theo mẫu, hash, provenance, group và trạng thái duyệt; không chứa lại nội dung SMS.
- `dataset_statistics.csv`: kích thước, rỗng, Unicode, URL/OTP/tiền và loại nguồn.
- `duplicate_report.csv`: exact normalized duplicate và nearest near-duplicate theo character TF-IDF.
- `label_audit.csv`: trạng thái gán nhãn, lỗi nhãn, điều kiện trainable.
- `class_distribution.csv`: candidate và verified theo bốn lớp.
- `source_distribution.csv`: phân bố theo nguồn, loại nguồn, nhãn gốc và trạng thái duyệt.
- `split_summary.csv`: phân bố theo train/validation/test/gold_test/robustness_test.
- `annotation_agreement.json`: số dòng có hai nhãn, agreement và Cohen's Kappa.

## Diễn giải an toàn

Candidate chưa được xem là dữ liệu huấn luyện. `UNMAPPED`, `REVIEW` và mẫu heuristic-derived chưa được người kiểm duyệt lại sẽ không xuất hiện trong processed train/validation/test. Near duplicate chỉ là cảnh báo để xem xét grouping; pipeline không tự xóa dữ liệu hoặc thay đổi nhãn gốc.

Nếu không có đủ dữ liệu verified cho bốn lớp, báo cáo phải giữ số thực tế và để split tương ứng rỗng. Đây là trạng thái hợp lệ của pipeline, không phải lỗi cần bịa thêm mẫu.
