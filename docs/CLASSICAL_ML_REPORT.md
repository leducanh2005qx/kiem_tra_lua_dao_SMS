# Classical ML Report – Phase 2

## Trạng thái

**BLOCKED – chưa có dataset bốn lớp verified và locked để huấn luyện.**

Pipeline đã được triển khai nhưng không chạy model và không tạo số liệu giả. Phase 1 hiện có candidate nhị phân ở trạng thái `UNMAPPED`, nên không được dùng cho benchmark bốn lớp.

Lý do kiểm tra: `dataset_lock_missing, empty_test, empty_train, empty_validation, missing_class_test, missing_class_train, missing_class_validation`.

## Đã triển khai

- MultinomialNB, ComplementNB, Logistic Regression và Linear SVM.
- Word unigram/bigram/trigram, character 2–5/3–5 và hybrid word-character TF-IDF.
- Grid alpha, C, class weight, min_df, max_df; calibrated SVM dùng CV nội bộ trên train.
- Metrics multiclass, PR-AUC OvR, high-risk binary, confusion matrix, lỗi và feature importance.
- Vectorizer chỉ fit trong pipeline trên train; validation dùng để chọn cấu hình; test chỉ đánh giá sau khóa.

## Số liệu chưa được công bố

Chưa có model tốt nhất trên validation, kết quả test, confusion matrix hoặc feature importance hợp lệ vì ba split train/validation/test đang rỗng. Việc ghi các con số lúc này sẽ là bịa số liệu.

Các file benchmark đã được tạo với header trong `results/classical_ml/`; `model_comparison.csv` chỉ ghi trạng thái BLOCKED, không ghi metric giả.

## Cách kích hoạt sau khi annotation

1. Điền annotation và chạy `python scripts/dataset_pipeline.py prepare`.
2. Kiểm tra leakage rồi chạy `python scripts/classical_ml_benchmark.py lock`.
3. Chạy `python scripts/classical_ml_benchmark.py run`.

Không xóa hoặc thay thế model nhị phân hiện tại.
