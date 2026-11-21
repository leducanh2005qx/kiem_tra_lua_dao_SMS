# PhoBERT Training Report – Phase 3

## Trạng thái

**NOT RUN – chờ dataset bốn lớp verified/locked và môi trường Google Colab GPU.**

Notebook và pipeline export đã được chuẩn bị tại `notebooks/colab/01_phobert_4class_training.ipynb`. Không có kết quả GPU, checkpoint, metric validation/test hoặc tuyên bố PhoBERT tốt hơn Classical ML ở thời điểm này.

## Lý do chưa huấn luyện

- Phase 1 hiện có `verified_rows=0`.
- `data/processed/train.csv`, `validation.csv` và `test.csv` chưa có dữ liệu bốn lớp.
- Chưa có `data/manifests/dataset_lock.json` hợp lệ.
- Máy phát triển Windows không được dùng để giả lập GPU Colab.

## Pipeline đã chuẩn bị

- GPU/device detection, `nvidia-smi`, PyTorch/CUDA/VRAM logging.
- `vinai/phobert-base`, `num_labels=4`, mapping nhãn cố định.
- VnCoreNLP RDRSegmenter, Unicode/diacritics checks và segmentation cache.
- Weighted CrossEntropyLoss theo phân bố train, AdamW, warmup, mixed precision, gradient accumulation, clipping, early stopping và best checkpoint theo validation Macro-F1.
- Đánh giá validation trước; chỉ sau khi chốt checkpoint mới đánh giá test khóa.
- Error analysis, confusion matrix, high-risk metrics, export manifest và CPU smoke test.

## Khi có dữ liệu

Đọc hướng dẫn tại `notebooks/colab/README.md`, chạy notebook trên Colab GPU, sau đó đối chiếu `results/benchmark/phobert_vs_classical_ml.csv`. Không dùng test để chọn epoch và không tuyên bố mô hình tốt hơn nếu benchmark cùng dataset chưa chứng minh điều đó.
