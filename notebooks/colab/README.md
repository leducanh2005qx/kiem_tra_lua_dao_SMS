# PhoBERT 4-class Google Colab

Notebook: `01_phobert_4class_training.ipynb`.

## Điều kiện trước khi chạy

1. Hoàn tất annotation Phase 1.
2. Chạy `python scripts/dataset_pipeline.py prepare` trên máy phát triển.
3. Chạy `python scripts/classical_ml_benchmark.py lock` thành công.
4. Copy toàn bộ repository và `data/manifests/dataset_lock.json` lên Google Drive hoặc Colab.

Notebook không tự tải/gộp dataset mới, không tự đổi nhãn, không dùng test để chọn epoch. Nếu split chưa có đủ bốn lớp verified hoặc lock hash không khớp, notebook dừng trước model loading/training.

## Chạy trên Colab

1. Runtime → Change runtime type → chọn GPU.
2. Mở notebook.
3. Sửa `PROJECT_ROOT` thành thư mục repository trên Drive/Colab.
4. Chạy lần lượt từ đầu đến cuối.
5. Kiểm tra GPU verification và log train trước khi chờ kết quả.

Checkpoint và export được lưu tại:

```text
MyDrive/SMS_Shield_VN/experiments/phobert/
```

Notebook dùng `vinai/phobert-base` (MIT theo model card được tham chiếu trong cấu hình). `vinai/phobert-base-v2` chỉ là biến thể nghiên cứu cần legal review vì AGPL-3.0.

## Tải model về Windows

Sau khi notebook export thành công, copy thư mục `models/phobert_4class/` về repository. Không copy dữ liệu SMS cá nhân cùng model. Có thể kiểm tra artifact bằng:

```powershell
python scripts/validate_phobert_export.py models/phobert_4class
```

Inference CPU chỉ được xác nhận khi máy Windows đã cài dependency transformer phù hợp và có model/tokenizer local.
