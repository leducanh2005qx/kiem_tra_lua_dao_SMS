# SMS Shield VN

Dự án phân loại SMS tiếng Việt với hai nhãn:

- `0`: SMS hợp lệ.
- `1`: SMS rác hoặc nghi vấn lừa đảo. Nhãn này không phải bằng chứng chắc chắn của hành vi lừa đảo.

## Chạy nhanh trên Windows

```powershell
cd SMS_Scam_Detector_NLP
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m pytest -q
streamlit run app.py
```

Dữ liệu thật đã có trong `data/raw/`. Nếu cần tải lại từ nguồn chính thức, chạy `python download_data.py`. Không dùng `full_dataset.csv` để thay thế cặp train/test hiện có.

## Quy trình thực nghiệm

1. Đọc CSV UTF-8 BOM, kiểm tra cột, nhãn, dòng rỗng, Unicode và exact duplicate.
2. Chuẩn hóa Unicode NFC, lowercase và khoảng trắng; giữ dấu tiếng Việt, URL, OTP, số tiền/điện thoại và placeholder. Teencode hỗn hợp như `T@i kh0an b4n` được chuẩn hóa bảo thủ; exact overlap giữa train/test được loại khỏi train.
3. Chia train đã làm sạch thành train-fit/validation theo stratified split, `random_state=42`. Vectorizer nằm trong `Pipeline`, nên không học từ validation hoặc test.
4. Giữ `nb_word_1` với `alpha=1.0`, `fit_prior=True` làm NB baseline. Thử 30 NB variants: `alpha ∈ {0.1, 0.3, 0.5, 1.0, 2.0}`, `fit_prior ∈ {True, False}`, word unigram, word 1–2 gram và character 3–5 gram.
5. Giữ Linear SVM làm đối chứng với ba vectorizer (`word_1`, `word_12`, `char_35`), `C ∈ {0.1, 0.5, 1.0, 2.0}` và `class_weight ∈ {None, balanced}`. Chọn NB tối ưu và SVM theo Macro-F1, sau đó recall/precision nhãn 1 trên validation; test chỉ được dùng để báo cáo cuối.

## Artifact sinh ra

- `models/nb_word_1.joblib`: NB baseline.
- `models/nb_optimized.joblib`: NB được chọn bằng validation.
- `models/svm.joblib`: Linear SVM được chọn (phương thức cụ thể nằm trong metadata).
- `models/metadata.json`: cấu hình, checksum dữ liệu và mapping artifact.
- `results/validation_metrics.csv`: toàn bộ grid NB và các ứng viên SVM.
- `results/test_metrics.csv`: Accuracy, Precision/Recall macro, Macro-F1 và Precision/Recall/F1 riêng cho nhãn 1.
- `results/test_predictions.csv`: dự đoán trên test; `results/error_analysis.csv` và `results/error_examples.csv` dùng nội dung đã che URL/số định danh thông dụng.
- `results/model_comparison.csv`: bảng so sánh cuối trên test, gồm Macro-F1, recall nhãn 1, false negative và thời gian fit.
- `results/confusion_matrix/`: confusion matrix từng model; `results/feature_explanations/`: đặc trưng trọng số hàng đầu của NB/SVM.
- `results/robustness_test_results.csv`: bộ kiểm thử biến thể độc lập, không được đưa vào train; `results/misclassified_samples.csv`: mẫu lỗi đã che định danh.
- `results/figures/`: phân bố lớp và biểu đồ so sánh Macro-F1.
- `results/quality_report.json`: chất lượng dữ liệu và các kiểm soát leakage.

Kết quả cũ được sao lưu tại `artifacts_archive/pre_optimization_2026-11-19/`; bản sao mã nguồn trước hoàn thiện Phase 1 ở `artifacts_archive/pre_phase1_completion_2026-11-21/`.

## Streamlit và Phase 2

Khởi động bằng:

```powershell
streamlit run app.py
```

Thanh điều hướng gồm: Tổng quan, Kiểm tra SMS, Kiểm tra hàng loạt, Lịch sử, Phản hồi, Thống kê mô hình và Giới thiệu.

- Tổng quan lấy số liệu hoạt động thật từ SQLite, hỗ trợ 7/30/90 ngày hoặc toàn thời gian; database trống hiển thị 0/trạng thái rỗng.
- Kiểm tra SMS giữ lựa chọn model và giải thích đặc trưng của Phase 1. Nội dung không được lưu mặc định; có checkbox đồng ý lưu.
- CSV yêu cầu cột `message`, hỗ trợ UTF-8/BOM, giới hạn 5 MB/10.000 dòng, không lưu file upload tự động và có lựa chọn lưu lịch sử.
- Lịch sử hỗ trợ tìm kiếm, lọc ngày/nhãn/model, phân trang, xem chi tiết, xóa một bản ghi, xóa toàn bộ sau xác nhận và xuất CSV.
- Phản hồi có trạng thái `pending`, `reviewed`, `rejected`; không tự động trở thành nhãn huấn luyện.
- Thống kê mô hình đọc riêng `results/test_metrics.csv` và confusion matrix, không trộn với số liệu hoạt động website.

SQLite nằm tại `data/app_history.db`. Schema gồm `analysis_history` và `prediction_feedback`; mọi truy vấn dùng parameterized SQL. Nội dung SMS chỉ được ghi khi người dùng đồng ý. Sao lưu/xóa:

```powershell
Copy-Item data\app_history.db data\app_history_backup.db
Remove-Item data\app_history.db
```

## Phase 1: pipeline dữ liệu bốn lớp

Pipeline Phase 1 nằm ở `scripts/dataset_pipeline.py` và không tự chuyển nhãn nhị phân sang bốn lớp:

```powershell
python scripts/dataset_pipeline.py init
python scripts/dataset_pipeline.py rebuild-queue
python scripts/dataset_pipeline.py prepare
python scripts/dataset_pipeline.py audit
```

Nhập nguồn CSV/JSON/JSONL vào queue bằng provenance đầy đủ:

```powershell
python scripts/dataset_pipeline.py import --input <file.csv> --source-name <name> --source-type real --source-url <url> --license "<license>"
python scripts/dataset_pipeline.py rebuild-queue
```

Để tải nguồn chính thức đã có URL file ổn định, dùng `download --source official_binary`; với GitHub/VNPhish cần tải thủ công sau khi kiểm tra license rồi dùng lệnh `import`. Chỉ dòng có `final_label` hợp lệ và `review_status` verified mới được xuất vào `data/processed/`. Hướng dẫn chi tiết ở `docs/ANNOTATION_GUIDE.md`, dataset card ở `docs/DATASET_CARD.md`, và báo cáo sinh ở `docs/DATA_QUALITY_REPORT.md`.

Hiện queue đã chứa snapshot binary local nhưng tất cả là `UNMAPPED`; vì vậy các split bốn lớp hiện chỉ có header. Đây là trạng thái an toàn và chính xác cho đến khi có human annotation.

## Phase 2: classical ML bốn lớp

Benchmark nằm ở `scripts/classical_ml_benchmark.py`. Lệnh `run` chỉ chạy khi train/validation/test bốn lớp đã có dữ liệu verified, không có overlap theo normalized text/group và đã có `data/manifests/dataset_lock.json`:

```powershell
python scripts/classical_ml_benchmark.py lock
python scripts/classical_ml_benchmark.py run
```

Hiện lệnh trả về `BLOCKED` vì Phase 1 chưa có mẫu verified; không có metric giả. Khi đủ dữ liệu, benchmark sẽ chọn cấu hình bằng validation, fit lại trên train+validation và chỉ đánh giá test sau đó. Kết quả nằm ở `results/classical_ml/`, báo cáo ở `docs/CLASSICAL_ML_REPORT.md`. Các model nhị phân cũ và model đang dùng trên website không bị thay thế.

## Phase 3: URL, đa lớp, PhoBERT và API

- `src/url_analyzer.py` trích xuất và phân tích URL hoàn toàn offline: hostname, path, query, IP literal, Punycode, URL rút gọn, `@`, mã hóa và dấu hiệu mạo danh thương hiệu. Heuristic không được cộng vào điểm model, không mở URL và không kết luận URL chắc chắn lừa đảo.
- Dữ liệu bốn lớp verified chưa tồn tại trong repository. Schema queue Phase 1 nằm ở `data/annotations/annotation_template.csv` và hướng dẫn ở `docs/ANNOTATION_GUIDE.md`. Chỉ dòng verified có đủ bốn nhãn mới được huấn luyện; nhãn nhị phân cũ không được tự chuyển đổi.
- PhoBERT nằm trong `src/phobert.py`, dependency tách tại `requirements-transformers.txt`, hỗ trợ word segmentation, CPU/GPU, checkpoint cục bộ, early stopping và metadata. Chưa tải/huấn luyện vì môi trường hiện tại không có Torch/Transformers/Underthesea và chưa có dataset đa lớp đáng tin cậy.
- FastAPI tùy chọn nằm ở `api/main.py`, dependency tại `requirements-api.txt`. Tài liệu endpoint: `docs/phase3/API.md`.
- Docker: `Dockerfile`, `docker-compose.yml`, `.dockerignore`. Hướng dẫn triển khai: `docs/phase3/DEPLOYMENT.md`.

Kiểm tra URL và trạng thái các module mới nằm trong trang **URL & hệ thống** của Streamlit. Module chưa đủ dependency/dữ liệu sẽ hiển thị trạng thái rõ ràng thay vì tạo nút hoạt động giả.

## Phase 4: calibration, robustness và production selection

Audit registry bằng:

```powershell
python scripts/phase4_selection.py audit
```

Phase 4 chỉ chấp nhận artifact bốn lớp có manifest đầy đủ, dataset lock, metrics validation/test, calibration và threshold đã kiểm tra. Giao diện thống nhất nằm ở `src/prediction_interface.py`; trạng thái `REVIEW` không phải lớp thứ năm. Khi chưa có production candidate, Streamlit tiếp tục dùng model nhị phân cũ và không bị thay thế tự động. Báo cáo ở `docs/FINAL_MODEL_SELECTION_REPORT.md`, registry ở `models/registry/model_registry.json`.

## Nguồn và giới hạn

Dataset: [Vietnamese Quality-Assured SMS Scam Dataset](https://huggingface.co/datasets/trannguyenthaituan/vietnamese_sms_dataset), được mô tả là CC BY 4.0 trong nguồn. Dự án chỉ phân tích text, không xác thực người gửi, domain, giao dịch hoặc tính hợp pháp của SMS. Template gần giống vẫn có thể tồn tại giữa các split; báo cáo hiện chỉ lọc exact duplicate và ghi rõ giới hạn này.
