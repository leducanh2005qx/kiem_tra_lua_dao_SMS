# Phase 0 – Data & Model Audit

**Dự án:** SMS Shield VN  
**Ngày kiểm toán:** 2026-11-22  
**Phạm vi:** đọc mã nguồn, artifact, dữ liệu, SQLite, test và nguồn dữ liệu bên ngoài. Không retrain, không thay model, không đổi nhãn gốc, không gộp dữ liệu.

## 1. Kết luận điều hành

Chưa nên chuyển thẳng sang huấn luyện bốn lớp. Pipeline nhị phân hiện tại hoạt động, nhưng dữ liệu gốc chỉ có hai nhãn `0/1`, trong đó nhãn `1` gộp spam và scam. Có rủi ro leakage đáng kể:

- Train raw: 2.394 dòng; test raw: 597 dòng.
- Sau normalize và loại duplicate, train còn 2.061 dòng; test còn 564 dòng do duplicate và contradictory normalized rows.
- Có 27 overlap raw và 28 overlap sau normalize giữa train/test. Pipeline hiện tại loại 28 mẫu khỏi train trước khi chia validation, nhưng raw split vẫn không độc lập tuyệt đối.
- Kiểm tra nearest-neighbor bằng character TF-IDF trên 2.991 dòng raw cho thấy 942 dòng có láng giềng gần nhất similarity ≥ 0,90 và 744 dòng ≥ 0,95. Đây là tín hiệu cảnh báo template leakage, không phải quyết định tự động xóa dữ liệu.
- SQLite hiện có 18 hoạt động website, trong đó 15 bản ghi đã đồng ý lưu nội dung và 3 bản ghi không lưu nội dung. Các bản ghi này tuyệt đối không được đưa vào train.
- `data/file test csv/sms_predictions.csv` là file người dùng tạo gồm 10 dòng có nội dung SMS và prediction; không phải dataset huấn luyện và cần được coi là dữ liệu nhạy cảm.

**Quyết định:** giữ nguyên model nhị phân hiện tại làm baseline; tạo data manifest và quy trình human annotation trước khi có bất kỳ benchmark bốn lớp nào.

## 2. Tình trạng mã nguồn và chức năng

### Phase 1 đã có

- `train.py`: load CSV, kiểm tra chất lượng, normalize, loại normalized overlap train/test, stratified train/validation, test hold-out.
- `src/preprocess.py`: NFC, lowercase, whitespace, teencode bảo thủ, giữ URL/OTP/placeholder.
- `src/modeling.py`: 30 cấu hình MultinomialNB, 24 cấu hình Linear SVM, word/character TF-IDF, pipeline serialized, giải thích feature.
- `models/`: `nb_word_1.joblib`, `nb_optimized.joblib`, `svm.joblib`, metadata.
- `results/`: validation/test metrics, model comparison, confusion matrix, feature explanations, robustness và misclassified samples.
- Test Phase 1 và Phase 2 vẫn chạy được.

### Phase 2 đã có

- `app.py`: kiểm tra đơn, CSV batch, Dashboard, lịch sử SQLite, feedback, model statistics, privacy controls.
- `src/storage.py`: `analysis_history`, `prediction_feedback`, parameterized SQL, pagination/filter/export/delete.
- Nội dung SMS không lưu mặc định.

### Phase 3 đang có nhưng không phải kết quả audit bốn lớp

- `src/url_analyzer.py`: heuristic URL offline, tách biệt điểm ML.
- `src/multiclass.py`: schema/validator/trainer tùy chọn; chưa có dataset bốn lớp đáng tin cậy.
- `src/phobert.py`: optional path; chưa cài dependency và chưa fine-tune.
- `api/`, Docker và tài liệu deployment là module tùy chọn.

### Tình trạng khác

- Không có notebook `.ipynb` trong repository.
- Không có Git repository ở `F:\XLNM\SMS_Scam_Detector_NLP` hoặc `F:\XLNM`; không thể xác nhận lịch sử commit/nguồn gốc bằng Git.
- Đã sao lưu hiện trạng tại `artifacts_archive/pre_phase0_audit_2026-11-22/`.

## 3. Pipeline, lưu model và kết quả hiện tại

Pipeline được serialize cùng preprocessing và vectorizer; website load từ `models/metadata.json` rồi load ba artifact ổn định. Test hiện tại không retrain trong audit.

| Artifact | Method | Accuracy | Macro-F1 | Recall nhãn 1 | False negative nhãn 1 |
|---|---|---:|---:|---:|---:|
| NB baseline | word unigram, alpha 1.0, fit_prior=True | 0,832 | 0,696 | 0,346 | 87 |
| NB tối ưu | word 1–2 gram, alpha 0.1, fit_prior=False | 0,902 | 0,870 | 0,850 | 20 |
| SVM khuyến nghị | char 3–5, C 0.5, class_weight=balanced | 0,913 | 0,889 | 0,947 | 7 |

Hai vấn đề model cần giữ trong kế hoạch sau:

1. NB baseline thiên về nhãn 0 do mất cân bằng và prior, bỏ sót nhiều nhãn 1.
2. SVM/NB đang học bài toán nhị phân; không thể suy ra `PROMOTION`, `PHISHING` và `FINANCIAL_SCAM` từ score hiện tại.

## 4. Kiểm toán dữ liệu hiện tại

| Chỉ số | Train raw | Test raw |
|---|---:|---:|
| Tổng dòng | 2.394 | 597 |
| Nhãn 0 | 1.754 (73,27%) | 439 (73,53%) |
| Nhãn 1 | 640 (26,73%) | 158 (26,47%) |
| Missing message/label | 0 / 0 | 0 / 0 |
| Duplicate raw rows | 303 | 31 |
| Duplicate sau normalize | 333 | 32 |
| Dòng sau loader hiện tại | 2.061 | 564 |

Phân bố độ dài message:

| Thống kê | Train | Test |
|---|---:|---:|
| Min | 1 | 19 |
| Median | 198,5 | 161,0 |
| Mean | 218,73 | 179,03 |
| P95 | 405,35 | 327,20 |
| P99 | 497,35 | 493,48 |
| Max | 881 | 851 |

Tín hiệu nội dung quan sát được:

- Có URL: train 1.210, test 128.
- Có chuỗi OTP: train 181, test 28.
- Có marker/từ ngữ tiền hoặc chuyển khoản theo heuristic audit: train 1.485, test 334.
- Có placeholder `[MONEY]`, `[NUMBER]`, `[TIME]`, `[DATE]`: train 1.930, test 498.
- Có dấu Unicode/diacritics: train 1.221, test 204.

Các con số trên là mô tả dữ liệu, không phải nhãn mới và không được dùng để tự gán bốn lớp.

### Encoding và provenance

- `train.csv` và `test.csv` đều UTF-8 BOM.
- `data/raw/source_metadata.json` lưu SHA-256, URL nguồn và thời điểm tải.
- Nguồn hiện tại công bố label `0 = legitimate`, `1 = spam/scam`; không phân biệt quảng cáo, phishing và financial scam.
- Raw CSV không có trường provenance từng dòng, loại dữ liệu, người tạo, synthetic flag hoặc source family. Không thể xác định chắc chắn dòng nào là dữ liệu thực tế hay template tổng hợp.

## 5. Leakage và split audit

- Raw train/test overlap: 27 message khác nhau.
- Normalized train/test overlap: 28 message khác nhau.
- Sau loader, 28 dòng train bị loại khỏi train trước khi split.
- Chia validation hiện tại được stratified với seed 42; sau loại overlap, fit có 1.626 dòng, validation có 407 dòng, fit/validation exact overlap bằng 0.
- Tuy nhiên near-duplicate/template overlap vẫn còn: test-to-train nearest char-TF-IDF similarity ≥ 0,90 ở 74 test rows và ≥ 0,95 ở 46 test rows; đây là audit chỉ báo, chưa phải dedup chuẩn cuối.
- Vì raw dataset chỉ có `message,label`, không thể split theo template family, campaign, source hoặc thời gian. Random stratified split có thể làm metric lạc quan.

## 6. SQLite và dữ liệu vận hành

Schema hiện có:

- `analysis_history`: id, created_at, model_name, predicted_label, score, score_type, message_length, optional message_text, optional message_hash.
- `prediction_feedback`: id, analysis_id, feedback_type, suggested_label, comment, created_at, review_status.

Audit xác nhận 18 hoạt động website hiện có và 1 feedback pending. Hoạt động website, file CSV upload và feedback không được tự động trộn vào raw train/test.

## 7. Các nguồn bên ngoài đã khảo sát

Chi tiết và quyết định nguồn nằm trong [DATA_SOURCE_REVIEW.md](DATA_SOURCE_REVIEW.md). Kết luận ngắn:

1. Hugging Face `trannguyenthaituan/vietnamese_sms_dataset`: cùng nguồn với local, 2,99k dòng, CSV, schema nhị phân, CC BY 4.0. Có thể dùng làm binary baseline/reference, không phải nguồn độc lập để bổ sung four-class.
2. GitHub `tocong282/vietnamese-fraud-detection`: README nói rõ xây từ dataset trên, dùng PhoBERT và có synthetic template augmentation cho minority classes. Đây là derivative/research reference, không được coi là independent test; license dữ liệu/model cần xác minh riêng trước khi dùng.
3. Hugging Face `wikiepeidia/vnphish-dataset`: JSON, 1,66k train, bốn giá trị label nhưng schema là phishing/bank-impersonation oriented, được gắn Synthetic và CC BY-NC-4.0. Không được gộp vào production hoặc benchmark độc lập trước khi kiểm tra license, nguồn, duplicate và mapping nhãn.

## 8. Các điểm cần người dùng xác minh

- Có được phép dùng các nguồn bên ngoài cho mục đích nghiên cứu/thương mại không?
- Bốn nhãn mục tiêu chính thức có đúng là `LEGITIMATE`, `PROMOTION`, `PHISHING`, `FINANCIAL_SCAM` không?
- Có chấp nhận dữ liệu synthetic trong train không? Nếu có, synthetic có được tách thành split riêng không?
- Một SMS vừa yêu cầu OTP vừa yêu cầu chuyển tiền sẽ ưu tiên lớp nào?
- Có nguồn dữ liệu thực tế có provenance và thời gian thu thập rõ ràng không?
- Có thể có hai annotator độc lập và một quy trình adjudication không?

## 9. Việc bắt buộc trước giai đoạn xây dữ liệu

1. Chốt data contract, label policy và quyền sử dụng từng nguồn.
2. Tạo manifest từng dòng: source, source_id, collected_at, synthetic flag, parent_id, annotator, confidence.
3. Chuẩn hóa và deduplicate theo exact normalized text; lập cluster near-duplicate để review, không xóa tự động.
4. Tách nguồn/parent/template family khỏi test; giữ một test set đóng băng, không dùng để chọn tham số.
5. Gán nhãn người thật theo [FOUR_CLASS_LABEL_GUIDELINES.md](FOUR_CLASS_LABEL_GUIDELINES.md), đưa mẫu mơ hồ vào `REVIEW_QUEUE`.
6. Đo inter-annotator agreement và ghi nhận lý do bất đồng.
7. Chỉ sau khi đủ dữ liệu ở cả bốn lớp mới benchmark NB/SVM/PhoBERT đa lớp.

**Phase 0 status: AUDIT COMPLETE. Phase 1 bốn lớp: NOT READY – chờ xác minh nguồn và annotation.**
