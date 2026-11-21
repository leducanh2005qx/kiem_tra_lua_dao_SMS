# ĐỀ CƯƠNG BÁO CÁO BTL NLP — SMS Shield VN

**Tên đề tài:** Xây dựng hệ thống phát hiện tin nhắn SMS rác và nghi vấn lừa đảo tiếng Việt sử dụng xử lý ngôn ngữ tự nhiên và học máy.

Báo cáo PDF đề xuất **20–30 trang nội dung chính**, chưa tính phụ lục; theo tài liệu hướng dẫn của giảng viên. File nộp gợi ý `NLP_BTL_MaNhom.pdf`, cùng repository/Colab, README, thư mục kết quả và demo. Không điền dữ liệu thực nghiệm trước khi chạy code.

## Bố cục đề xuất và số trang

### Bìa + mục lục (không tính trang chính)
- Tên trường, khoa, môn, đề tài, giảng viên, nhóm, sinh viên, năm học.
- Danh mục bảng, hình, từ viết tắt nếu có.

### 1. Giới thiệu (2–3 trang)
1.1 Bối cảnh SMS rác, SMS giả mạo; vì sao phân loại văn bản tiếng Việt đáng nghiên cứu.
1.2 Bài toán: text input → `label` nhị phân. Đầu ra gồm nhãn và minh họa các đặc trưng tác động. Không đưa ra kết luận pháp lý.
1.3 Mục tiêu: hai pipeline, đánh giá công bằng; demo tra cứu; có thể tái lập.
1.4 Câu hỏi nghiên cứu: SVM tốt hơn NB không? n-gram ảnh hưởng thế nào? mẫu không dấu và OTP hợp lệ gây lỗi gì?
1.5 Phạm vi và giới hạn nghiên cứu.

### 2. Dữ liệu và tiền xử lý (4–5 trang)
2.1 Nguồn dữ liệu, giấy phép, cách tải; trích dẫn dataset gốc.
2.2 Cấu trúc CSV: `message`, `label`. Bảng `0` hợp lệ / `1` spam-scam.
2.3 Quy mô gốc 2.394 train + 597 test (theo nguồn), quy mô sau lọc theo `quality_report.json`.
2.4 Khảo sát EDA: số mẫu/lớp, độ dài ký tự, bản ghi trống, trùng chính xác, mẫu minh họa ẩn PII. Có thể bổ sung histogram từ `processed/`.
2.5 Tiền xử lý: Unicode NFC → lowercase → chuẩn hóa khoảng trắng; nêu vì sao không xóa dấu/URL/`không`.
2.6 Nguy cơ data leakage, near duplicate; nêu giới hạn.

### 3. Phương pháp và kiến trúc (4–5 trang)
3.1 Luồng dữ liệu: CSV → Clean → TF-IDF → NB/SVM → chọn validation → test → Streamlit.
3.2 TF-IDF: khái niệm tần suất từ và độ hiếm toàn corpus; `TfidfVectorizer` fit trên train.
3.3 Baseline: MultinomialNB alpha=1.0, word unigrams.
3.4 Đối chứng: LinearSVC C=1.0, kiểm tra 3 biểu diễn word 1, word 1–2, char 3–5.
3.5 Kiến trúc file: `download_data.py`, `train.py`, `src/*`, `app.py`, `results/*`.
3.6 Website demo: nhập SMS, chọn mô hình, dự đoán hàng loạt, thống kê, phân tích đặc trưng.

### 4. Thiết kế thực nghiệm (4–5 trang)
4.1 Cấu hình máy, phiên bản Python, scikit-learn, pandas. Ghi phiên bản thực tế, không tự điền.
4.2 Tập train của nguồn chia train-fit 80% và validation 20%, stratified seed=42; test độc lập.
4.3 Thực nghiệm 1: NB word 1-gram, baseline.
4.4 Thực nghiệm 2: SVM word unigram vs bigram vs char ngram (khảo sát có kiểm soát).
4.5 Tiêu chí chọn biến thể SVM: Macro-F1 trên validation; sau đó refit train rồi đánh giá một lần trên test.
4.6 Công thức: Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R); Macro-F1 là trung bình F1 từng lớp; Accuracy = số dự đoán đúng/tổng mẫu.
4.7 Kiểm soát rò rỉ, sai lệch do dữ liệu nhỏ và nhãn gộp.

### 5. Kết quả và thảo luận (5–7 trang)
5.1 Bảng validation: trích `validation_metrics.csv` (số liệu thực tế).
5.2 Bảng test: trích `test_metrics.csv` (bắt buộc), làm tròn 3 chữ số nhưng giữ CSV gốc.
5.3 Biểu đồ so sánh Macro-F1 và confusion matrix từ `results/figures/`.
5.4 Phân tích 5–10 lỗi đại diện từ `error_analysis.csv`:
    - SMS OTP thật bị nhận nhầm?
    - SMS lừa đảo dùng ngôn ngữ trung tính?
    - Tin nhắn không dấu/trộn ngôn ngữ?
    - Mẫu gần trùng template?
    - Từ khóa như “trúng thưởng” xuất hiện trong cảnh báo chống lừa đảo?
5.5 Screenshots 4 tab website, chú thích dữ liệu minh họa khác dữ liệu test.
5.6 Thảo luận ưu, nhược; sai lệch mẫu, độ tin cậy và ngữ cảnh sử dụng.

### 6. Kết luận và hướng phát triển (1–2 trang)
- Nêu kết quả có bằng chứng (không chốt SVM thắng nếu số liệu ngược lại).
- Hướng cải tiến: thu thập nhãn 3 lớp (hợp lệ/spam/scam), bổ sung corpus không dấu, đánh giá near-duplicate, thử PhoBERT và calibration, kiểm tra link qua danh tiếng tên miền (ngoài phạm vi NLP thuần).

### 7. Phân công, tài liệu tham khảo và phụ lục (2–3 trang)
- Thành viên, task, commit GitHub và minh chứng đóng góp.
- Trích nguồn dataset, scikit-learn, Streamlit, và các tài liệu học thuật đã thật sự đọc.
- Phụ lục: các lệnh cài, phiên bản thư viện, hash dataset, giới hạn sử dụng AI (nếu có).

## Rubric cần đối chiếu (100 điểm)

| Thành phần | Tối đa | Minh chứng |
|---|---:|---|
| Xác định bài toán & dữ liệu | 15 | Ch.1–2 + EDA + nguồn |
| Thiết kế và triển khai mô hình | 25 | Ch.3 + code + 2 pipeline |
| Thực nghiệm & chỉ số | 25 | Ch.4–5, cùng tập test, có validation và ablation |
| Phân tích và thảo luận | 15 | Ch.5, 5–10 trường hợp lỗi |
| Báo cáo & tính tái lập | 20 | PDF, README, code, kết quả, hướng dẫn chạy |

**Bảng bắt buộc:** so sánh baseline vs đối chứng, có chỉ số. **Bắt buộc phân tích lỗi:** không chỉ nói mô hình tốt. Mọi số liệu phải do thực nghiệm của nhóm sinh ra.
