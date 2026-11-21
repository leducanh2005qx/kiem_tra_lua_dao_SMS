# KỊCH BẢN THUYẾT TRÌNH VÀ DEMO — SMS SHIELD VN

**Thời lượng:** khoảng 10 phút trình bày + 5 phút hỏi đáp (theo hướng dẫn bài tập lớn).

## 12 slide đề xuất

| Slide | Nội dung trên slide | Người trình bày cần nói | Thời gian |
|---|---|---|---:|
| 1 | Tên đề tài và nhóm | Vấn đề, mục tiêu | 30s |
| 2 | Bối cảnh và ví dụ SMS | Tại sao SMS rác/nghi vấn lừa đảo khó phân loại | 45s |
| 3 | Mục tiêu và phạm vi | Nhãn 0/1, không khẳng định gian lận chắc chắn | 40s |
| 4 | Dataset và nguồn | Hugging Face, kích thước, cách gán nhãn theo nguồn | 55s |
| 5 | EDA và tiền xử lý | Biểu đồ lớp, giữ dấu và negation | 55s |
| 6 | Sơ đồ pipeline | Train → validation → test, không leakage | 50s |
| 7 | Hai thuật toán | TF-IDF + NB vs TF-IDF + Linear SVM | 55s |
| 8 | Thiết kế thí nghiệm | Word/char n-gram; đánh giá Macro-F1, Recall | 50s |
| 9 | Bảng kết quả | `test_metrics.csv`, so sánh có số liệu thật | 55s |
| 10 | Phân tích lỗi | Confusion matrix, 5 ví dụ tiêu biểu | 60s |
| 11 | DEMO website | Paste SMS, xem kết quả, xem dashboard | 90s |
| 12 | Kết luận và giới hạn | Tại sao mô hình chưa xác minh an toàn tuyệt đối | 35s |

Điều chỉnh nhịp trình bày theo thực tế; ưu tiên kết quả thực nghiệm và demo hơn slide lý thuyết dài.

## Trình tự demo gợi ý (90 giây)

1. Mở `streamlit run app.py` trước giờ thuyết trình.
2. Ở tab **Kiểm tra tin nhắn**, nhập SMS demo có nội dung đáng ngờ do nhóm tự soạn (không chứa URL độc hại thật), nhấn **Phân tích SMS**.
3. Đọc nhãn dự đoán và nêu lưu ý mô hình có thể dự đoán nhầm. Không khẳng định chắc chắn đầu ra cụ thể trước khi chạy.
4. Nhập SMS xác nhận đơn hàng bình thường, so sánh kết quả nếu dự đoán khác.
5. Chuyển tab **Thống kê & Thực nghiệm** trình chiếu bảng Macro-F1, confusion matrix và trường hợp lỗi thực tế.
6. Nếu còn thời gian, mở tab **Kiểm tra hàng loạt** với CSV minh họa 3–5 tin nhắn được tự soạn, tránh dùng thông tin cá nhân.

## Câu hỏi có khả năng được giảng viên hỏi

- **Vì sao gọi là NLP?** Vì xử lý ngôn ngữ tiếng Việt, biến văn bản thành vector TF-IDF, rồi phân loại bằng học máy.
- **Hai mô hình khác nhau thế nào?** Naive Bayes dựa trên xác suất có điều kiện, SVM tìm siêu phẳng phân chia dữ liệu vector.
- **Tại sao dùng Macro-F1?** Để đánh giá cả hai lớp và không bị độ chính xác chung che khuất một lớp yếu.
- **Vì sao cần validation?** Chọn biểu diễn/siêu tham số mà không xem kết quả test.
- **Vì sao cùng SMS có thể nhận nhầm?** Dataset ít, đặc trưng chung giữa tin thật và tin giả, thiếu ngữ cảnh/số gửi/link thực tế.
- **Làm sao tránh học thuộc?** Tách train/test, chỉ fit vectorizer trên train, kiểm tra trùng lặp; thừa nhận chưa xử lý đầy đủ near duplicate.
- **Vì sao nhãn 1 là rác / nghi vấn lừa đảo chứ không phải lừa đảo?** Vì nhãn gốc của dataset gộp spam và scam.
- **Nếu Accuracy cao thì đủ chưa?** Không; phải xem Precision, Recall, Macro-F1, confusion matrix và false negatives.

## Những ảnh nên chụp để đưa vào slide

- Ảnh tab phân tích một SMS có kết quả dự đoán thật.
- Ảnh biểu đồ phân bố lớp.
- Ảnh bảng test metrics.
- Ảnh confusion matrix của SVM.
- Ảnh một trường hợp sai và giải thích lý do.

**Không tự dựng kết quả hoặc tự ghi accuracy/F1 nếu chưa huấn luyện dữ liệu thật.**
