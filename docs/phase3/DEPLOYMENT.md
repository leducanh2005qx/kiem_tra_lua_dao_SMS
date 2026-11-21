# Hướng dẫn triển khai Phase 3

## Local

```powershell
python -m pytest -q
streamlit run app.py
```

API tùy chọn cần `requirements-api.txt`. PhoBERT cần `requirements-transformers.txt`, GPU là tùy chọn; CPU có thể chậm và cần RAM lớn hơn pipeline TF-IDF.

## Docker

```powershell
docker compose build
docker compose up streamlit api
```

Streamlit chạy ở `http://localhost:8501`, API ở `http://localhost:8000/docs`. Volume `./data:/app/data` giữ SQLite cục bộ. Không đưa `data/app_history.db`, secrets hoặc file upload vào image.

## Streamlit Community Cloud / Render

- Streamlit Community Cloud: chỉ triển khai giao diện từ repository đã được cấp quyền; không commit SQLite cá nhân. SQLite trên môi trường ephemeral có thể mất khi app restart, nên cần cơ chế backup riêng nếu dùng thật.
- Render/Fly.io tương thích: triển khai API bằng `uvicorn api.main:app --host 0.0.0.0 --port $PORT`; dùng persistent disk nếu cần giữ SQLite.
- Bật HTTPS, giới hạn request ở reverse proxy, đặt timeout và theo dõi lỗi HTTP 4xx/5xx. API hiện chưa có authentication/rate-limit nên không mở public trực tiếp khi chưa bổ sung lớp bảo vệ.

## PhoBERT

PhoBERT là tính năng nghiên cứu tùy chọn, không thay thế NB/SVM. Model card `vinai/phobert-base-v2` nêu input phải được word-segmented và license AGPL-3.0; cần kiểm tra tương thích pháp lý trước khi đóng gói thương mại. Không tải weights trong Docker mặc định và không báo cáo benchmark khi chưa fine-tune trên dữ liệu được gán nhãn đáng tin cậy.
