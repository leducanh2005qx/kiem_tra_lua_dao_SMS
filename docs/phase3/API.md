# SMS Shield VN API

API là module tùy chọn, độc lập với Streamlit. Cài dependency:

```powershell
python -m pip install -r requirements-api.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

OpenAPI UI: `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /health`: trạng thái API và số model đã load.
- `GET /models`: danh sách model cùng trạng thái load.
- `POST /predict`: `{ "message": "...", "model_name": "svm", "include_explanation": true }`.
- `POST /predict/batch`: `{ "messages": ["..."], "model_name": "svm" }`, tối đa 100 SMS/request.
- `GET /multiclass/models`: trạng thái artifact 4 lớp và mapping `LEGITIMATE`, `PROMOTION`, `PHISHING`, `FINANCIAL_SCAM`.
- `POST /multiclass/predict`: `{ "message": "...", "model_name": "svm" }`; trả nhãn model, class scores, confidence đã calibration hoặc `NEEDS_REVIEW`.

`score_type=posterior_label_1` là posterior của Naive Bayes. `score_type=decision_margin` là margin của SVM, không phải xác suất.

API không lưu SMS vào SQLite, không log nội dung request, không truy cập URL và chỉ trả về phân tích URL offline.

Các endpoint `/multiclass/*` trả HTTP 503 khi chưa có dataset verified, dataset lock và ba artifact 4 lớp hoàn chỉnh; không tự ánh xạ model nhị phân cũ.
