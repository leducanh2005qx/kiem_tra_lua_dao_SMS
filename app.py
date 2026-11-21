"""SMS Shield VN Streamlit application for analysis, dashboard and review."""
from datetime import date, timedelta
import json
import unicodedata

import joblib
import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import APP_HISTORY_DB, LABEL_NAMES, MODEL_DIR, MULTICLASS_ANNOTATIONS, PROCESSED_DIR, RESULTS_DIR
from src.multiclass import multiclass_readiness
from src.modeling import explain_message
from src.phobert import dependency_status
from src.preprocess import normalize_message, validate_message
from src.storage import (
    activity_series,
    activity_summary,
    add_analysis,
    add_feedback,
    clear_history,
    count_history,
    delete_analysis,
    feedback_csv,
    history_csv,
    label_counts,
    list_feedback,
    list_history,
    model_counts,
    update_feedback_status,
)
from src.storage import initialize_database
from src.url_analyzer import analyze_sms_urls


st.set_page_config(page_title="SMS Shield VN", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.35rem; max-width: 1280px;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    [data-testid="stMetric"] {border:1px solid #dbe4ee; padding:12px; border-radius:12px; background:#ffffff;}
    .risk-note {padding: 0.75rem 1rem; border-radius: 10px; background:#fff7ed; border-left:4px solid #f59e0b;}
    mark {background:#fde68a; padding:0.08rem 0.2rem; border-radius:0.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_LABELS = {
    "nb_baseline": "Naive Bayes – baseline",
    "nb_optimized": "Naive Bayes – tối ưu",
    "svm": "Linear SVM – khuyến nghị",
}
MODEL_ORDER = ["nb_baseline", "nb_optimized", "svm"]


@st.cache_resource
def load_models():
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        return {}, {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    artifacts = meta.get("model_artifacts", {})
    loaded = {}
    for key, artifact_key in [("nb_baseline", "nb_word_1"), ("nb_optimized", "nb_optimized"), ("svm", "svm")]:
        filename = artifacts.get(artifact_key)
        if filename and (MODEL_DIR / filename).exists():
            loaded[key] = joblib.load(MODEL_DIR / filename)
    return loaded, meta


@st.cache_data
def load_experiment_results():
    result = {}
    for key, path in {
        "test": RESULTS_DIR / "test_metrics.csv",
        "validation": RESULTS_DIR / "validation_metrics.csv",
        "train": PROCESSED_DIR / "clean_train.csv",
    }.items():
        result[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return result


def score_prediction(model, messages):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(messages)[:, 1], "posterior_label_1"
    return model.decision_function(messages), "decision_margin"


def model_caption(meta, key):
    artifact_key = {"nb_baseline": "nb_word_1", "nb_optimized": "nb_optimized", "svm": "svm"}.get(key, key)
    config = meta.get("model_configurations", {}).get(artifact_key, {})
    details = []
    for name in ["vectorizer", "alpha", "fit_prior", "C", "class_weight"]:
        if name in config:
            details.append(f"{name}={config[name]}")
    return ", ".join(details) or "artifact đã huấn luyện"


def highlight_evidence(message, evidence):
    """Highlight model-observed features while preserving Vietnamese accents."""
    text = normalize_message(message)
    folded_chars, source_indexes = [], []
    for source_index, character in enumerate(text):
        for item in unicodedata.normalize("NFD", character):
            if not unicodedata.combining(item):
                folded_chars.append(item.lower())
                source_indexes.append(source_index)
    folded_text = "".join(folded_chars)
    ranges = []
    for feature in sorted({row["feature"] for row in evidence}, key=len, reverse=True):
        folded_feature = "".join(
            item.lower()
            for character in feature
            for item in unicodedata.normalize("NFD", character)
            if not unicodedata.combining(item)
        )
        if not folded_feature.strip():
            continue
        start = folded_text.find(folded_feature)
        while start >= 0:
            end = start + len(folded_feature)
            ranges.append((source_indexes[start], source_indexes[end - 1] + 1))
            start = folded_text.find(folded_feature, start + 1)
    ranges.sort()
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    chunks, cursor = [], 0
    for start, end in merged:
        chunks.extend([text[cursor:start], f"<mark>{text[start:end]}</mark>"])
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def select_period(key):
    today = date.today()
    if key == "7 ngày gần nhất":
        return today - timedelta(days=6), today
    if key == "30 ngày gần nhất":
        return today - timedelta(days=29), today
    if key == "90 ngày gần nhất":
        return today - timedelta(days=89), today
    return None, None


def privacy_message(row):
    if row.get("message_text"):
        return row["message_text"]
    return "Nội dung không được lưu theo lựa chọn riêng tư của người dùng."


def all_history_rows(filters):
    rows, page = [], 1
    while True:
        batch, total = list_history(page=page, page_size=100, **filters)
        rows.extend(batch)
        if len(rows) >= total or not batch:
            return rows
        page += 1


def all_feedback_rows(filters):
    rows, page = [], 1
    while True:
        batch, total = list_feedback(page=page, page_size=100, **filters)
        rows.extend(batch)
        if len(rows) >= total or not batch:
            return rows
        page += 1


def render_prediction(model_key, sms, model):
    prediction = int(model.predict([sms])[0])
    scores, score_type = score_prediction(model, [sms])
    score = float(scores[0])
    if prediction == 1:
        st.error("Kết quả: SMS rác / nghi vấn lừa đảo. Không nhấp link hoặc cung cấp thông tin nhạy cảm trước khi kiểm chứng.")
    else:
        st.success("Kết quả: Chưa phát hiện dấu hiệu theo mô hình. Điều này không đồng nghĩa SMS đã được xác minh an toàn.")
    st.write("**Nhãn dự đoán:**", f"{prediction} – {LABEL_NAMES[prediction]}")
    if score_type == "posterior_label_1":
        st.caption(f"Posterior nhãn 1: {score:.3f} — không phải bằng chứng chắc chắn.")
    else:
        st.caption(f"Decision margin Linear SVM: {score:.3f} — không phải xác suất.")
    evidence = explain_message(model, sms, n=10)
    relevant = [row for row in evidence if row["relevant_to_prediction"]]
    st.write("**Đặc trưng tác động đến dự đoán:**")
    if relevant:
        st.markdown(highlight_evidence(sms, relevant), unsafe_allow_html=True)
        table = pd.DataFrame(relevant)
        table["hướng tới"] = table["toward_label"].map(LABEL_NAMES)
        st.dataframe(
            table[["feature", "contribution", "hướng tới"]].rename(
                columns={"feature": "Đặc trưng", "contribution": "Đóng góp", "hướng tới": "Hướng tới nhãn"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Không có đặc trưng riêng đủ nổi bật để hiển thị.")
    st.caption("Giải thích là đóng góp TF-IDF quan sát được, không phải danh sách từ khóa hay bằng chứng nhân quả.")
    url_report = analyze_sms_urls(sms)
    if url_report["contains_urls"]:
        st.subheader("Phân tích URL độc lập")
        st.caption("Đây là heuristic offline, không phải kết luận của model phân loại SMS và không thực hiện truy cập mạng.")
        level = url_report["risk_level"]
        (st.error if level == "cao" else st.warning if level == "trung bình" else st.info)(
            f"Mức cảnh báo URL: {level} (điểm heuristic {url_report['risk_score']}/100). {url_report['recommendation']}"
        )
        url_rows = []
        for item in url_report["analyses"]:
            url_rows.append({"URL": item["url"], "Hostname": item["hostname"], "Mức": item["risk_level"], "Điểm": item["risk_score"], "Dấu hiệu": "; ".join(item["reasons"]) or "Không nổi bật"})
        st.dataframe(pd.DataFrame(url_rows), hide_index=True, use_container_width=True)


def page_check_sms(models, meta):
    st.header("Kiểm tra SMS")
    st.caption("Phân tích một tin nhắn với pipeline đã huấn luyện; nội dung không được lưu mặc định.")
    examples = {
        "Nhập nội dung của bạn": "",
        "Ví dụ cảnh báo": "Tài khoản của bạn sắp bị khóa. Cung cấp OTP để xác minh ngay!",
        "Ví dụ bình thường": "Đơn hàng của quý khách đã được giao thành công. Cảm ơn quý khách.",
    }
    example = st.selectbox("Ví dụ mẫu", list(examples))
    sms = st.text_area("Nội dung SMS", value=examples[example], height=150, max_chars=5000)
    model_key = st.selectbox("Mô hình", [key for key in MODEL_ORDER if key in models], format_func=lambda key: MODEL_LABELS[key])
    st.caption(f"Cấu hình: {model_caption(meta, model_key)}")
    save_content = st.checkbox("Tôi đồng ý lưu nội dung SMS để xem lại trong lịch sử", value=False)
    if st.button("Phân tích SMS", type="primary"):
        try:
            validate_message(sms)
        except ValueError as exc:
            st.error(str(exc))
        else:
            model = models[model_key]
            prediction = int(model.predict([sms])[0])
            score, score_type = score_prediction(model, [sms])
            analysis_id = add_analysis(model_key, prediction, float(score[0]), score_type, sms, save_content=save_content, db_path=APP_HISTORY_DB)
            st.session_state["last_analysis"] = {"id": analysis_id, "model_key": model_key, "message": sms, "prediction": prediction, "score": float(score[0]), "save_content": save_content}
            st.session_state.pop("feedback_sent", None)
    last = st.session_state.get("last_analysis")
    if last and last["model_key"] in models:
        st.divider()
        render_prediction(last["model_key"], last["message"], models[last["model_key"]])
        st.info(f"Đã ghi nhận hoạt động #{last['id']}. Nội dung được {'lưu theo đồng ý.' if last.get('save_content') else 'bảo vệ, không lưu.'}")
        if not st.session_state.get("feedback_sent"):
            st.subheader("Bạn đánh giá kết quả của AI như thế nào?")
            with st.form("single_feedback_form"):
                feedback_label = st.radio("Đánh giá", ["Kết quả đúng", "Kết quả chưa đúng", "Chưa chắc chắn"], horizontal=True)
                suggested = None
                if feedback_label == "Kết quả chưa đúng":
                    suggested_text = st.selectbox("Nhãn bạn cho là phù hợp", ["SMS hợp lệ", "SMS rác/nghi vấn lừa đảo", "Không đủ thông tin"])
                    suggested = {"SMS hợp lệ": 0, "SMS rác/nghi vấn lừa đảo": 1, "Không đủ thông tin": 2}[suggested_text]
                comment = st.text_area("Ghi chú tùy chọn", max_chars=1000)
                submitted = st.form_submit_button("Gửi phản hồi")
            if submitted:
                feedback_type = {"Kết quả đúng": "correct", "Kết quả chưa đúng": "incorrect", "Chưa chắc chắn": "uncertain"}[feedback_label]
                add_feedback(last["id"], feedback_type, suggested, comment, APP_HISTORY_DB)
                st.session_state["feedback_sent"] = True
                st.success("Đã lưu phản hồi ở trạng thái chờ kiểm duyệt; phản hồi không tự động thành nhãn train.")


def page_batch(models):
    st.header("Kiểm tra hàng loạt")
    st.caption("CSV chỉ được xử lý trong phiên hiện tại; ứng dụng không tự động lưu file upload.")
    uploaded = st.file_uploader("Chọn CSV UTF-8 hoặc UTF-8 BOM có cột `message`", type=["csv"])
    model_key = st.selectbox("Mô hình", [key for key in MODEL_ORDER if key in models], format_func=lambda key: MODEL_LABELS[key], key="batch_model")
    save_history = st.checkbox("Lưu hoạt động vào lịch sử", value=False)
    save_content = st.checkbox("Đồng ý lưu nội dung SMS trong lịch sử", value=False, disabled=not save_history)
    if not uploaded:
        st.info("Chưa có file CSV.")
        return
    if uploaded.size > 5 * 1024 * 1024:
        st.error("File vượt quá giới hạn 5 MB.")
        return
    try:
        try:
            incoming = pd.read_csv(uploaded, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded.seek(0)
            incoming = pd.read_csv(uploaded, encoding="utf-8")
    except Exception as exc:
        st.error(f"Không thể đọc CSV: {exc}")
        return
    if "message" not in incoming.columns:
        st.error("CSV phải có cột `message`.")
        return
    if len(incoming) > 10000:
        st.error("Mỗi lượt tối đa 10.000 dòng.")
        return
    messages = incoming["message"].fillna("").astype(str)
    if messages.map(lambda value: not bool(normalize_message(value))).any():
        st.error("CSV có SMS rỗng; hãy bổ sung hoặc loại bỏ các dòng này.")
        return
    if st.button("Phân tích CSV", type="primary"):
        model = models[model_key]
        predictions = model.predict(messages)
        scores, score_type = score_prediction(model, messages)
        output = incoming.copy()
        output["prediction"] = predictions
        output["prediction_label"] = [LABEL_NAMES[int(value)] for value in predictions]
        st.session_state["batch_output"] = output
        if save_history:
            for message, prediction, score in zip(messages, predictions, scores):
                add_analysis(model_key, int(prediction), float(score), score_type, message, save_content=save_content, db_path=APP_HISTORY_DB)
            st.success(f"Đã lưu {len(output)} hoạt động; {'có' if save_content else 'không có'} nội dung SMS.")
    output = st.session_state.get("batch_output")
    if output is not None:
        st.metric("Số SMS nhãn 1", int((output["prediction"] == 1).sum()))
        st.dataframe(output.head(100), use_container_width=True, hide_index=True)
        st.download_button("Tải kết quả CSV", output.to_csv(index=False).encode("utf-8-sig"), "sms_predictions.csv", "text/csv")


def page_dashboard():
    st.header("Tổng quan hoạt động")
    st.caption("Các chỉ số dưới đây lấy từ lịch sử phân tích thực tế trong SQLite, không có dữ liệu mặc định giả.")
    period = st.selectbox("Khoảng thời gian", ["7 ngày gần nhất", "30 ngày gần nhất", "90 ngày gần nhất", "Tất cả thời gian"])
    date_from, date_to = select_period(period)
    summary = activity_summary(date_from, date_to, APP_HISTORY_DB)
    cards = [("Tổng SMS đã phân tích", summary["total"]), ("SMS hợp lệ", summary["valid"]), ("SMS đáng ngờ", summary["suspicious"]), ("Tỷ lệ đáng ngờ", f"{summary['suspicious_rate']:.1%}"), ("Tổng phản hồi", summary["feedback_total"]), ("Phản hồi chờ xem xét", summary["feedback_pending"])]
    columns = st.columns(3)
    for index, (label, value) in enumerate(cards):
        columns[index % 3].metric(label, value)
    if not summary["total"]:
        st.info("Chưa có dữ liệu hoạt động trong khoảng thời gian này. Hãy phân tích một SMS để bắt đầu.")
        return
    left, right = st.columns(2)
    labels = pd.DataFrame(label_counts(date_from, date_to, APP_HISTORY_DB))
    if not labels.empty:
        labels["label"] = labels["label"].map(LABEL_NAMES)
        with left:
            st.subheader("Phân bố nhãn dự đoán")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(labels["total"], labels=labels["label"], autopct="%1.0f%%", colors=["#22c55e", "#ef4444"])
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
    series = pd.DataFrame(activity_series(date_from, date_to, APP_HISTORY_DB))
    with right:
        st.subheader("Số SMS theo ngày")
        if series.empty:
            st.info("Chưa có dữ liệu theo ngày.")
        else:
            st.line_chart(series.set_index("day")[["total", "valid", "suspicious"]])
    st.subheader("Số SMS theo mô hình")
    models = pd.DataFrame(model_counts(date_from, date_to, APP_HISTORY_DB))
    if not models.empty:
        models["model_name"] = models["model_name"].map(MODEL_LABELS).fillna(models["model_name"])
        st.bar_chart(models.set_index("model_name")["total"])
    st.subheader("Phản hồi người dùng")
    feedback_data = pd.DataFrame({"Trạng thái": ["Chờ xem xét", "Đã xem xét / từ chối"], "Số lượng": [summary["feedback_pending"], max(0, summary["feedback_total"] - summary["feedback_pending"])]})
    st.bar_chart(feedback_data.set_index("Trạng thái"))


def page_history():
    st.header("Lịch sử phân tích")
    st.caption("Bản ghi không có nội dung chỉ lưu thời gian, model, nhãn, điểm số và độ dài; không thể khôi phục SMS.")
    search = st.text_input("Tìm kiếm nội dung đã được đồng ý lưu, model hoặc hash")
    label_text = st.selectbox("Lọc theo nhãn", ["Tất cả", "SMS hợp lệ", "SMS rác/nghi vấn"])
    label = {"Tất cả": None, "SMS hợp lệ": 0, "SMS rác/nghi vấn": 1}[label_text]
    model_text = st.selectbox("Lọc theo mô hình", ["Tất cả", *MODEL_ORDER], format_func=lambda value: "Tất cả" if value == "Tất cả" else MODEL_LABELS.get(value, value))
    model_key = None if model_text == "Tất cả" else model_text
    date_filter = st.checkbox("Lọc theo ngày", value=False)
    date_from = st.date_input("Từ ngày", value=date.today() - timedelta(days=30), disabled=not date_filter)
    date_to = st.date_input("Đến ngày", value=date.today(), disabled=not date_filter)
    filters = {"search": search, "label": label, "model_name": model_key, "date_from": date_from if date_filter else None, "date_to": date_to if date_filter else None}
    page_size = st.selectbox("Số dòng mỗi trang", [10, 20, 50])
    total = count_history(**filters, db_path=APP_HISTORY_DB)
    pages = max(1, (total + page_size - 1) // page_size)
    page_number = st.number_input("Trang", min_value=1, max_value=pages, value=1)
    rows, _ = list_history(**filters, page=page_number, page_size=page_size, db_path=APP_HISTORY_DB)
    if not rows:
        st.info("Chưa có bản ghi phù hợp.")
    else:
        table = pd.DataFrame(rows)
        table["model_name"] = table["model_name"].map(MODEL_LABELS).fillna(table["model_name"])
        table["result"] = table["predicted_label"].map(LABEL_NAMES)
        table["content"] = table.apply(privacy_message, axis=1)
        st.dataframe(table[["id", "created_at", "content", "model_name", "result", "message_length", "score_type"]], hide_index=True, use_container_width=True)
        selected_id = st.selectbox("Xem chi tiết bản ghi", [row["id"] for row in rows])
        selected = next(row for row in rows if row["id"] == selected_id)
        with st.expander(f"Chi tiết #{selected_id}", expanded=True):
            st.write({"Thời gian": selected["created_at"], "Mô hình": MODEL_LABELS.get(selected["model_name"], selected["model_name"]), "Kết quả": LABEL_NAMES[selected["predicted_label"]], "Điểm": selected["prediction_score"], "Loại điểm": selected["score_type"], "Độ dài": selected["message_length"]})
            st.text_area("Nội dung", privacy_message(selected), disabled=True, key=f"history_content_{selected_id}")
            if st.button("Xóa bản ghi này", key=f"delete_{selected_id}"):
                delete_analysis(selected_id, APP_HISTORY_DB)
                st.success("Đã xóa bản ghi.")
                st.rerun()
    export_rows = all_history_rows(filters)
    st.download_button("Xuất lịch sử CSV", history_csv(export_rows), "sms_history.csv", "text/csv", disabled=not export_rows)
    st.divider()
    confirm = st.checkbox("Tôi xác nhận xóa toàn bộ lịch sử phân tích", value=False)
    if st.button("Xóa toàn bộ lịch sử", type="secondary", disabled=not confirm):
        clear_history(APP_HISTORY_DB)
        st.success("Đã xóa toàn bộ lịch sử và feedback liên quan.")
        st.rerun()


def page_feedback():
    st.header("Quản lý phản hồi")
    st.caption("Feedback chỉ là tín hiệu cần kiểm duyệt, không tự động trở thành nhãn huấn luyện.")
    status_text = st.selectbox("Trạng thái", ["Tất cả", "pending", "reviewed", "rejected"], format_func=lambda value: {"Tất cả": "Tất cả", "pending": "Chờ xem xét", "reviewed": "Đã xem xét", "rejected": "Từ chối"}[value])
    search = st.text_input("Tìm feedback", key="feedback_search")
    filters = {"status": None if status_text == "Tất cả" else status_text, "search": search}
    rows, _ = list_feedback(**filters, page=1, page_size=50, db_path=APP_HISTORY_DB)
    if not rows:
        st.info("Chưa có phản hồi phù hợp.")
    else:
        table = pd.DataFrame(rows)
        table["model_name"] = table["model_name"].map(MODEL_LABELS).fillna(table["model_name"])
        table["content"] = table.apply(privacy_message, axis=1)
        st.dataframe(table[["id", "analysis_id", "created_at", "feedback_type", "suggested_label", "comment", "review_status", "content"]], hide_index=True, use_container_width=True)
        selected_id = st.selectbox("Feedback cần xử lý", [row["id"] for row in rows])
        selected = next(row for row in rows if row["id"] == selected_id)
        st.write({"Phân tích": selected["analysis_id"], "Nhãn dự đoán": LABEL_NAMES[selected["predicted_label"]], "Nội dung": privacy_message(selected), "Ghi chú": selected["comment"] or "(không có)"})
        action = st.selectbox("Cập nhật trạng thái", ["reviewed", "rejected", "pending"], format_func=lambda value: {"reviewed": "Đã xem xét", "rejected": "Từ chối", "pending": "Đưa về chờ xem xét"}[value])
        if st.button("Lưu trạng thái feedback"):
            update_feedback_status(selected_id, action, APP_HISTORY_DB)
            st.success("Đã cập nhật trạng thái.")
            st.rerun()
    export_rows = all_feedback_rows(filters)
    st.download_button("Xuất feedback CSV", feedback_csv(export_rows), "prediction_feedback.csv", "text/csv", disabled=not export_rows)


def page_model_stats(results):
    st.header("Thống kê chất lượng mô hình")
    st.caption("Đây là kết quả đánh giá trên tập test độc lập, tách biệt với hoạt động sử dụng website.")
    metrics = results["test"]
    if metrics.empty:
        st.info("Chưa có test metrics.")
        return
    display = metrics.copy()
    display["model"] = display["model"].map(MODEL_LABELS).fillna(display["model"])
    columns = ["model", "accuracy", "macro_f1", "precision_macro", "recall_macro", "recall_label_1", "false_negative_label_1"]
    st.dataframe(display[columns].rename(columns={"model": "Mô hình", "accuracy": "Accuracy", "macro_f1": "Macro-F1", "precision_macro": "Precision macro", "recall_macro": "Recall macro", "recall_label_1": "Recall nhãn 1", "false_negative_label_1": "False Negative nhãn 1"}), hide_index=True, use_container_width=True)
    st.subheader("Confusion matrix")
    image_columns = st.columns(len(display))
    for index, model_name in enumerate(metrics["model"]):
        path = RESULTS_DIR / "confusion_matrix" / f"{model_name}.png"
        if path.exists():
            image_columns[index].image(str(path), caption=MODEL_LABELS.get(model_name, model_name), use_container_width=True)
    with st.expander("Bảng lựa chọn tham số trên validation"):
        st.dataframe(results["validation"], hide_index=True, use_container_width=True)


def page_system_status(models, meta):
    st.header("URL & trạng thái hệ thống")
    st.caption("Các tính năng nghiên cứu nâng cao không tự động thay thế pipeline nhị phân đang hoạt động.")
    st.subheader("Model hiện có")
    st.dataframe(pd.DataFrame([{"model": MODEL_LABELS[key], "loaded": key in models, "configuration": model_caption(meta, key)} for key in MODEL_ORDER]), hide_index=True, use_container_width=True)
    st.subheader("Nghiên cứu đa lớp")
    readiness = multiclass_readiness(MULTICLASS_ANNOTATIONS)
    if readiness["ready_for_training"]:
        st.success("Đã có dữ liệu annotation đủ điều kiện thử nghiệm đa lớp.")
    else:
        st.info("Chưa đủ dữ liệu đa lớp đáng tin cậy; tính năng chưa được bật và không có nút huấn luyện giả.")
    st.json(readiness)
    st.subheader("PhoBERT tùy chọn")
    status = dependency_status()
    if all(status[name] for name in ["torch", "transformers", "underthesea"]):
        st.success(f"Đủ dependency PhoBERT; thiết bị: {status['device']}. Chưa có checkpoint fine-tune mặc định.")
    else:
        st.info("PhoBERT chưa cấu hình trong môi trường chính. Cài requirements-transformers.txt khi có dữ liệu và tài nguyên phù hợp.")
    st.json(status)
    st.subheader("API tùy chọn")
    st.write("API độc lập với Streamlit. Cài `requirements-api.txt`, sau đó chạy `uvicorn api.main:app --host 127.0.0.1 --port 8000`. Tài liệu: `docs/phase3/API.md`.")


def page_about(meta):
    st.header("Giới thiệu & quyền riêng tư")
    st.markdown(
        """
        **SMS Shield VN** phân loại SMS tiếng Việt thành `0 – hợp lệ` và `1 – rác/nghi vấn lừa đảo`.

        - Không mở URL trong SMS và không gửi nội dung SMS ra ngoài.
        - Lịch sử không lưu nội dung theo mặc định; chỉ lưu khi người dùng đồng ý.
        - Feedback luôn ở trạng thái chờ kiểm duyệt và không tự động đi vào tập train.
        - CSV upload chỉ tồn tại trong phiên xử lý và không được tự động ghi vào SQLite.
        """
    )
    st.subheader("Sao lưu và xóa database")
    st.code("Copy-Item data\\app_history.db data\\app_history_backup.db\nRemove-Item data\\app_history.db", language="powershell")
    st.caption(f"Database hiện tại: {APP_HISTORY_DB}")
    if meta:
        with st.expander("Metadata model"):
            st.json(meta)


initialize_database(APP_HISTORY_DB)
models, metadata = load_models()
experiment_results = load_experiment_results()

st.title("SMS Shield VN")
st.caption("Phân tích SMS tiếng Việt theo hướng an toàn, riêng tư và có thể kiểm duyệt")
if not models:
    st.warning("Chưa có model. Hãy chạy `python train.py` trước khi mở website.")

with st.sidebar:
    st.header("Điều hướng")
    page = st.radio("Trang", ["Tổng quan", "Kiểm tra SMS", "Kiểm tra hàng loạt", "Lịch sử", "Phản hồi", "Thống kê mô hình", "URL & hệ thống", "Giới thiệu"])
    st.divider()
    st.caption("Dữ liệu hoạt động lưu cục bộ trong SQLite. Nội dung SMS không được lưu nếu chưa đồng ý.")

if page == "Tổng quan":
    page_dashboard()
elif page == "Kiểm tra SMS":
    if models:
        page_check_sms(models, metadata)
    else:
        st.info("Chưa có model để phân tích.")
elif page == "Kiểm tra hàng loạt":
    if models:
        page_batch(models)
    else:
        st.info("Chưa có model để phân tích.")
elif page == "Lịch sử":
    page_history()
elif page == "Phản hồi":
    page_feedback()
elif page == "Thống kê mô hình":
    page_model_stats(experiment_results)
elif page == "URL & hệ thống":
    page_system_status(models, metadata)
else:
    page_about(metadata)
