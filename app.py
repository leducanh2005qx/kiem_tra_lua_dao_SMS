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

from src.config import ANNOTATION_LABEL_DB, ANNOTATION_QUEUE, APP_HISTORY_DB, LABEL_NAMES, MODEL_DIR, MULTICLASS_ANNOTATIONS, PROCESSED_DIR, RESULTS_DIR
from src.annotation_store import LABEL_OPTIONS, annotation_csv, annotation_history, annotation_sources, annotation_stats, export_annotations, hierarchical_stats, initialize_annotation_store, list_annotations, quality_summary, save_annotation, save_hierarchical_annotation
from src.annotation_assist import suggest_hierarchical_label
from src.hierarchical import INTENT_OPTIONS, LAYER1_OPTIONS, NON_SCAM_TYPES, SCAM_TYPE_OPTIONS, normalize_intents
from src.hierarchical_runtime import load_hierarchical_models, predict_hierarchical
from src.hierarchical_training import audit_hierarchical_queue
from src.multiclass import multiclass_readiness
from src.modeling import explain_message
from src.multiclass_runtime import load_multiclass_models, predict_multiclass
from src.phobert import dependency_status
from src.preprocess import normalize_message, validate_message
from src.risk_analyzer import analyze_risk
from src.hybrid_decision import FINAL_LABELS, decide_hybrid
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
    "nb_baseline": "Naive Bayes – mô hình cơ sở",
    "nb_optimized": "Naive Bayes – mô hình tối ưu",
    "svm": "SVM tuyến tính – khuyến nghị",
}
MODEL_ORDER = ["nb_baseline", "nb_optimized", "svm"]
MULTICLASS_DISPLAY = {
    "LEGITIMATE": "0 – Tin nhắn hợp lệ",
    "PROMOTION": "1 – Tin nhắn quảng cáo/khuyến mãi",
    "PHISHING": "2 – Giả mạo/đánh cắp thông tin",
    "FINANCIAL_SCAM": "3 – Lừa đảo tài chính",
    "NEEDS_REVIEW": "Cần xem lại – NEEDS_REVIEW",
}
HIERARCHY_DISPLAY = {
    "NOT_SCAM": "Không lừa đảo",
    "SCAM": "Lừa đảo",
    "NEEDS_REVIEW": "Cần xem lại",
    "LEGITIMATE": "Tin nhắn hợp lệ",
    "PROMOTION": "Tin nhắn quảng cáo/khuyến mãi",
    "PHISHING": "Giả mạo/đánh cắp thông tin",
    "FINANCIAL_SCAM": "Lừa đảo tài chính",
    "OTHER_SCAM": "Lừa đảo khác",
}
HYBRID_DISPLAY = {
    "LEGITIMATE": "LEGITIMATE – Tin nhắn hợp lệ",
    "PROMOTION": "PROMOTION – Quảng cáo / Spam",
    "PHISHING": "PHISHING – Giả mạo / Đánh cắp thông tin",
    "FINANCIAL_SCAM": "FINANCIAL_SCAM – Lừa đảo tài chính",
    "NEEDS_REVIEW": "NEEDS_REVIEW – Chưa đủ căn cứ để kết luận",
}
INTENT_DISPLAY = {
    "ASK_OTP": "Yêu cầu OTP",
    "ASK_PASSWORD": "Yêu cầu mật khẩu/thông tin đăng nhập",
    "ASK_TRANSFER": "Yêu cầu chuyển tiền/thanh toán",
    "FAKE_BANK": "Giả mạo ngân hàng",
    "FAKE_FEE": "Dụ đóng phí",
    "MALICIOUS_LINK": "Sử dụng đường dẫn đáng ngờ",
    "FAKE_REWARD": "Giả mạo trúng thưởng",
    "FAKE_RECRUITMENT": "Giả mạo tuyển dụng",
    "INVESTMENT_OFFER": "Dụ đầu tư",
    "IMPERSONATION": "Mạo danh",
    "OTHER": "Ý định khác",
}
STATUS_DISPLAY = {
    "UNMAPPED": "Chưa gắn nhãn",
    "VERIFIED": "Đã xác minh",
    "GOLD_VERIFIED": "Đã xác minh chuẩn",
    "ROBUSTNESS_VERIFIED": "Đã xác minh độ bền",
    "NEEDS_REVIEW": "Cần xem lại",
}
SCAM_TYPE_DISPLAY = {
    "": "Không chọn",
    "phishing": "Lừa đảo lấy thông tin",
    "payment": "Thanh toán giả mạo",
    "fee": "Thu phí giả mạo",
    "investment": "Đầu tư giả mạo",
    "recruitment": "Tuyển dụng giả mạo",
    "prize": "Trúng thưởng giả mạo",
    "impersonation": "Mạo danh",
    "other": "Khác",
    "PHISHING": "Giả mạo/đánh cắp thông tin",
    "FINANCIAL_SCAM": "Lừa đảo tài chính",
    "OTHER_SCAM": "Lừa đảo khác",
}
SCORE_TYPE_DISPLAY = {
    "posterior_label_1": "Xác suất hậu nghiệm của nhãn 1",
    "decision_margin": "Biên quyết định",
    "calibrated_probability": "Xác suất đã hiệu chỉnh",
}
FEEDBACK_TYPE_DISPLAY = {
    "correct": "Kết quả đúng",
    "incorrect": "Kết quả chưa đúng",
    "uncertain": "Chưa chắc chắn",
}
FEEDBACK_STATE_DISPLAY = {
    "pending_review": "Chờ xác minh",
    "verified": "Đã xác minh",
    "rejected": "Từ chối",
    "duplicate": "Mẫu trùng",
}


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


@st.cache_resource
def initialize_annotation_store_once():
    """Initialize and migrate the local annotation database once per app process."""
    return initialize_annotation_store(ANNOTATION_LABEL_DB, ANNOTATION_QUEUE)


def score_prediction(model, messages):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(messages)[:, 1], "posterior_label_1"
    return model.decision_function(messages), "decision_margin"


def model_caption(meta, key):
    artifact_key = {"nb_baseline": "nb_word_1", "nb_optimized": "nb_optimized", "svm": "svm"}.get(key, key)
    config = meta.get("model_configurations", {}).get(artifact_key, {})
    details = []
    config_names = {
        "vectorizer": "bộ tách đặc trưng",
        "alpha": "alpha",
        "fit_prior": "dùng xác suất trước",
        "C": "hệ số C",
        "class_weight": "trọng số lớp",
    }
    for name in ["vectorizer", "alpha", "fit_prior", "C", "class_weight"]:
        if name in config:
            details.append(f"{config_names[name]}={config[name]}")
    return ", ".join(details) or "tệp mô hình đã huấn luyện"


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


def render_prediction(model_key, sms, model, hybrid_result=None):
    prediction = int(model.predict([sms])[0])
    scores, score_type = score_prediction(model, [sms])
    score = float(scores[0])
    if hybrid_result is None:
        risk = analyze_risk(sms, prediction, score, score_type)
        hybrid_result = decide_hybrid(prediction, score, score_type, risk)

    st.subheader("DỰ ĐOÁN CỦA MACHINE LEARNING")
    st.write({"Mô hình": MODEL_LABELS.get(model_key, model_key), "Nhãn nhị phân ML gốc": f"{prediction} – {LABEL_NAMES[prediction]}", "Điểm ML gốc": score, "Loại điểm": SCORE_TYPE_DISPLAY.get(score_type, score_type)})
    if score_type == "posterior_label_1":
        st.caption(f"Xác suất đầu ra của mô hình: {score:.3f} — chưa hiệu chỉnh, không phải xác suất lừa đảo thực tế.")
    else:
        st.caption(f"Biên quyết định của SVM tuyến tính: {score:.3f} — không phải xác suất.")

    final_label = hybrid_result["final_label"]
    st.subheader("KẾT QUẢ HYBRID")
    if final_label == "NEEDS_REVIEW":
        st.warning("NEEDS_REVIEW – Chưa đủ căn cứ để phân loại chính xác. Người dùng nên kiểm tra thêm.")
    elif final_label in {"PHISHING", "FINANCIAL_SCAM"}:
        st.error(f"{HYBRID_DISPLAY[final_label]} – Không nhấp vào đường dẫn hoặc cung cấp thông tin nhạy cảm trước khi kiểm chứng.")
    elif final_label == "PROMOTION":
        st.info(HYBRID_DISPLAY[final_label])
    else:
        st.success(f"{HYBRID_DISPLAY[final_label]} – Không phát hiện dấu hiệu đáng ngờ rõ ràng.")
    st.write({
        "Nhãn cuối cùng": HYBRID_DISPLAY.get(final_label, final_label),
        "Mức cảnh báo": hybrid_result.get("risk_level", "thấp"),
        "Rule có ghi đè ML": "Có" if hybrid_result.get("rule_override") else "Không",
        "Cần người kiểm tra": "Có" if hybrid_result.get("needs_review") else "Không",
    })
    st.write("**Lý do quyết định:**", hybrid_result.get("decision_reason", "Chưa có lý do cụ thể."))
    if hybrid_result.get("reasons"):
        st.write("**Các lý do và dấu hiệu:**")
        for reason in hybrid_result["reasons"]:
            st.markdown(f"- {reason}")

    with st.expander("CHI TIẾT PHÂN TÍCH DẤU HIỆU", expanded=bool(hybrid_result.get("signals"))):
        if hybrid_result.get("signals"):
            signal_rows = []
            for signal in hybrid_result["signals"]:
                signal_rows.append({"Mã rule": signal.get("code"), "Nhóm": HYBRID_DISPLAY.get(signal.get("label"), signal.get("label")), "Mức độ": signal.get("severity"), "Hành động": signal.get("requested_action"), "Giải thích": signal.get("explanation"), "Bằng chứng": " | ".join(signal.get("evidence", []))})
            st.dataframe(pd.DataFrame(signal_rows), hide_index=True, use_container_width=True)
        else:
            st.info("Chưa phát hiện dấu hiệu rủi ro rõ ràng.")
        if hybrid_result.get("evidence"):
            st.write("**Đoạn văn bản làm căn cứ:**")
            for item in hybrid_result["evidence"]:
                st.markdown(f"> {item}")
        if hybrid_result.get("urls"):
            url_rows = []
            for item in hybrid_result["urls"]:
                url_rows.append({"URL gốc": item.get("original_url", item.get("url")), "URL chuẩn hóa": item.get("normalized_url"), "Hostname": item.get("hostname"), "Trạng thái": item.get("status"), "Thương hiệu": ", ".join(item.get("brand_hits", [])), "Dấu hiệu": "; ".join(item.get("reasons", []))})
            st.dataframe(pd.DataFrame(url_rows), hide_index=True, use_container_width=True)
        else:
            st.info("SMS không chứa URL được nhận diện.")

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
    st.caption("Phân tích URL và rule là hậu xử lý ngoại tuyến; không phải bằng chứng nhân quả và không truy cập mạng.")
    return hybrid_result


def page_check_multiclass(models, meta):
    st.header("Kiểm tra SMS – 4 lớp")
    st.caption("Chỉ sử dụng mô hình 4 lớp đã huấn luyện trên dữ liệu đã xác minh và có hiệu chỉnh/ngưỡng từ tập kiểm tra xác nhận.")
    model_key = st.selectbox("Mô hình 4 lớp", sorted(models), format_func=lambda value: MODEL_LABELS.get(value, value), key="multiclass_model")
    sms = st.text_area("Nội dung SMS", height=150, max_chars=5000, key="multiclass_message")
    if st.button("Phân tích 4 lớp", type="primary"):
        try:
            result = predict_multiclass(models[model_key], meta, sms, model_key)
        except ValueError as exc:
            st.error(str(exc))
            return
        label = result["label"]
        if label == "NEEDS_REVIEW":
            st.warning("Kết quả: CẦN XEM LẠI – mô hình chưa đủ tin cậy để kết luận tự động.")
        elif label in {"PHISHING", "FINANCIAL_SCAM"}:
            st.error(f"Kết quả: {MULTICLASS_DISPLAY.get(label, label)}")
        else:
            st.success(f"Kết quả: {MULTICLASS_DISPLAY.get(label, label)}")
        st.write({"Nhãn mô hình dự đoán": MULTICLASS_DISPLAY.get(result["predicted_label"], result["predicted_label"]), "Trạng thái": MULTICLASS_DISPLAY.get(label, label), "Độ tin cậy": result["confidence"], "Loại điểm": "Xác suất đã hiệu chỉnh" if result["score_type"] == "calibrated_probability" else "Xác suất chưa hiệu chỉnh"})
        st.dataframe(pd.DataFrame([{"Lớp": MULTICLASS_DISPLAY.get(key, key), "Điểm": value} for key, value in result["class_scores"].items()]), hide_index=True, use_container_width=True)
        if result["signals"]:
            st.caption("Dấu hiệu quan sát được từ mô hình; không phải bằng chứng nhân quả.")
            signals = pd.DataFrame(result["signals"]).rename(columns={"feature": "Đặc trưng", "contribution": "Mức đóng góp", "toward_label": "Hướng tới nhãn", "method": "Phương pháp"})
            st.dataframe(signals, hide_index=True, use_container_width=True)


def page_check_hierarchical(models, meta):
    st.header("Kiểm tra SMS phân cấp")
    st.caption("Lớp 1 xác định có phải lừa đảo không; lớp 2 chỉ phân loại loại lừa đảo khi lớp 1 là SCAM.")
    model_key = st.selectbox("Mô hình phân cấp", sorted(models), key="hierarchical_model")
    sms = st.text_area("Nội dung SMS", height=150, max_chars=5000, key="hierarchical_message")
    if st.button("Phân tích phân cấp", type="primary"):
        try:
            result = predict_hierarchical(models, meta, sms, model_key)
        except ValueError as exc:
            st.error(str(exc))
            return
        if result["status"] == "NEEDS_REVIEW":
            st.warning("Kết quả: CẦN XEM LẠI – mô hình chưa đủ tin cậy để kết luận tự động.")
        elif result["status"] == "NOT_SCAM":
            st.success("Kết quả lớp 1: Chưa xác định là lừa đảo.")
        else:
            st.error(f"Kết quả lớp 1: LỪA ĐẢO · loại: {HIERARCHY_DISPLAY.get(result['layer2_label'], result['layer2_label'])}")
        output = {
            "Lớp 1": HIERARCHY_DISPLAY.get(result["layer1_label"], result["layer1_label"]),
            "Độ tin cậy lớp 1": result["layer1_confidence"],
            "Loại lừa đảo lớp 2": HIERARCHY_DISPLAY.get(result["layer2_label"], result["layer2_label"] or "Không áp dụng"),
            "Độ tin cậy lớp 2": result["layer2_confidence"] if result["layer2_confidence"] is not None else "Không áp dụng",
            "Trạng thái": HIERARCHY_DISPLAY.get(result["status"], result["status"]),
        }
        st.write(output)


def page_check_sms(models, meta):
    st.header("Kiểm tra SMS")
    st.caption("Phân tích một tin nhắn với quy trình đã huấn luyện; nội dung không được lưu mặc định.")
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
            risk = analyze_risk(sms, prediction, float(score[0]), score_type)
            hybrid = decide_hybrid(prediction, float(score[0]), score_type, risk)
            analysis_id = add_analysis(model_key, prediction, float(score[0]), score_type, sms, save_content=save_content, db_path=APP_HISTORY_DB, hybrid_result=hybrid)
            st.session_state["last_analysis"] = {"id": analysis_id, "model_key": model_key, "message": sms, "prediction": prediction, "score": float(score[0]), "score_type": score_type, "hybrid": hybrid, "save_content": save_content}
            st.session_state.pop("feedback_sent", None)
    last = st.session_state.get("last_analysis")
    if last and last["model_key"] in models:
        st.divider()
        hybrid = render_prediction(last["model_key"], last["message"], models[last["model_key"]], last.get("hybrid"))
        st.info(f"Đã ghi nhận hoạt động #{last['id']}. Nội dung được {'lưu theo đồng ý.' if last.get('save_content') else 'bảo vệ, không lưu.'}")
        if not st.session_state.get("feedback_sent"):
            with st.expander("Phản hồi kết quả chưa chính xác?", expanded=False):
                with st.form("single_feedback_form"):
                    feedback_label = st.radio("Đánh giá", ["Kết quả đúng", "Kết quả chưa đúng", "Chưa chắc chắn"], horizontal=True)
                    suggested = None
                    suggested_code = None
                    if feedback_label != "Kết quả đúng":
                        suggested_code = st.selectbox("Nhãn bạn cho là phù hợp", ["LEGITIMATE", "PROMOTION", "PHISHING", "FINANCIAL_SCAM", "NEEDS_REVIEW"], format_func=lambda value: HYBRID_DISPLAY[value])
                        suggested = {"LEGITIMATE": 0, "PROMOTION": 1, "PHISHING": 2, "FINANCIAL_SCAM": 2, "NEEDS_REVIEW": 2}[suggested_code]
                    comment = st.text_area("Lý do sửa nhãn / ghi chú tùy chọn", max_chars=1000)
                    submitted = st.form_submit_button("Gửi phản hồi")
            if submitted:
                feedback_type = {"Kết quả đúng": "correct", "Kết quả chưa đúng": "incorrect", "Chưa chắc chắn": "uncertain"}[feedback_label]
                add_feedback(last["id"], feedback_type, suggested, comment, APP_HISTORY_DB, suggested_label_code=suggested_code, feedback_reason=comment)
                st.session_state["feedback_sent"] = True
                st.success("Đã lưu phản hồi ở trạng thái chờ kiểm duyệt; phản hồi không tự động thành nhãn huấn luyện.")


def page_batch(models):
    st.header("Kiểm tra hàng loạt")
    st.caption("Tệp CSV chỉ được xử lý trong phiên hiện tại; ứng dụng không tự động lưu tệp tải lên.")
    uploaded = st.file_uploader("Chọn tệp CSV UTF-8 hoặc UTF-8 BOM có cột nội dung tin nhắn", type=["csv"])
    model_key = st.selectbox("Mô hình", [key for key in MODEL_ORDER if key in models], format_func=lambda key: MODEL_LABELS[key], key="batch_model")
    save_history = st.checkbox("Lưu hoạt động vào lịch sử", value=False)
    save_content = st.checkbox("Đồng ý lưu nội dung SMS trong lịch sử", value=False, disabled=not save_history)
    if not uploaded:
        st.info("Chưa có tệp CSV.")
        return
    if uploaded.size > 5 * 1024 * 1024:
        st.error("Tệp vượt quá giới hạn 5 MB.")
        return
    try:
        try:
            incoming = pd.read_csv(uploaded, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded.seek(0)
            incoming = pd.read_csv(uploaded, encoding="utf-8")
    except Exception as exc:
        st.error(f"Không thể đọc tệp CSV: {exc}")
        return
    if "message" not in incoming.columns:
        st.error("Tệp CSV phải có cột nội dung tin nhắn.")
        return
    if len(incoming) > 10000:
        st.error("Mỗi lượt tối đa 10.000 dòng.")
        return
    messages = incoming["message"].fillna("").astype(str)
    if messages.map(lambda value: not bool(normalize_message(value))).any():
        st.error("Tệp CSV có SMS rỗng; hãy bổ sung hoặc loại bỏ các dòng này.")
        return
    if st.button("Phân tích CSV", type="primary"):
        model = models[model_key]
        predictions = model.predict(messages)
        scores, score_type = score_prediction(model, messages)
        output = incoming.copy()
        output["prediction"] = predictions
        output["prediction_label"] = [LABEL_NAMES[int(value)] for value in predictions]
        hybrid_results = [decide_hybrid(int(prediction), float(score), score_type, analyze_risk(message, int(prediction), float(score), score_type)) for message, prediction, score in zip(messages, predictions, scores)]
        output["hybrid_label"] = [result["final_label"] for result in hybrid_results]
        output["risk_level"] = [result["risk_level"] for result in hybrid_results]
        output["rule_override"] = [result["rule_override"] for result in hybrid_results]
        st.session_state["batch_output"] = output
        if save_history:
            for message, prediction, score, hybrid_result in zip(messages, predictions, scores, hybrid_results):
                add_analysis(model_key, int(prediction), float(score), score_type, message, save_content=save_content, db_path=APP_HISTORY_DB, hybrid_result=hybrid_result)
            st.success(f"Đã lưu {len(output)} hoạt động; {'có' if save_content else 'không có'} nội dung SMS.")
    output = st.session_state.get("batch_output")
    if output is not None:
        st.metric("Số SMS bị xếp vào nhóm đáng ngờ theo ML", int((output["prediction"] == 1).sum()))
        display_output = output.copy()
        display_output["hybrid_label"] = display_output["hybrid_label"].map(HYBRID_DISPLAY).fillna(display_output["hybrid_label"])
        display_output = display_output.rename(columns={"message": "Nội dung SMS", "prediction": "Nhãn số ML", "prediction_label": "Nhãn ML", "hybrid_label": "Nhãn Hybrid", "risk_level": "Mức cảnh báo", "rule_override": "Rule ghi đè ML"})
        st.dataframe(display_output.head(100), use_container_width=True, hide_index=True)
        st.download_button("Tải kết quả CSV", output.to_csv(index=False).encode("utf-8-sig"), "sms_predictions.csv", "text/csv")


def page_label_training_data():
    """Fast hierarchical annotation workflow backed by local SQLite."""
    initialize_annotation_store_once()
    st.header("GẮN NHÃN DỮ LIỆU HUẤN LUYỆN")
    st.caption("Trước tiên xác định SMS có phải lừa đảo không. Chỉ SMS xác định là lừa đảo mới cần chọn loại lừa đảo ở lớp 2.")

    stats = hierarchical_stats(ANNOTATION_LABEL_DB)
    cards = [
        ("Tổng SMS", stats["total"]),
        ("Đã xác minh", stats["verified"]),
        ("Chưa xử lý", stats["unlabeled"]),
        ("Cần xem lại", stats["needs_review"]),
        ("Gợi ý máy", stats["machine_suggestions"]),
        ("Đã xác định là lừa đảo", stats["layer1"]["SCAM"]),
    ]
    columns = st.columns(len(cards))
    for index, (label, value) in enumerate(cards):
        columns[index].metric(label, value)
    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.subheader("Phân bố lớp 1")
        layer1_frame = pd.DataFrame({"Nhãn": list(stats["layer1"]), "Số lượng": list(stats["layer1"].values())})
        layer1_frame["Nhãn"] = layer1_frame["Nhãn"].map(HIERARCHY_DISPLAY)
        st.bar_chart(layer1_frame.set_index("Nhãn"))
    with right_chart:
        st.subheader("Phân bố loại lừa đảo lớp 2")
        layer2_frame = pd.DataFrame({"Loại": list(stats["layer2"]), "Số lượng": list(stats["layer2"].values())})
        layer2_frame["Loại"] = layer2_frame["Loại"].map(HIERARCHY_DISPLAY)
        st.bar_chart(layer2_frame.set_index("Loại"))

    quality = quality_summary(ANNOTATION_LABEL_DB)
    if quality["duplicate_groups"] or quality["near_duplicate_rows"] or quality["empty_messages"] or quality["invalid_labels"] or quality["verified_missing_reason"]:
        st.warning(f"Kiểm tra chất lượng: {quality}")
    else:
        st.info(f"Kiểm tra chất lượng: {quality['duplicate_groups']} nhóm trùng; chưa phát hiện lỗi rỗng/nhãn/lý do.")

    filter_labels = {"suspicious_unverified": "Ưu tiên SMS nghi vấn chưa xác minh", "suggested_unverified": "Gợi ý máy chưa xác minh", "unlabeled": "Chưa xử lý", "needs_review": "Cần xem lại", "verified": "Đã xác minh", "all": "Tất cả"}
    left, middle, right = st.columns([1.2, 2, 2])
    with left:
        filter_mode = st.selectbox("Bộ lọc trạng thái", list(filter_labels), format_func=lambda value: filter_labels[value], key="annotation_filter_mode")
    with middle:
        search = st.text_input("Tìm theo ID, nội dung hoặc nguồn", key="annotation_search")
    with right:
        sources = [""] + annotation_sources(ANNOTATION_LABEL_DB)
        source = st.selectbox("Nguồn dữ liệu", sources, format_func=lambda value: value or "Tất cả nguồn", key="annotation_source")

    _, total = list_annotations(ANNOTATION_LABEL_DB, filter_mode, search, source, page=1, page_size=1)
    if not total:
        st.success("Không còn SMS phù hợp với bộ lọc hiện tại.")
        st.download_button("Xuất CSV dữ liệu gắn nhãn", annotation_csv(ANNOTATION_LABEL_DB), "annotation_queue.csv", "text/csv")
        return
    current_position = min(max(1, int(st.session_state.get("annotation_position", 1))), total)
    current_position = int(st.number_input("Vị trí trong hàng đợi", min_value=1, max_value=total, value=current_position, step=1, key="annotation_position_input"))
    st.session_state["annotation_position"] = current_position
    rows, _ = list_annotations(ANNOTATION_LABEL_DB, filter_mode, search, source, page=current_position, page_size=1)
    current = rows[0]

    st.caption(f"SMS {current_position}/{total} · ID: {current['sample_id']}")
    st.text_area("Nội dung SMS", current["message"], height=180, disabled=True, key=f"annotation_message_{current['sample_id']}")
    details = {
        "Nguồn": current.get("source_name") or "Không rõ",
        "Loại nguồn": current.get("source_type") or "Không rõ",
        "Nhãn cũ 0/1": current.get("original_label") or current.get("raw_label") or "Không có",
        "Trạng thái": STATUS_DISPLAY.get(current.get("review_status"), current.get("review_status") or "Chưa xử lý"),
        "Lớp 1 hiện tại": HIERARCHY_DISPLAY.get(current.get("layer1_label"), current.get("layer1_label") or "Chưa xác định"),
        "Loại lừa đảo lớp 2": HIERARCHY_DISPLAY.get(current.get("layer2_label"), current.get("layer2_label") or "Không áp dụng"),
    }
    st.json(details)
    suggested_layer1 = current.get("suggested_layer1") or ""
    suggested_layer2 = current.get("suggested_layer2") or ""
    suggested_non_scam = current.get("suggested_non_scam_type") or ""
    try:
        suggestion_reasons = json.loads(current.get("suggestion_reasons") or "[]")
    except json.JSONDecodeError:
        suggestion_reasons = [current.get("suggestion_reasons") or ""]
    try:
        suggested_intents = normalize_intents(json.loads(current.get("suggested_intents") or "[]"))
    except json.JSONDecodeError:
        suggested_intents = normalize_intents(current.get("suggested_intents"))
    if suggested_layer1:
        suggestion_text = HIERARCHY_DISPLAY[suggested_layer1]
        if suggested_layer2:
            suggestion_text += f" → {HIERARCHY_DISPLAY.get(suggested_layer2, suggested_layer2)}"
        elif suggested_non_scam:
            suggestion_text += f" → {HIERARCHY_DISPLAY.get(suggested_non_scam, suggested_non_scam)}"
        st.info(f"Gợi ý máy: **{suggestion_text}** · điểm sàng lọc {current.get('suggested_confidence') or '0.000'} · chưa phải nhãn xác minh.\n\n" + "; ".join(reason for reason in suggestion_reasons if reason))

    with st.form(f"annotation_form_{current['sample_id']}"):
        decision_options = ["Đúng – xác nhận nhanh", "Sai – sửa lại", "Chưa chắc chắn"]
        default_decision = 0 if suggested_layer1 in {"NOT_SCAM", "SCAM"} else 2
        decision = st.radio("Gợi ý máy có đúng không?", decision_options, index=default_decision, horizontal=True)
        layer1_label = "NEEDS_REVIEW"
        layer2_label = ""
        non_scam_type = ""
        if decision == decision_options[0] and suggested_layer1:
            layer1_label = suggested_layer1
            layer2_label = suggested_layer2
            non_scam_type = suggested_non_scam
            st.write(f"Lưu nhanh: **{HIERARCHY_DISPLAY.get(layer1_label, layer1_label)}**")
        elif decision == decision_options[1]:
            layer1_label = st.selectbox("Lớp 1", ["NOT_SCAM", "SCAM"], format_func=lambda value: HIERARCHY_DISPLAY[value])
            if layer1_label == "SCAM":
                layer2_label = st.selectbox("Loại lừa đảo – lớp 2", SCAM_TYPE_OPTIONS, format_func=lambda value: HIERARCHY_DISPLAY[value])
            else:
                non_scam_type = st.selectbox("Loại tin không lừa đảo", NON_SCAM_TYPES, format_func=lambda value: HIERARCHY_DISPLAY[value])
        intents = st.multiselect("Ý định / dấu hiệu phụ", INTENT_OPTIONS, default=[item for item in suggested_intents if item in INTENT_OPTIONS], format_func=lambda value: INTENT_DISPLAY[value])
        default_reason = current.get("label_reason") or ("Xác nhận gợi ý máy." if decision == decision_options[0] and suggested_layer1 else "")
        reason = st.text_area("Lý do gắn nhãn / lý do cần xem lại", value=default_reason, height=80)
        annotator_id = st.text_input("Mã người gắn nhãn", value=current.get("annotator_id") or "", help="Dùng mã nội bộ, không nhập thông tin cá nhân không cần thiết.")
        confidence_options = ["Chưa chọn", "Thấp", "Vừa", "Cao"]
        confidence = st.select_slider("Mức tin cậy", options=confidence_options, value=current.get("confidence") if current.get("confidence") in confidence_options else "Chưa chọn")
        notes = st.text_area("Ghi chú", value=current.get("notes") or "", height=60)
        submitted = st.form_submit_button("Xác nhận và chuyển SMS tiếp theo", type="primary")
    if submitted:
        if decision == decision_options[2]:
            layer1_label, layer2_label, non_scam_type = "NEEDS_REVIEW", "", ""
        try:
            save_hierarchical_annotation(ANNOTATION_LABEL_DB, ANNOTATION_QUEUE, current["sample_id"], layer1_label, layer2_label, non_scam_type, intents, reason, annotator_id, confidence, notes, expected_revision=current.get("revision"))
            st.session_state["annotation_position"] = current_position if filter_mode in {"unlabeled", "suggested_unverified", "suspicious_unverified"} else min(current_position + 1, total)
            st.success("Đã lưu phân loại phân cấp và cập nhật hàng đợi CSV.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    previous, skip, history = st.columns(3)
    with previous:
        if st.button("Quay lại", disabled=current_position <= 1):
            st.session_state["annotation_position"] = current_position - 1
            st.rerun()
    with skip:
        if st.button("Bỏ qua SMS hiện tại", disabled=current_position >= total):
            st.session_state["annotation_position"] = current_position + 1
            st.rerun()
    with history:
        events = annotation_history(ANNOTATION_LABEL_DB, current["sample_id"])
        if events:
            with st.expander(f"Lịch sử sửa nhãn ({len(events)})"):
                history_table = pd.DataFrame(events)[["created_at", "previous_label", "new_label", "review_status", "annotator_id"]].rename(columns={"created_at": "Thời gian", "previous_label": "Nhãn trước đó", "new_label": "Nhãn mới", "review_status": "Trạng thái", "annotator_id": "Mã người gắn nhãn"})
                history_table["Nhãn trước đó"] = history_table["Nhãn trước đó"].map(HIERARCHY_DISPLAY).fillna(history_table["Nhãn trước đó"])
                history_table["Nhãn mới"] = history_table["Nhãn mới"].map(HIERARCHY_DISPLAY).fillna(history_table["Nhãn mới"])
                history_table["Trạng thái"] = history_table["Trạng thái"].map(STATUS_DISPLAY).fillna(history_table["Trạng thái"])
                st.dataframe(history_table, hide_index=True, use_container_width=True)

    st.download_button("Xuất toàn bộ dữ liệu gắn nhãn đã lưu", annotation_csv(ANNOTATION_LABEL_DB), "annotation_queue_export.csv", "text/csv")


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
    st.caption("Bản ghi không có nội dung chỉ lưu thời gian, mô hình, nhãn, điểm số và độ dài; không thể khôi phục SMS.")
    search = st.text_input("Tìm kiếm nội dung đã được đồng ý lưu, mô hình hoặc mã băm")
    label_text = st.selectbox("Lọc theo nhãn", ["Tất cả", "SMS hợp lệ", "SMS rác/nghi vấn"])
    label = {"Tất cả": None, "SMS hợp lệ": 0, "SMS rác/nghi vấn": 1}[label_text]
    model_text = st.selectbox("Lọc theo mô hình", ["Tất cả", *MODEL_ORDER], format_func=lambda value: "Tất cả" if value == "Tất cả" else MODEL_LABELS.get(value, value))
    model_key = None if model_text == "Tất cả" else model_text
    date_filter = st.checkbox("Lọc theo ngày", value=False)
    date_from = st.date_input("Từ ngày", value=date.today() - timedelta(days=30), disabled=not date_filter)
    date_to = st.date_input("Đến ngày", value=date.today(), disabled=not date_filter)
    filters = {"search": search, "label": label, "model_name": model_key, "date_from": date_from if date_filter else None, "date_to": date_to if date_filter else None}
    page_size_choice = st.selectbox("Số dòng mỗi trang", [10, 20, 50, "Tất cả"])
    total = count_history(**filters, db_path=APP_HISTORY_DB)
    if page_size_choice == "Tất cả":
        page_number = 1
        rows = all_history_rows(filters)
        st.caption(f"Đang hiển thị toàn bộ {len(rows)} bản ghi phù hợp.")
    else:
        page_size = int(page_size_choice)
        pages = max(1, (total + page_size - 1) // page_size)
        page_number = st.number_input("Trang", min_value=1, max_value=pages, value=1)
        rows, _ = list_history(**filters, page=page_number, page_size=page_size, db_path=APP_HISTORY_DB)
    if not rows:
        st.info("Chưa có bản ghi phù hợp.")
    else:
        table = pd.DataFrame(rows)
        table["model_name"] = table["model_name"].map(MODEL_LABELS).fillna(table["model_name"])
        table["result"] = table["predicted_label"].map(LABEL_NAMES)
        table["hybrid_result"] = table.get("hybrid_label", pd.Series(index=table.index, dtype=str)).map(HYBRID_DISPLAY).fillna(table.get("hybrid_label", ""))
        table["content"] = table.apply(privacy_message, axis=1)
        table["score_type"] = table["score_type"].map(SCORE_TYPE_DISPLAY).fillna(table["score_type"])
        history_display = table[["id", "created_at", "content", "model_name", "result", "hybrid_result", "risk_level", "message_length", "score_type"]].rename(columns={"id": "Mã", "created_at": "Thời gian", "content": "Nội dung", "model_name": "Mô hình", "result": "Kết quả ML", "hybrid_result": "Kết quả Hybrid", "risk_level": "Mức cảnh báo", "message_length": "Độ dài", "score_type": "Loại điểm"})
        st.dataframe(history_display, hide_index=True, use_container_width=True)
        selected_id = st.selectbox("Xem chi tiết bản ghi", [row["id"] for row in rows])
        selected = next(row for row in rows if row["id"] == selected_id)
        with st.expander(f"Chi tiết #{selected_id}", expanded=True):
            st.write({"Thời gian": selected["created_at"], "Mô hình": MODEL_LABELS.get(selected["model_name"], selected["model_name"]), "Kết quả ML": LABEL_NAMES[selected["predicted_label"]], "Kết quả Hybrid": HYBRID_DISPLAY.get(selected.get("hybrid_label"), selected.get("hybrid_label") or "Chưa có"), "Mức cảnh báo": selected.get("risk_level") or "Chưa có", "Điểm": selected["prediction_score"], "Loại điểm": SCORE_TYPE_DISPLAY.get(selected["score_type"], selected["score_type"]), "Độ dài": selected["message_length"]})
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
        st.success("Đã xóa toàn bộ lịch sử và phản hồi liên quan.")
        st.rerun()


def page_feedback():
    st.header("Quản lý phản hồi")
    st.caption("Phản hồi chỉ là tín hiệu cần kiểm duyệt, không tự động trở thành nhãn huấn luyện.")
    status_text = st.selectbox("Trạng thái", ["Tất cả", "pending_review", "verified", "rejected", "duplicate"], format_func=lambda value: "Tất cả" if value == "Tất cả" else FEEDBACK_STATE_DISPLAY[value])
    search = st.text_input("Tìm phản hồi", key="feedback_search")
    filters = {"status": None if status_text == "Tất cả" else status_text, "search": search}
    rows, _ = list_feedback(**filters, page=1, page_size=50, db_path=APP_HISTORY_DB)
    if not rows:
        st.info("Chưa có phản hồi phù hợp.")
    else:
        table = pd.DataFrame(rows)
        table["model_name"] = table["model_name"].map(MODEL_LABELS).fillna(table["model_name"])
        table["content"] = table.apply(privacy_message, axis=1)
        table["suggested_label_display"] = table.get("suggested_label_code", pd.Series(index=table.index, dtype=str)).map(HYBRID_DISPLAY).fillna(table.get("suggested_label_code", ""))
        table["review_state_display"] = table.get("review_state", pd.Series(index=table.index, dtype=str)).map(FEEDBACK_STATE_DISPLAY).fillna(table.get("review_state", ""))
        feedback_display = table[["id", "analysis_id", "created_at", "feedback_type", "suggested_label_display", "comment", "review_state_display", "content"]].rename(columns={"id": "Mã", "analysis_id": "Mã phân tích", "created_at": "Thời gian", "feedback_type": "Loại phản hồi", "suggested_label_display": "Nhãn đề xuất", "comment": "Ghi chú", "review_state_display": "Trạng thái", "content": "Nội dung"})
        feedback_display["Loại phản hồi"] = feedback_display["Loại phản hồi"].map(FEEDBACK_TYPE_DISPLAY).fillna(feedback_display["Loại phản hồi"])
        feedback_display["Nhãn đề xuất"] = feedback_display["Nhãn đề xuất"].map(MULTICLASS_DISPLAY).fillna(feedback_display["Nhãn đề xuất"])
        feedback_display["Trạng thái"] = feedback_display["Trạng thái"].map(STATUS_DISPLAY).fillna(feedback_display["Trạng thái"])
        st.dataframe(feedback_display, hide_index=True, use_container_width=True)
        selected_id = st.selectbox("Phản hồi cần xử lý", [row["id"] for row in rows])
        selected = next(row for row in rows if row["id"] == selected_id)
        st.write({"Phân tích": selected["analysis_id"], "Nhãn ML": LABEL_NAMES[selected["predicted_label"]], "Nhãn Hybrid": HYBRID_DISPLAY.get(selected.get("hybrid_label"), selected.get("hybrid_label") or "Chưa có"), "Nội dung": privacy_message(selected), "Ghi chú": selected["comment"] or "(không có)"})
        action = st.selectbox("Cập nhật trạng thái", ["verified", "rejected", "pending_review", "duplicate"], format_func=lambda value: FEEDBACK_STATE_DISPLAY[value])
        if st.button("Lưu trạng thái phản hồi"):
            update_feedback_status(selected_id, action, APP_HISTORY_DB)
            st.success("Đã cập nhật trạng thái.")
            st.rerun()
    export_rows = all_feedback_rows(filters)
    st.download_button("Xuất phản hồi CSV", feedback_csv(export_rows), "prediction_feedback.csv", "text/csv", disabled=not export_rows)


def page_model_stats(results):
    st.header("Thống kê chất lượng mô hình")
    st.caption("Đây là kết quả đánh giá trên tập kiểm tra độc lập, tách biệt với hoạt động sử dụng website.")
    metrics = results["test"]
    if metrics.empty:
        st.info("Chưa có chỉ số đánh giá.")
        return
    display = metrics.copy()
    display["model"] = display["model"].map(MODEL_LABELS).fillna(display["model"])
    columns = ["model", "accuracy", "macro_f1", "precision_macro", "recall_macro", "recall_label_1", "false_negative_label_1"]
    st.dataframe(display[columns].rename(columns={"model": "Mô hình", "accuracy": "Độ chính xác", "macro_f1": "F1 trung bình", "precision_macro": "Độ chính xác trung bình", "recall_macro": "Độ bao phủ trung bình", "recall_label_1": "Độ bao phủ nhãn 1", "false_negative_label_1": "Bỏ sót nhãn 1"}), hide_index=True, use_container_width=True)
    st.subheader("Ma trận nhầm lẫn")
    image_columns = st.columns(len(display))
    for index, model_name in enumerate(metrics["model"]):
        path = RESULTS_DIR / "confusion_matrix" / f"{model_name}.png"
        if path.exists():
            image_columns[index].image(str(path), caption=MODEL_LABELS.get(model_name, model_name), use_container_width=True)
    with st.expander("Bảng lựa chọn tham số trên tập xác nhận"):
        validation_display = results["validation"].rename(
            columns={
                "model": "Mô hình",
                "vectorizer": "Bộ tách đặc trưng",
                "accuracy": "Độ chính xác",
                "macro_f1": "F1 trung bình",
                "precision_macro": "Độ chính xác trung bình",
                "recall_macro": "Độ bao phủ trung bình",
            }
        )
        if "Mô hình" in validation_display:
            validation_display["Mô hình"] = validation_display["Mô hình"].map(MODEL_LABELS).fillna(validation_display["Mô hình"])
        st.dataframe(validation_display, hide_index=True, use_container_width=True)


def page_system_status(models, meta):
    st.header("Đường dẫn (URL) và trạng thái hệ thống")
    st.caption("Các tính năng nghiên cứu nâng cao không tự động thay thế quy trình nhị phân đang hoạt động.")
    st.subheader("Mô hình hiện có")
    st.dataframe(pd.DataFrame([{"Mô hình": MODEL_LABELS[key], "Đã tải": "Có" if key in models else "Không", "Cấu hình": model_caption(meta, key)} for key in MODEL_ORDER]), hide_index=True, use_container_width=True)
    st.subheader("Nghiên cứu phân loại nhiều lớp")
    readiness = multiclass_readiness(MULTICLASS_ANNOTATIONS)
    if readiness["ready_for_training"]:
        st.success("Đã có dữ liệu gắn nhãn đủ điều kiện thử nghiệm nhiều lớp.")
    else:
        st.info("Chưa đủ dữ liệu nhiều lớp đáng tin cậy; tính năng chưa được bật và không có kết quả huấn luyện giả.")
    st.write({"Có thể huấn luyện": "Có" if readiness.get("ready_for_training") else "Không", "Số mẫu huấn luyện": readiness.get("trainable_rows", 0), "Phân bố lớp": readiness.get("class_counts", {}), "Lỗi kiểm tra": readiness.get("errors", [])})
    st.subheader("Mô hình PhoBERT tùy chọn")
    status = dependency_status()
    if all(status[name] for name in ["torch", "transformers", "underthesea"]):
        st.success(f"Đã đủ thư viện PhoBERT; thiết bị: {status['device']}. Chưa có điểm kiểm tra đã tinh chỉnh mặc định.")
    else:
        st.info("PhoBERT chưa được cấu hình trong môi trường chính. Cài các thư viện mô hình khi có dữ liệu và tài nguyên phù hợp.")
    st.write({"Đã có PyTorch": "Có" if status.get("torch") else "Không", "Đã có Transformers": "Có" if status.get("transformers") else "Không", "Đã có Underthesea": "Có" if status.get("underthesea") else "Không", "Thiết bị": status.get("device", "Không rõ")})
    st.subheader("Phân loại 4 lớp")
    if multiclass_metadata.get("status") == "ready":
        st.success("Mô hình 4 lớp đã được xác minh và sẵn sàng kiểm tra.")
        st.write({"Mô hình đã tải": [MODEL_LABELS.get(name, name) for name in sorted(multiclass_models)], "Ánh xạ lớp": {key: MULTICLASS_DISPLAY.get(key, key) for key in multiclass_metadata.get("label_mapping", {})}, "Kích thước các tập": multiclass_metadata.get("split_sizes")})
    else:
        st.warning("Chưa có mô hình 4 lớp đủ điều kiện. Mô hình nhị phân cũ không được dùng như mô hình 4 lớp.")
        st.write({"Trạng thái": "Chưa sẵn sàng", "Lý do": multiclass_metadata.get("reason", "Chưa có tệp mô hình")})
    st.subheader("Phân loại phân cấp lừa đảo")
    hierarchical_ready = audit_hierarchical_queue(ANNOTATION_QUEUE, min_per_class=20)
    if hierarchical_models:
        st.success("Mô hình phân cấp đã được huấn luyện và sẵn sàng kiểm tra.")
    else:
        st.info("Chưa có mô hình phân cấp. Cần xác minh đủ dữ liệu lớp 1 và loại lừa đảo lớp 2.")
    st.write({"Có thể huấn luyện": "Có" if hierarchical_ready.get("ready_for_training") else "Không", "Số mẫu đã xác minh": hierarchical_ready.get("verified_rows", 0), "Phân bố lớp 1": hierarchical_ready.get("layer1_counts", {}), "Phân bố lớp 2": hierarchical_ready.get("layer2_counts", {}), "Lỗi kiểm tra": hierarchical_ready.get("errors", [])})
    st.subheader("Giao diện API tùy chọn")
    st.write("Giao diện API độc lập với Streamlit. Cài thư viện API, sau đó chạy máy chủ tại cổng 8000. Tài liệu nằm trong thư mục tài liệu API.")


def page_about(meta):
    st.header("Giới thiệu & quyền riêng tư")
    st.markdown(
        """
        **SMS Shield VN** phân loại SMS tiếng Việt thành `0 – hợp lệ` và `1 – rác/nghi vấn lừa đảo`.

        - Không mở đường dẫn trong SMS và không gửi nội dung SMS ra ngoài.
        - Lịch sử không lưu nội dung theo mặc định; chỉ lưu khi người dùng đồng ý.
        - Phản hồi luôn ở trạng thái chờ kiểm duyệt và không tự động đi vào tập huấn luyện.
        - Tệp tải lên chỉ tồn tại trong phiên xử lý và không được tự động ghi vào cơ sở dữ liệu cục bộ.
        """
    )
    st.subheader("Sao lưu và xóa cơ sở dữ liệu")
    st.code("Copy-Item data\\app_history.db data\\app_history_backup.db\nRemove-Item data\\app_history.db", language="powershell")
    st.caption(f"Cơ sở dữ liệu hiện tại: {APP_HISTORY_DB}")
    if meta:
        with st.expander("Thông tin kỹ thuật của mô hình"):
            st.json(meta)


initialize_database(APP_HISTORY_DB)
initialize_annotation_store_once()
models, metadata = load_models()
multiclass_models, multiclass_metadata = load_multiclass_models(MODEL_DIR)
hierarchical_models, hierarchical_metadata = load_hierarchical_models(MODEL_DIR)
experiment_results = load_experiment_results()

st.title("SMS Shield VN")
st.caption("Phân tích SMS tiếng Việt theo hướng an toàn, riêng tư và có thể kiểm duyệt")
if not models:
    st.warning("Chưa có mô hình. Hãy chạy `python train.py` trước khi mở ứng dụng.")

with st.sidebar:
    st.header("Điều hướng")
    pages = ["Tổng quan", "Kiểm tra SMS", "Kiểm tra hàng loạt", "GẮN NHÃN DỮ LIỆU HUẤN LUYỆN"]
    if hierarchical_models:
        pages.append("Kiểm tra SMS phân cấp")
    if multiclass_models:
        pages.append("Kiểm tra SMS 4 lớp")
    pages.extend(["Lịch sử", "Phản hồi", "Thống kê mô hình", "URL & hệ thống", "Giới thiệu"])
    page = st.radio("Trang", pages)
    st.divider()
    st.caption("Dữ liệu hoạt động được lưu cục bộ. Nội dung SMS không được lưu nếu chưa đồng ý.")

if page == "Tổng quan":
    page_dashboard()
elif page == "Kiểm tra SMS":
    if models:
        page_check_sms(models, metadata)
    else:
        st.info("Chưa có mô hình để phân tích.")
elif page == "Kiểm tra hàng loạt":
    if models:
        page_batch(models)
    else:
        st.info("Chưa có mô hình để phân tích.")
elif page == "GẮN NHÃN DỮ LIỆU HUẤN LUYỆN":
    page_label_training_data()
elif page == "Kiểm tra SMS phân cấp":
    page_check_hierarchical(hierarchical_models, hierarchical_metadata)
elif page == "Kiểm tra SMS 4 lớp":
    page_check_multiclass(multiclass_models, multiclass_metadata)
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
