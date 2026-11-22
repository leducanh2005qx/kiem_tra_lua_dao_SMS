"""Stateless FastAPI service for the serialized SMS models.

Install ``requirements-api.txt`` before starting this optional service.  The
service never writes SMS content to SQLite and never fetches user URLs.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.config import LABEL_NAMES, MODEL_DIR
from src.modeling import explain_message
from src.multiclass_runtime import load_multiclass_models, predict_multiclass
from src.hierarchical_runtime import load_hierarchical_models, predict_hierarchical
from src.preprocess import validate_message
from src.risk_analyzer import analyze_risk
from src.hybrid_decision import decide_hybrid
from src.url_analyzer import analyze_sms_urls


MAX_BATCH_SIZE = 100
MODEL_LABELS = {
    "nb_baseline": "Naive Bayes – mô hình cơ sở",
    "nb_optimized": "Naive Bayes – tối ưu",
    "svm": "SVM tuyến tính – khuyến nghị",
}


def _load_models():
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        return {}, {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    artifacts = meta.get("model_artifacts", {})
    loaded = {}
    for key in MODEL_LABELS:
        filename = artifacts.get({"nb_baseline": "nb_word_1", "nb_optimized": "nb_optimized", "svm": "svm"}[key])
        if filename and (MODEL_DIR / filename).exists():
            loaded[key] = joblib.load(MODEL_DIR / filename)
    return loaded, meta


def _load_multiclass():
    return load_multiclass_models(MODEL_DIR)


def _load_hierarchical():
    return load_hierarchical_models(MODEL_DIR)


def _score(model, messages):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(messages)[:, 1], "posterior_label_1"
    return model.decision_function(messages), "decision_margin"


try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class PredictRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=5000)
        model_name: str = Field(default="svm", max_length=40)
        include_explanation: bool = True


    class BatchRequest(BaseModel):
        messages: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
        model_name: str = Field(default="svm", max_length=40)
        include_explanation: bool = False


    class MulticlassPredictRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=5000)
        model_name: str = Field(default="svm", max_length=40)


    class HierarchicalPredictRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=5000)
        model_name: str = Field(default="svm", max_length=40)


    app = FastAPI(title="SMS Shield VN API", version="3.0.0", description="Stateless Vietnamese SMS classification API")

    @app.get("/health")
    def health():
        models, _ = _load_models()
        multiclass, multiclass_metadata = _load_multiclass()
        hierarchical, hierarchical_metadata = _load_hierarchical()
        return {"status": "ok", "models_loaded": len(models), "multiclass_status": multiclass_metadata.get("status", "blocked"), "multiclass_models_loaded": len(multiclass), "hierarchical_status": hierarchical_metadata.get("status", "blocked"), "hierarchical_models_loaded": len(hierarchical), "max_batch_size": MAX_BATCH_SIZE}


    @app.get("/models")
    def models_endpoint():
        models, metadata = _load_models()
        return {
            "models": [
                {"name": key, "label": MODEL_LABELS[key], "loaded": key in models, "method": metadata.get("model_configurations", {}).get({"nb_baseline": "nb_word_1", "nb_optimized": "nb_optimized", "svm": "svm"}[key], {}).get("method")}
                for key in MODEL_LABELS
            ]
        }


    @app.get("/multiclass/models")
    def multiclass_models_endpoint():
        models, metadata = _load_multiclass()
        return {
            "status": metadata.get("status", "blocked"),
            "models": sorted(models),
            "label_mapping": metadata.get("label_mapping", {}),
            "reason": metadata.get("reason"),
        }


    @app.post("/multiclass/predict")
    def multiclass_predict(request: MulticlassPredictRequest):
        models, metadata = _load_multiclass()
        if metadata.get("status") != "ready":
            raise HTTPException(status_code=503, detail="Model multiclass chưa được huấn luyện và xác minh đầy đủ.")
        if request.model_name not in models:
            raise HTTPException(status_code=422, detail=f"Model multiclass không khả dụng: {request.model_name}")
        try:
            result = predict_multiclass(models[request.model_name], metadata, request.message, request.model_name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["model_name"] = request.model_name
        result["url_analysis"] = analyze_sms_urls(request.message)
        return result


    @app.get("/hierarchical/models")
    def hierarchical_models_endpoint():
        models, metadata = _load_hierarchical()
        return {
            "status": metadata.get("status", "blocked"),
            "models": sorted(models),
            "layer1_mapping": metadata.get("layer1_mapping", {}),
            "layer2_mapping": metadata.get("layer2_mapping", {}),
        }


    @app.post("/hierarchical/predict")
    def hierarchical_predict(request: HierarchicalPredictRequest):
        models, metadata = _load_hierarchical()
        if metadata.get("status") != "ready":
            raise HTTPException(status_code=503, detail="Mô hình phân cấp chưa được huấn luyện và xác minh đầy đủ.")
        if request.model_name not in models:
            raise HTTPException(status_code=422, detail=f"Mô hình phân cấp không khả dụng: {request.model_name}")
        try:
            validate_message(request.message)
            result = predict_hierarchical(models, metadata, request.message, request.model_name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["model_name"] = request.model_name
        result["url_analysis"] = analyze_sms_urls(request.message)
        return result


    def _predict_one(message, model_name, include_explanation):
        try:
            validate_message(message)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        models, _ = _load_models()
        if model_name not in MODEL_LABELS or model_name not in models:
            raise HTTPException(status_code=422, detail=f"Model không khả dụng: {model_name}")
        model = models[model_name]
        prediction = int(model.predict([message])[0])
        scores, score_type = _score(model, [message])
        risk = analyze_risk(message, prediction, float(scores[0]), score_type)
        hybrid = decide_hybrid(prediction, float(scores[0]), score_type, risk)
        response = {
            "predicted_label": prediction,
            "label_name": LABEL_NAMES[prediction],
            "model_name": model_name,
            "score": float(scores[0]),
            "score_type": score_type,
            "hybrid": hybrid,
            "hybrid_label": hybrid["final_label"],
            "risk_level": hybrid["risk_level"],
            "rule_override": hybrid["rule_override"],
            "explanation": explain_message(model, message, n=10) if include_explanation else [],
            "url_analysis": analyze_sms_urls(message),
        }
        return response


    @app.post("/predict")
    def predict(request: PredictRequest):
        return _predict_one(request.message, request.model_name, request.include_explanation)


    @app.post("/predict/batch")
    def predict_batch(request: BatchRequest):
        models, _ = _load_models()
        if request.model_name not in MODEL_LABELS or request.model_name not in models:
            raise HTTPException(status_code=422, detail=f"Model không khả dụng: {request.model_name}")
        cleaned = []
        for message in request.messages:
            try:
                validate_message(message)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            cleaned.append(message)
        model = models[request.model_name]
        predictions = model.predict(cleaned)
        scores, score_type = _score(model, cleaned)
        return {
            "model_name": request.model_name,
            "count": len(cleaned),
            "items": [
                {
                    "predicted_label": int(prediction),
                    "label_name": LABEL_NAMES[int(prediction)],
                    "score": float(score),
                    "score_type": score_type,
                    "hybrid": decide_hybrid(int(prediction), float(score), score_type, analyze_risk(message, int(prediction), float(score), score_type)),
                    "explanation": explain_message(model, message, n=10) if request.include_explanation else [],
                    "url_analysis": analyze_sms_urls(message),
                }
                for message, prediction, score in zip(cleaned, predictions, scores)
            ],
        }

except ImportError:
    # Keep importing the repository possible without installing API extras.
    app = None
    PredictRequest = None
    BatchRequest = None
