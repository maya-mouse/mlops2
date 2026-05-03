import logging
import time
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .schemas import IrisFeatures, PredictionResponse, DriftRequest, DriftResponse
from .metrics import (
    REGISTRY, PREDICTION_COUNTER, PREDICTION_LATENCY, 
    PREDICTION_CONFIDENCE, ERROR_COUNTER, MODEL_LOADED, 
    DRIFT_CHECKS, DRIFT_DETECTED
)
from .drift import DriftDetector
from .logging_config import setup_logging

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
REFERENCE_PATH = Path(__file__).resolve().parent.parent / "reference_stats.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


setup_logging()
logger = logging.getLogger("ml-api")

app = FastAPI(
    title="Iris ML API (Variant 19) with Monitoring",
    description="ML API з Prometheus-метриками та Drift Detection",
    version="2.0.0"
)

model = None
drift_detector: DriftDetector | None = None

@app.on_event("startup")
def startup_event() -> None:
    global model, drift_detector
    
    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        logger.error("Model file not found", extra={"path": str(MODEL_PATH)})
        return
        
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)
    
    if REFERENCE_PATH.exists():
        ref_data = joblib.load(REFERENCE_PATH)
        drift_detector = DriftDetector(
            reference=ref_data["X"],
            feature_names=ref_data["feature_names"]
        )
        logger.info("Startup complete", extra={"event": "startup", "drift_ready": True})
    else:
        logger.warning("Reference stats missing", extra={"event": "startup", "drift_ready": False})

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start_time
    
    if request.url.path == "/predict":
        PREDICTION_LATENCY.observe(latency)
        
    return response

@app.get("/")
def root():
    return {"status": "ok", "variant": 19, "service": "Iris ML API"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "drift_detector_ready": drift_detector is not None
    }

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    if model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    x = np.array([[features.sepal_length, features.sepal_width, 
                   features.petal_length, features.petal_width]])
    
    try:
        class_id = int(model.predict(x)[0])
        proba = float(model.predict_proba(x)[0, class_id])
        class_name = CLASS_NAMES[class_id]
        
        PREDICTION_COUNTER.labels(class_name=class_name, status="success").inc()
        PREDICTION_CONFIDENCE.observe(proba)
        
        logger.info("Prediction made", extra={
            "event": "prediction",
            "class": class_name,
            "confidence": round(proba, 4)
        })
        
        return PredictionResponse(
            class_id=class_id,
            class_name=class_name,
            probability=round(proba, 4)
        )
    except Exception as e:
        ERROR_COUNTER.labels(error_type="inference_failed").inc()
        logger.error("Inference error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Prediction error")

@app.post("/check-drift", response_model=DriftResponse)
def check_drift(payload: DriftRequest):
    if drift_detector is None:
        ERROR_COUNTER.labels(error_type="drift_not_ready").inc()
        raise HTTPException(status_code=503, detail="Drift detector not ready")
    
    DRIFT_CHECKS.inc()
    current_data = np.array(payload.samples)
    result = drift_detector.detect(current_data, alpha=payload.alpha)
    
    for feat, info in result["per_feature"].items():
        if info["drift_detected"]:
            DRIFT_DETECTED.labels(feature=feat).inc()
            
    logger.info("Drift check completed", extra={
        "event": "drift_check",
        "detected": result["drift_detected"],
        "drifted_count": result["n_drifted_features"]
    })
    
    return DriftResponse(**result)