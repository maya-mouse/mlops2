from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from .schemas import IrisFeatures, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris ML API (Variant 19)")
model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"DEBUG: Looking for model at {MODEL_PATH}") # Це з'явиться в логах GitHub Actions
    if not MODEL_PATH.exists():
        print("DEBUG: Model NOT FOUND")
        return
    model = joblib.load(MODEL_PATH)


@app.get("/")
def root() -> dict:
 return {"status": "ok", "service": "Iris ML API"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    x = np.array([[features.sepal_length, features.sepal_width, 
                   features.petal_length, features.petal_width]])
    
    class_id = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0, class_id])
    
    return PredictionResponse(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        probability=round(proba, 4)
    )