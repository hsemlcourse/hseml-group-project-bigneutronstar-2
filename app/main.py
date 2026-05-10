from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to load the model
MODEL_PATH = PROJECT_ROOT / "models" / "cp3_stacking_model.pkl"
model_artifact = None
try:
    with open(MODEL_PATH, "rb") as f:
        model_artifact = pickle.load(f)
except FileNotFoundError:
    print("Warning: cp3_stacking_model.pkl not found. API won't predict.")

app = FastAPI(title="Gold Price Direction API")

class PredictionRequest(BaseModel):
    # Instead of passing 96 features manually, we'll accept a dictionary of features
    # In a real scenario, this would likely take raw OHLCV and compute internally
    features: dict

class PredictionResponse(BaseModel):
    prediction_class: int
    probabilities: dict
    confidence: float
    trade_signal: str

@app.get("/")
def read_root():
    return {"message": "Gold Price Direction API is running. Use /predict for inference."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model_artifact is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    opt_thresh = model_artifact.get("optimal_p_thresh", 0.59)
    
    try:
        # Convert dictionary to DataFrame using correct feature ordering
        df = pd.DataFrame([request.features])
        # Ensure all columns exist, fill missing with 0 for safety
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        X = df[feature_cols].values
        
        probas = model.predict_proba(X)[0]
        # probas is [prob_0, prob_1, prob_2]
        pred_class = int(np.argmax(probas))
        confidence = float(probas[pred_class])
        
        signal = "HOLD"
        if pred_class == 0 and confidence >= opt_thresh:
            signal = "SELL (Down)"
        elif pred_class == 2 and confidence >= opt_thresh:
            signal = "BUY (Up)"
            
        return {
            "prediction_class": pred_class,
            "probabilities": {
                "0 (Down)": float(probas[0]),
                "1 (Flat)": float(probas[1]),
                "2 (Up)": float(probas[2])
            },
            "confidence": confidence,
            "trade_signal": signal
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
