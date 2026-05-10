from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_full_df

# Attempt to load the model
MODEL_PATH = PROJECT_ROOT / "models" / "cp3_stacking_model.pkl"
model_artifact = None
try:
    with open(MODEL_PATH, "rb") as f:
        model_artifact = pickle.load(f)
except FileNotFoundError:
    print("Warning: cp3_stacking_model.pkl not found. API won't predict.")

app = FastAPI(title="Gold Price Direction API")

# Add CORS middleware to allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    features: dict

class PredictionResponse(BaseModel):
    prediction_class: int
    probabilities: dict
    confidence: float
    trade_signal: str

@app.get("/")
def read_root():
    return {"message": "Gold Price Direction API is running. Use /api/latest for real data."}

@app.get("/api/latest")
def get_latest_data():
    """
    Fetches the latest real data from our dataset, builds features, 
    runs the prediction, and returns the chart data and result.
    """
    if model_artifact is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Build the full dataframe with all features (uses local raw data + merges)
        df, _ = build_full_df(horizon=24, threshold=0.002, use_external=True)
        
        # Get the last 50 hours for the chart
        recent_data = df.tail(50).copy()
        
        # Prepare chart data (DateTime, Close price)
        chart_data = []
        for _, row in recent_data.iterrows():
            chart_data.append({
                "time": str(row["DateTime"]),
                "price": float(row["Close"])
            })
            
        # Get the very last row for prediction
        last_row = df.iloc[-1:]
        
        # Extract features for the model
        model = model_artifact["model"]
        feature_cols = model_artifact["feature_cols"]
        opt_thresh = model_artifact.get("optimal_p_thresh", 0.59)
        
        # Ensure all columns exist
        for col in feature_cols:
            if col not in last_row.columns:
                last_row[col] = 0.0
                
        X = last_row[feature_cols].values
        
        # Run prediction
        probas = model.predict_proba(X)[0]
        pred_class = int(np.argmax(probas))
        confidence = float(probas[pred_class])
        
        signal = "HOLD"
        if pred_class == 0 and confidence >= opt_thresh:
            signal = "SELL (Down)"
        elif pred_class == 2 and confidence >= opt_thresh:
            signal = "BUY (Up)"
            
        return {
            "ticker": "Gold (GC=F)",
            "last_price": float(last_row["Close"].iloc[0]),
            "timestamp": str(last_row["DateTime"].iloc[0]),
            "chart_data": chart_data,
            "prediction": {
                "prediction_class": pred_class,
                "probabilities": {
                    "Down": float(probas[0]),
                    "Flat": float(probas[1]),
                    "Up": float(probas[2])
                },
                "confidence": confidence,
                "trade_signal": signal,
                "threshold": opt_thresh
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
