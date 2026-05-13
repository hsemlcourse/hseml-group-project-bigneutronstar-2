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

MODEL_PATH = PROJECT_ROOT / "models" / "cp3_stacking_model.pkl"
model_artifact = None
try:
    with open(MODEL_PATH, "rb") as f:
        model_artifact = pickle.load(f)
except FileNotFoundError:
    pass

app = FastAPI(title="Gold Price Direction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Gold Price Direction API is running. Use /api/latest for real data."}

@app.get("/api/latest")
def get_latest_data():
    if model_artifact is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        df, _ = build_full_df(horizon=24, threshold=0.002, use_external=True)
        
        recent_data = df.tail(720).copy()
        
        model = model_artifact["model"]
        feature_cols = model_artifact["feature_cols"]
        opt_thresh = model_artifact.get("optimal_p_thresh", 0.59)
        
        for col in feature_cols:
            if col not in recent_data.columns:
                recent_data[col] = 0.0
                
        X_recent = recent_data[feature_cols].values
        
        probas_all = model.predict_proba(X_recent)
        
        chart_data = []
        for i, (_, row) in enumerate(recent_data.iterrows()):
            probas = probas_all[i]
            pred_class = int(np.argmax(probas))
            confidence = float(probas[pred_class])
            
            signal = "HOLD"
            if pred_class == 0 and confidence >= opt_thresh:
                signal = "SELL (Down)"
            elif pred_class == 2 and confidence >= opt_thresh:
                signal = "BUY (Up)"
                
            chart_data.append({
                "time": str(row["DateTime"]),
                "price": float(row["Close"]),
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
            })
            
        last_item = chart_data[-1]
            
        return {
            "ticker": "Gold (GC=F)",
            "last_price": last_item["price"],
            "timestamp": last_item["time"],
            "chart_data": chart_data,
            "prediction": last_item["prediction"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
