import streamlit as st
import requests
import json
import numpy as np

st.set_page_config(page_title="Gold Price Direction CP3", page_icon="🪙", layout="centered")

st.title("🪙 Gold Price Direction Predictor")
st.markdown("""
This application uses a **Stacking Ensemble** model (Iteration 6 / CP3) to predict whether the price of Gold (GC=F) will go UP, DOWN, or remain FLAT over the next 24 hours.

The model uses 96 features including intermarket ratios (Silver, Oil, DXY, TNX, VIX), momentum acceleration, and cyclic time features.
""")

st.sidebar.header("Backend Connection")
api_url = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000/predict")

st.header("Mock Feature Input")
st.markdown("Since the model requires 96 features, we've provided a button to generate a random plausible feature set for demonstration. In a production environment, this would be fetched live from Yahoo Finance.")

if st.button("Generate Random Features & Predict"):
    # Generate random features just to test the API endpoint
    # In reality we'd pull latest data from yfinance
    features = {}
    import random
    
    # Just to give it some data, the API will fill missing with 0 anyway
    for i in range(96):
        features[f"feature_{i}"] = random.uniform(-1, 1)
        
    # Let's add some of the known feature names to make it look real
    features["gold_ret1d"] = random.uniform(-0.02, 0.02)
    features["vix_level"] = random.uniform(10, 30)
    features["gold_silver_ratio"] = random.uniform(70, 90)
    features["rsi_14"] = random.uniform(30, 70)
    
    st.write("Sending request to FastAPI...")
    
    try:
        response = requests.post(api_url, json={"features": features})
        if response.status_code == 200:
            result = response.json()
            
            st.success("Prediction Received!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction Class", result["prediction_class"])
            col2.metric("Confidence", f"{result['confidence']:.1%}")
            col3.metric("Trade Signal", result["trade_signal"])
            
            st.subheader("Class Probabilities")
            st.json(result["probabilities"])
            
            if "BUY" in result["trade_signal"]:
                st.balloons()
            elif "SELL" in result["trade_signal"]:
                st.snow()
                
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Connection Error. Is the FastAPI server running? (Run `uvicorn app.main:app --reload` first)")

st.markdown("---")
st.markdown("**CP3 Deployment Requirement:** The FastAPI backend serves the ML model, and this Streamlit app acts as the frontend interface.")
