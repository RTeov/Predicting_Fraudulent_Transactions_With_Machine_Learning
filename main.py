from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# Example: Load a trained model (update the path/model name as needed)
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class TransactionInput(BaseModel):
    Transaction_Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Transaction_Amount: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running."}

@app.post("/predict")
def predict(input: TransactionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    features = np.array([[getattr(input, f) for f in input.__fields__]])
    pred = model.predict(features)[0]
    prob = float(model.predict_proba(features)[0][1]) if hasattr(model, "predict_proba") else None
    return {"fraud_flag": int(pred), "fraud_probability": prob}
