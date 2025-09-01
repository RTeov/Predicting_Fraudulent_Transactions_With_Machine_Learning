from fastapi import APIRouter, HTTPException
from app.models.schemas import TransactionInput
from app.models.predictor import get_model, predict_transaction

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Fraud Detection API is running."}

@router.post("/predict")
def predict(input: TransactionInput):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return predict_transaction(model, input)
