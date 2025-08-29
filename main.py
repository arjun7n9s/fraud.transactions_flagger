import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API", version="1.0")

class Transaction(BaseModel):
    step: float
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_CASH_IN: bool
    type_CASH_OUT: bool
    type_DEBIT: bool
    type_PAYMENT: bool
    type_TRANSFER: bool

model = joblib.load('XGBoost.joblib')
print("Model loaded successfully!")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fraud Detection API is running."}

@app.post("/predict/")
def predict_fraud(transaction: Transaction):
    data = transaction.dict()
    input_df = pd.DataFrame([data])
    fraud_probability = model.predict_proba(input_df)[0][1]
    threshold = 0.3
    return {
        "fraud_probability": float(fraud_probability),
        "is_flagged": bool(fraud_probability > threshold)
    }
