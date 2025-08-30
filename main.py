import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import shap
import time
import requests
import json

# --- 1. Initialize App and Load Models ---
app = FastAPI(title="Explainable AI Fraud Detection System", version="4.0")

# Load the core XGBoost model
model = joblib.load('XGBoost.joblib')
print("XGBoost model loaded successfully!")

# Load the data sample and create the SHAP explainer
X_train_sample = pd.read_csv('X_train_sample.csv')
for col in X_train_sample.columns:
    if X_train_sample[col].dtype == 'bool' or X_train_sample[col].dtype == 'object':
        X_train_sample[col] = X_train_sample[col].astype(int)
explainer = shap.TreeExplainer(model, X_train_sample)
print("SHAP Explainer initialized successfully!")

user_history = {} # In-memory store for velocity

# --- 2. Pydantic Model (Input Structure) ---
class Transaction(BaseModel):
    step: float
    nameOrig: str
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

# --- 3. Gemini Helper Function (The Translator) ---
def get_gemini_explanation(json_data: dict) -> str:
    prompt = f"""
    You are a senior fraud analyst AI. Your task is to interpret the JSON output from our XGBoost fraud model and write a concise, clear summary for a non-technical manager.

    Here is the model's technical data:
    {json.dumps(json_data, indent=2)}

    Based on this, please provide a summary that includes:
    1. A clear verdict (e.g., "High risk of fraud detected.").
    2. The top 2-3 reasons, explained in simple business terms. The 'explanation' section contains the key technical drivers from SHAP.
    3. A concluding sentence on the model's confidence.

    Keep your entire response to under 70 words.
    """
    api_key = "*********"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        return "Could not generate an AI explanation at this time."

# --- 4. The Main Prediction Endpoint ---
@app.post("/predict_and_explain/")
def predict_and_explain(transaction: Transaction):
    # Prepare data
    data_dict = transaction.dict()
    user_id = data_dict.pop('nameOrig')
    input_df = pd.DataFrame([data_dict])
    for col in input_df.columns:
        if input_df[col].dtype == 'bool': input_df[col] = input_df[col].astype(int)

    # Prediction
    fraud_probability = model.predict_proba(input_df)[0][1]
    threshold = 0.30
    is_flagged = bool(fraud_probability > threshold)

    response = {
        "fraud_probability": float(fraud_probability),
        "is_flagged": is_flagged
    }

    # Explanation and Gemini Summary
    if is_flagged:
        shap_values = explainer.shap_values(input_df)
        feature_importance = pd.Series(np.abs(shap_values[0]), index=input_df.columns)
        top_reasons = feature_importance.nlargest(3).to_dict()
        response["explanation"] = {f"reason_{i+1}": {"feature": name, "impact_score": score} for i, (name, score) in enumerate(top_reasons.items())}
        response["natural_language_explanation"] = get_gemini_explanation(response)
    else:
        response["natural_language_explanation"] = "This transaction is low-risk. Both AI and rule-based checks clear it for processing."

    # Add context (remains the same)
    current_time = time.time()
    timestamps = user_history.get(user_id, [])
    tx_velocity_5m = sum(1 for ts in timestamps if current_time - ts < 300)
    timestamps.append(current_time)
    user_history[user_id] = timestamps
    response["context"] = {"user_id": user_id, "transactions_in_last_5_mins": tx_velocity_5m}

    return response






# import joblib
# import pandas as pd
# import numpy as np
# from fastapi import FastAPI
# from pydantic import BaseModel
# import shap  # <-- UPGRADE: Import the SHAP library for explainability
# import time  # <-- UPGRADE: Import time for real-time velocity tracking

# # 1. Initialize the FastAPI application
# app = FastAPI(title="Advanced Fraud Detection API", version="2.0")

# # 2. Define the input data structure - UPGRADED
# # We add 'nameOrig' to track user history for velocity checks.
# class Transaction(BaseModel):
#     step: float
#     nameOrig: str  # <-- UPGRADE: Added user ID
#     amount: float
#     oldbalanceOrg: float
#     newbalanceOrig: float
#     oldbalanceDest: float
#     newbalanceDest: float
#     type_CASH_IN: bool
#     type_CASH_OUT: bool
#     type_DEBIT: bool
#     type_PAYMENT: bool
#     type_TRANSFER: bool

# # 3. Load model and initialize the SHAP Explainer - UPGRADED & FIXED
# model = joblib.load('XGBoost.joblib')
# print("Model loaded successfully!")

# # Load the data sample we created for the explainer's reference
# X_train_sample = pd.read_csv('X_train_sample.csv')

# # --- THE FIX: Ensure all columns are numeric for SHAP ---
# # Loop through all columns in the sample data
# for col in X_train_sample.columns:
#     # If a column is a boolean (True/False) or object (text), convert it to integer (1/0)
#     if X_train_sample[col].dtype == 'bool' or X_train_sample[col].dtype == 'object':
#         X_train_sample[col] = X_train_sample[col].astype(int)
# print("Data types in sample file corrected for SHAP.")
# # --- END OF FIX ---

# # Create the SHAP explainer object. This is done once on startup.
# explainer = shap.TreeExplainer(model, X_train_sample)
# print("SHAP Explainer initialized successfully!")


# # 4. In-Memory Store for Real-Time Velocity - UPGRADE
# # For a hackathon, a simple dictionary is perfect for simulating a real-time database.
# # It will store { 'user_id': [timestamp1, timestamp2, ...] }
# user_history = {}


# # 5. Define the API endpoints

# @app.get("/")
# def read_root():
#     """A simple health check endpoint."""
#     return {"status": "ok", "message": "Advanced Fraud Detection API is running."}

# @app.post("/predict/")
# def predict_fraud(transaction: Transaction):
#     """
#     Accepts transaction data and returns a detailed fraud assessment,
#     including a prediction, probability, real-time context, and an explanation.
#     """
#     # --- UPGRADE 1: Real-Time Feature Calculation ---
#     user_id = transaction.nameOrig
#     current_time = time.time()
    
#     # Get user's transaction timestamps from our in-memory store
#     timestamps = user_history.get(user_id, [])
    
#     # Count how many transactions happened in the last 5 minutes (300 seconds)
#     tx_velocity_5m = sum(1 for ts in timestamps if current_time - ts < 300)

#     # Update the user's history with the current transaction time
#     timestamps.append(current_time)
#     user_history[user_id] = timestamps

#     # --- Standard Prediction Logic ---
#     data = transaction.dict()
#     # The model was not trained on 'nameOrig', so we remove it before predicting
#     data.pop('nameOrig')
    
#     input_df = pd.DataFrame([data])
    
#     # Enforce the same boolean to integer conversion for the input data
#     for col in input_df.columns:
#         if input_df[col].dtype == 'bool':
#             input_df[col] = input_df[col].astype(int)

#     fraud_probability = model.predict_proba(input_df)[0][1]
#     threshold = 0.30
#     is_flagged = bool(fraud_probability > threshold)
    
#     response = {
#         "fraud_probability": float(fraud_probability),
#         "is_flagged": is_flagged,
#         "context": {  # <-- UPGRADE: Provide real-time context with the prediction
#             "user_id": user_id,
#             "transactions_in_last_5_mins": tx_velocity_5m
#         }
#     }

#     # --- UPGRADE 2: Add Explainable AI (XAI) if flagged ---
#     if is_flagged:
#         # Calculate SHAP values to understand this specific prediction
#         shap_values = explainer.shap_values(input_df)
        
#         # Get the top 3 features that contributed most to the fraud score
#         feature_names = input_df.columns
#         feature_importance = pd.Series(np.abs(shap_values[0]), index=feature_names)
#         top_reasons = feature_importance.nlargest(3).to_dict()
        
#         # Format the reasons for a clear, readable output
#         explanation = {f"reason_{i+1}": {"feature": name, "impact_score": score} 
#                        for i, (name, score) in enumerate(top_reasons.items())}

#         response["explanation"] = explanation # Add the explanation to the response

#     return response

