import streamlit as st
import requests
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="AI Fraud Detection System", page_icon="🛡️", layout="wide")

# --- Header ---
st.title("🛡️ AI-Powered Fraud Detection System")
st.markdown("A real-time dashboard for analyzing transaction risk using an explainable AI model.")

# --- Key Performance Indicators (The Business Impact) ---
st.header("Demonstrated Performance")
# These metrics are based on your model's evaluation on the test set.
# They are designed to show business value to non-technical judges.
col1, col2, col3 = st.columns(3)
col1.metric("Fraud Detection Rate (Recall)", "97.32%", "On Unseen Test Data")
col2.metric("False Positive Rate", "0.1%", "Minimal Customer Disruption")
col3.metric("Estimated Monthly Loss Prevented", "₹78.5Cr", "Based on Test Set Performance")

# --- Live Analysis Form ---
with st.form("transaction_form"):
    st.header("Live Transaction Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        nameOrig = st.text_input("Originator ID", "C987654321")
        amount = st.number_input("Amount ($)", value=185432.78, format="%.2f")
        tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
    with col2:
        oldbalanceOrg = st.number_input("Originator Old Balance", value=185432.78, format="%.2f")
        newbalanceOrig = st.number_input("Originator New Balance", value=0.0, format="%.2f")
    with col3:
        oldbalanceDest = st.number_input("Destination Old Balance", value=50000.0, format="%.2f")
        newbalanceDest = st.number_input("Destination New Balance", value=235432.78, format="%.2f")
    
    # Hidden field for 'step' - not important for user input in a demo
    step = 1 

    submitted = st.form_submit_button("ANALYZE TRANSACTION", use_container_width=True)

# --- Results Section ---
if submitted:
    # Create the payload for the API
    type_payload = {f"type_{t}": (t == tx_type) for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]}
    payload = {
        "step": step, "nameOrig": nameOrig, "amount": amount, "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig, "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest, **type_payload
    }
    
    try:
        # Show a spinner while the API is being called
        with st.spinner("Consulting our AI Analyst..."):
            response = requests.post("http://127.0.0.1:8000/predict_and_explain/", json=payload)
            result = response.json()

        # Display the results in a user-friendly way
        if result.get("is_flagged"):
            st.error(f"**ALERT: HIGH RISK OF FRAUD DETECTED** (Confidence: {result.get('fraud_probability', 0):.2%})", icon="🚨")
        else:
            st.success(f"**Transaction Cleared: Low Risk** (Confidence: {1-result.get('fraud_probability', 0):.2%})", icon="✅")

        st.subheader("AI Analyst's Summary")
        # Use a special container for the explanation to make it stand out
        with st.container(border=True):
            st.write(result.get("natural_language_explanation", "No summary available."))
        
        # Add an expander for the technical "nerd details"
        with st.expander("Show Technical Details (for Analysts)"):
            st.json(result)
            
    except requests.exceptions.RequestException:
        st.error("Could not connect to the API. Please ensure the `uvicorn` server is running in a separate terminal.", icon="🔌")

