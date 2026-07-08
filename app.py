import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details below and click Predict.")

# -----------------------------
# User Inputs
# -----------------------------
time = st.number_input("Time", min_value=0.0, value=0.0)

features = []

for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0, format="%.6f")
    features.append(value)

amount = st.number_input("Amount", min_value=0.0, value=0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    columns = [
        "Time",
        "V1","V2","V3","V4","V5","V6","V7",
        "V8","V9","V10","V11","V12","V13",
        "V14","V15","V16","V17","V18","V19",
        "V20","V21","V22","V23","V24","V25",
        "V26","V27","V28",
        "Amount"
    ]

    data = [[time] + features + [amount]]

    df = pd.DataFrame(data, columns=columns)

    try:
        # Scale Time and Amount
        df["Time"] = scaler.transform(df[["Time"]]).flatten()
        df["Amount"] = scaler.transform(df[["Amount"]]).flatten()

        prediction = model.predict(df)

        if prediction[0] == 0:
            st.success("✅ Legitimate Transaction")
        else:
            st.error("🚨 Fraudulent Transaction")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
