
import streamlit as st
import pandas as pd


# Load model and scaler
import joblib

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details below.")

# Inputs
time = st.number_input("Time", value=0.0)

values = []

for i in range(1,29):
    values.append(st.number_input(f"V{i}", value=0.0))

amount = st.number_input("Amount", value=0.0)

if st.button("Predict"):

    data = [[time] + values + [amount]]

    columns = [
        "Time",
        "V1","V2","V3","V4","V5","V6","V7",
        "V8","V9","V10","V11","V12","V13",
        "V14","V15","V16","V17","V18","V19",
        "V20","V21","V22","V23","V24","V25",
        "V26","V27","V28",
        "Amount"
    ]

    df = pd.DataFrame(data, columns=columns)

    # Scale Time and Amount
    df["Time"] = scaler.transform(df[["Time"]])
    df["Amount"] = scaler.transform(df[["Amount"]])

    prediction = model.predict(df)

    if prediction[0] == 0:
        st.success("✅ Legitimate Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")
