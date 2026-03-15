import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model
model = joblib.load("model/stress_model.pkl")

st.title("🧠 Stress Detection App (Cloud-Friendly)")
st.write("Upload CSV files with pre-computed features (mean, std, etc.) to predict stress levels.")

# Upload CSV
csv_file = st.file_uploader("Upload CSV with features", type=["csv"])
if csv_file:
    df = pd.read_csv(csv_file)
    prediction = model.predict(df)
    st.write("Predicted stress levels:", prediction)
