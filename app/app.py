# app.py — Smart Diabetes Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("best_model.pkl")

# Define expected features and defaults
expected_features = [
    "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6",
    "bmi_squared", "age_bmi"
]

default_values = {
    "age": 50, "sex": 1, "bmi": 25.0, "bp": 80.0,
    "s1": 100.0, "s2": 100.0, "s3": 100.0, "s4": 100.0,
    "s5": 100.0, "s6": 100.0
}

# App title
st.title("🩺 Diabetes Progression Predictor")
st.write("Choose features to include manually or upload a file. Missing values will be handled automatically.")

# Input mode
mode = st.radio("Choose input method:", ["Manual Entry", "Upload File"])

# -----------------------------
# Manual Entry Mode
# -----------------------------
if mode == "Manual Entry":
    st.subheader("🔢 Select and Enter Features")

    user_input = {}

    for feature in default_values:
        if st.checkbox(f"Include {feature}"):
            if feature == "sex":
                user_input["sex"] = st.selectbox("Sex", ["Male", "Female"])
                user_input["sex"] = 1 if user_input["sex"] == "Male" else 0
            elif feature == "age":
                user_input["age"] = st.slider("Age", 0, 120, default_values["age"])
            elif feature == "bmi":
                user_input["bmi"] = st.slider("BMI", 10.0, 50.0, default_values["bmi"])
            elif feature == "bp":
                user_input["bp"] = st.slider("Blood Pressure", 40.0, 120.0, default_values["bp"])
            else:
                user_input[feature] = st.number_input(f"{feature}", value=default_values[feature])
        else:
            user_input[feature] = default_values[feature]

    # Feature engineering
    user_input["bmi_squared"] = user_input["bmi"] ** 2
    user_input["age_bmi"] = user_input["age"] * user_input["bmi"]

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.success(f"📈 Predicted Disease Progression Score: {prediction:.2f}")

# -----------------------------
# File Upload Mode
# -----------------------------
else:
    st.subheader("📁 Upload CSV or Excel File")
    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Fill missing columns with defaults
        for col in default_values:
            if col not in df.columns:
                df[col] = default_values[col]

        # Encode sex if needed
        if "sex" in df.columns:
            df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).fillna(default_values["sex"])

        # Feature engineering
        df["bmi_squared"] = df["bmi"] ** 2
        df["age_bmi"] = df["age"] * df["bmi"]

        # Ensure column order
        df = df[expected_features]

        # Predict
        predictions = model.predict(df)
        df["Predicted_Progression"] = predictions

        st.success("✅ Predictions complete")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", data=csv, file_name="predictions.csv", mime="text/csv")
