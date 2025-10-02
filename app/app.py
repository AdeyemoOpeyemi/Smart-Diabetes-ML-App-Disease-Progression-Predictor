# app.py — Smart Diabetes Prediction App

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Progression Predictor", page_icon="🩺", layout="centered")

# -----------------------------
# Core configuration
# -----------------------------
ALL_FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
ENGINEERED_FEATURES = ["bmi_squared", "age_bmi"]
EXPECTED_FEATURES = ALL_FEATURES + ENGINEERED_FEATURES

DEFAULT_VALUES = {
    "age": 50, "sex": 1, "bmi": 25.0, "bp": 80.0,
    "s1": 100.0, "s2": 100.0, "s3": 100.0, "s4": 100.0,
    "s5": 100.0, "s6": 100.0
}

SEX_MAP = {"Male": 1, "Female": 0}

# -----------------------------
# Utilities
# -----------------------------
def safe_float(value, default):
    try:
        return float(value)
    except:
        return default

def ensure_features(df, features, defaults):
    for col in features:
        if col not in df.columns:
            df[col] = defaults[col]
    return df[features]

def encode_upload(df_raw, selected_features, defaults):
    df = df_raw.copy()

    if "sex" in df.columns and "sex" in selected_features:
        df["sex"] = df["sex"].map(SEX_MAP).fillna(defaults["sex"]).astype(int)

    for col in df.columns:
        if col in selected_features and col != "sex":
            df[col] = df[col].apply(lambda x: safe_float(x, defaults[col]))

    # Feature engineering
    if "bmi" in df.columns:
        df["bmi_squared"] = df["bmi"] ** 2
    if "age" in df.columns and "bmi" in df.columns:
        df["age_bmi"] = df["age"] * df["bmi"]

    return ensure_features(df, EXPECTED_FEATURES, defaults)

def predict_with_model(model, X):
    return model.predict(X)

# -----------------------------
# Load model from models/ folder under Diabetes/
# -----------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up from app/
    model_path = os.path.join(BASE_DIR, "model", "best_model.pkl")  # <- models/ under Diabetes/

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please place 'best_model.pkl' inside the 'models/' folder under Diabetes.")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Sidebar: feature selection
# -----------------------------
st.sidebar.header("Select Features to Include")
selected_features = st.sidebar.multiselect(
    "Pick features to include",
    options=ALL_FEATURES,
    default=ALL_FEATURES
)

if not selected_features:
    st.warning("Please select at least one feature to continue.")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Diabetes Progression Predictor")
st.write("Predict diabetes progression using manual inputs or file upload. Missing values are auto-filled.")

mode = st.radio("Choose input method:", ["Manual Entry", "Upload File"], horizontal=True)

# -----------------------------
# Manual Entry
# -----------------------------
if mode == "Manual Entry":
    st.subheader("🔢 Enter Patient Details")
    user_input = {}
    col1, col2 = st.columns(2)

    for feature in selected_features:
        if feature == "sex":
            val = col1.selectbox("Sex", ["Male", "Female"])
            user_input["sex"] = SEX_MAP[val]
        elif feature == "age":
            user_input["age"] = col1.slider("Age", 0, 120, DEFAULT_VALUES["age"])
        elif feature == "bmi":
            user_input["bmi"] = col1.slider("BMI", 10.0, 50.0, DEFAULT_VALUES["bmi"])
        elif feature == "bp":
            user_input["bp"] = col2.slider("Blood Pressure", 40.0, 120.0, DEFAULT_VALUES["bp"])
        else:
            user_input[feature] = col2.number_input(f"{feature}", value=DEFAULT_VALUES[feature])

    # Feature engineering
    if "bmi" in user_input:
        user_input["bmi_squared"] = user_input["bmi"] ** 2
    if "age" in user_input and "bmi" in user_input:
        user_input["age_bmi"] = user_input["age"] * user_input["bmi"]

    input_df = ensure_features(pd.DataFrame([user_input]), EXPECTED_FEATURES, DEFAULT_VALUES)

    if st.button("Predict"):
        prediction = predict_with_model(model, input_df)[0]
        st.success(f"📈 Predicted Disease Progression Score: {prediction:.2f}")

# -----------------------------
# File Upload
# -----------------------------
else:
    st.subheader("📁 Upload CSV or Excel File")
    st.write("Include any of the features you selected in the sidebar. Missing fields are auto-filled.")

    # Dynamic template CSV
    template_df = pd.DataFrame([{f: DEFAULT_VALUES[f] for f in selected_features}])
    csv_data = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Template", data=csv_data, file_name="diabetes_input_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df_raw.head())

        X = encode_upload(df_raw, selected_features, DEFAULT_VALUES)
        preds = predict_with_model(model, X)

        out_df = df_raw.copy()
        out_df["Predicted_Progression"] = preds

        st.success("✅ Predictions complete")
        st.dataframe(out_df)

        csv_out = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", data=csv_out, file_name="predictions.csv", mime="text/csv")

# -----------------------------
# Help section
# -----------------------------
with st.expander("ℹ️ How this app works"):
    st.markdown(f"""
- Enter data manually or upload a CSV/Excel file for bulk predictions.
- Missing values are automatically filled with smart defaults.
- Sidebar allows you to select which features to include (currently selected: {', '.join(selected_features)}).
- Feature engineering (`bmi_squared` and `age_bmi`) is applied automatically.
- Ensure `best_model.pkl` is inside the `models/` folder under Diabetes.
""")
