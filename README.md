#  Smart Diabetes ML App — Disease Progression Predictor

This project builds a **machine learning model** and an **interactive Streamlit web app** to predict **diabetes disease progression scores** using the **Diabetes dataset**.  
It supports **manual entry** of patient details and **file upload (CSV/Excel)** for batch predictions.

---

##  Features

- Train a regression model on the Diabetes dataset.
- Predict disease progression based on:
  - Age
  - Sex
  - BMI
  - Blood Pressure
  - Serum measurements (`s1`–`s6`)
- Feature engineering:
  - BMI² (non-linear effect of BMI)
  - Age × BMI interaction
- Streamlit app supports:
  - Manual entry (with smart defaults).
  - Bulk predictions via CSV/Excel upload.
- Results can be downloaded as CSV.

---


---

##  Dataset

- **Name**: Diabetes dataset (scikit-learn version)
- **Target Variable**: Disease progression score (continuous regression target)
- **Key Features**:
  - Age
  - Sex (Male/Female → encoded as 1/0)
  - BMI
  - Blood Pressure (BP)
  - Serum values: s1, s2, s3, s4, s5, s6
- **Engineered Features**:
  - `bmi_squared`
  - `age_bmi`

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-AdeyemoOpeyemi/smart-diabetes-ml-app.git
   cd smart-diabetes-ml-app


