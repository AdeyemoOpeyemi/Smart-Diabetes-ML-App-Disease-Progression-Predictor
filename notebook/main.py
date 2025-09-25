# ===============================
# diabetes_predict_console.py
# ===============================
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# Load model
# ===============================
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    print("❌ Error: best_model.pkl not found.")
    exit()

# ===============================
# Define expected features and defaults
# ===============================
expected_features = [
    "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6",
    "bmi_squared", "age_bmi"
]

default_values = {
    "age": 50, "sex": 1, "bmi": 25.0, "bp": 80.0,
    "s1": 100.0, "s2": 100.0, "s3": 100.0, "s4": 100.0,
    "s5": 100.0, "s6": 100.0
}

# ===============================
# Console App
# ===============================
def main():
    print("🩺 Diabetes Progression Predictor (Console Version)")
    print("Type 'exit' at any prompt to quit.\n")

    while True:
        print("Choose input method:")
        print("1 - Manual Entry")
        print("2 - Upload CSV/Excel File")
        choice = input("Enter 1 or 2: ").strip()
        if choice.lower() == "exit":
            break

        input_df = None

        # -----------------------------
        # Manual Entry Mode
        # -----------------------------
        if choice == "1":
            user_input = {}
            print("\nEnter patient details:")

            for feature in ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]:
                val = input(f"{feature} (default={default_values[feature]}): ").strip()
                if val.lower() == "exit":
                    return

                if feature == "sex":
                    if val.lower() in ["male", "m", "1"]:
                        user_input["sex"] = 1
                    elif val.lower() in ["female", "f", "0"]:
                        user_input["sex"] = 0
                    else:
                        user_input["sex"] = default_values["sex"]
                else:
                    try:
                        user_input[feature] = float(val) if val else default_values[feature]
                    except:
                        user_input[feature] = default_values[feature]

            # Feature engineering
            user_input["bmi_squared"] = user_input["bmi"] ** 2
            user_input["age_bmi"] = user_input["age"] * user_input["bmi"]

            input_df = pd.DataFrame([user_input])

        # -----------------------------
        # File Upload Mode
        # -----------------------------
        elif choice == "2":
            path = input("Enter CSV or Excel file path: ").strip()
            if path.lower() == "exit":
                break
            if not os.path.exists(path):
                print("❌ File not found. Try again.")
                continue

            try:
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                continue

            # Fill missing columns with defaults
            for col in ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]:
                if col not in df.columns:
                    df[col] = default_values[col]

            # Encode sex if needed
            if "sex" in df.columns:
                df["sex"] = df["sex"].map({"Male": 1, "Female": 0}).fillna(default_values["sex"])

            # Feature engineering
            df["bmi_squared"] = df["bmi"] ** 2
            df["age_bmi"] = df["age"] * df["bmi"]

            # Ensure column order
            input_df = df[expected_features]

        else:
            print("❌ Invalid choice. Try again.")
            continue

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(input_df)
        input_df["Predicted_Progression"] = predictions

        print("\n✅ Predictions:")
        print(input_df)

        # Save to CSV
        save = input("\nDo you want to save results to CSV? (y/n): ").strip().lower()
        if save == "y":
            out_file = input("Enter output CSV file name: ").strip()
            input_df.to_csv(out_file, index=False)
            print(f"Results saved to {out_file}")

        cont = input("\nDo you want to predict again? (y/n): ").strip().lower()
        if cont != "y":
            break

if __name__ == "__main__":
    main()
