import pickle
import pandas as pd
from typing import Dict


# -------------------------------
# Load Models
# -------------------------------
models = {}

for model_name in ["randomforest", "xgboost", "catboost"]:
    with open(f"customer_churn_{model_name}.pkl", "rb") as f:
        data = pickle.load(f)
        models[model_name] = {
            "model": data["model"],
            "features": data["features_names"]
        }


# -------------------------------
# Load Encoders and Scalers
# -------------------------------
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("monthlycharges_scaler.pkl", "rb") as f:
    monthly_scaler = pickle.load(f)

with open("totalcharges_scaler.pkl", "rb") as f:
    total_scaler = pickle.load(f)


# -------------------------------
# Model Version (optional for MLflow later)
# -------------------------------
MODEL_VERSION = "1.0.0"


# -------------------------------
# Prediction Function
# -------------------------------
def predict_output(user_input: Dict, model_name: str = "randomforest") -> Dict:

    model_name = model_name.lower()
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not available.")

    model = models[model_name]["model"]
    features = models[model_name]["features"]

    # Convert input to DataFrame
    df_input = pd.DataFrame([user_input])

    # Ensure correct feature order
    df_input = df_input[features]

    # -------------------------------
    # Encode categorical columns
    # -------------------------------
    for col, encoder in encoders.items():
        if col in df_input.columns:
            df_input[col] = encoder.transform(df_input[col])

    # -------------------------------
    # Scale numerical columns
    # -------------------------------
    if "MonthlyCharges" in df_input.columns:
        df_input["MonthlyCharges"] = monthly_scaler.transform(
            df_input[["MonthlyCharges"]]
        )

    if "TotalCharges" in df_input.columns:
        df_input["TotalCharges"] = total_scaler.transform(
            df_input[["TotalCharges"]]
        )

    # -------------------------------
    # Predict
    # -------------------------------
    prediction = model.predict(df_input)[0]
    probabilities = model.predict_proba(df_input)[0]

    churn_prob = float(probabilities[1])
    no_churn_prob = float(probabilities[0])

    predicted_label = "Churn" if prediction == 1 else "No Churn"
    confidence = max(churn_prob, no_churn_prob)

    return {
        "predicted_label": predicted_label,
        "churn_probability": round(churn_prob, 4),
        "no_churn_probability": round(no_churn_prob, 4),
        "confidence": round(confidence, 4)
    }
