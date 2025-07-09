import streamlit as st
import pandas as pd
import pickle

st.title("🔍 Customer Churn Prediction App")

# Load all models
models = {}
for model_name in ["randomforest", "xgboost", "catboost"]:
    with open(f"customer_churn_{model_name}.pkl", "rb") as f:
        data = pickle.load(f)
        models[model_name] = {
            "model": data["model"],
            "features": data["features_names"]
        }

# Load encoders and scalers
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("monthlycharges_scaler.pkl", "rb") as f:
    monthly_scaler = pickle.load(f)

with open("totalcharges_scaler.pkl", "rb") as f:
    total_scaler = pickle.load(f)

# Feature options dictionary
feature_options = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['No phone service', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ]
}

# Select model
model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "CatBoost"])
model_key = model_choice.lower().replace(" ", "")
model = models[model_key]["model"]
features = models[model_key]["features"]

st.markdown("### ✍️ Input Customer Details")
input_data = {}
for feature in features:
    if feature in ['MonthlyCharges', 'TotalCharges','tenure']:
        input_data[feature] = st.number_input(f"{feature}", step=1.0)
    elif feature in feature_options:
        input_data[feature] = st.selectbox(feature, feature_options[feature])
    else:
        st.warning(f"Unexpected feature not handled in UI: {feature}")

if st.button("Predict Churn"):
    df_input = pd.DataFrame([input_data])

    # Encode categorical columns
    for col, encoder in encoders.items():
        df_input[col] = encoder.transform(df_input[col])

    # Scale numerical columns
    df_input["MonthlyCharges"] = monthly_scaler.transform(df_input[["MonthlyCharges"]])
    df_input["TotalCharges"] = total_scaler.transform(df_input[["TotalCharges"]])

    # Predict
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    st.subheader("🔮 Prediction Result")
    st.write("😊 **No Churn**" if prediction == 0 else "⚠️ **Customer is likely to Churn**")
    st.write(f"🔢 Probability of churn: **{prob:.2%}**")
    
st.markdown("📂 [View Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")

