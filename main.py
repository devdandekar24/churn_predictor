from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Literal

from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn using multiple ML models",
    version="1.0.0"
)


# -------------------------------
# Root Endpoint
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running 🚀"
    }


# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict_churn(
    user_input: UserInput,
    model_name: Literal["randomforest", "xgboost", "catboost"] = Query(
        "randomforest",
        description="Select the ML model to use for prediction"
    )
):
    try:
        result = predict_output(
            user_input=user_input.model_dump(),
            model_name=model_name
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
