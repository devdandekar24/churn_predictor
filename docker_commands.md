# Docker Deployment Guide

Here are the commands you need to verify the container locally and push it to AWS.

## 1. Build the Image
Run this command in the project root (where `Dockerfile` is located):

```powershell
docker build -t churn-predictor-backend .
```

## 2. Run the Container Locally
Run the container mapping port 8000:

```powershell
docker run -p 8000:8000 churn-predictor-backend
```

## 3. Verify the Container
Once the container is running, the API will be available at `http://localhost:8000`.

### Option A: Use the verification script
I have created a python script `verify_api.py` in the root folder which tests both health check and prediction endpoints.

```powershell
# Ensure you are in the project root
python verify_api.py
```

### Option B: Using cURL (PowerShell)

**Health Check:**
```powershell
curl http://localhost:8000/health
```

**Prediction Test:**
```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0, "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}'
```

## 4. Retag and Push to AWS ECR (Reference)
When you are ready to deploy to AWS:

```powershell
# 1. Login to ECR
aws ecr get-login-password --region <YOUR_REGION> | docker login --username AWS --password-stdin <YOUR_ACCOUNT_ID>.dkr.ecr.<YOUR_REGION>.amazonaws.com

# 2. Tag the image
docker tag churn-predictor-backend:latest <YOUR_ACCOUNT_ID>.dkr.ecr.<YOUR_REGION>.amazonaws.com/churn-predictor-backend:latest

# 3. Push the image
docker push <YOUR_ACCOUNT_ID>.dkr.ecr.<YOUR_REGION>.amazonaws.com/churn-predictor-backend:latest
```
