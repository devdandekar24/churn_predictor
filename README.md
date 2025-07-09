
# 🔍 Customer Churn Prediction Using Machine Learning

This project aims to build a model to predict customer churn based on customer demographics, account information, and service usage using the **Telco Customer Churn dataset** from Kaggle.

> 📂 **Dataset Link**: [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📌 Objective

To build an accurate machine learning model that identifies customers likely to churn so that telecom companies can take proactive steps to retain them.

---

## ⚙️ Project Workflow

- Data cleaning and preprocessing
- Label encoding of categorical features
- **Handled class imbalance using downsampling**  
  _(SMOTE caused severe overfitting: Random Forest hit 99.8% train accuracy)_
- Training with multiple classifiers
- **Hyperparameter tuning using `RandomizedSearchCV`** (also tried GridSearchCV)
- Model evaluation using test accuracy, F1-score, recall, and ROC AUC
- Deployment using **Streamlit**

---

## 🧠 Models Trained & Compared

| Model         | ROC AUC (Test) | Accuracy | Recall (Churn) | F1-score (Churn) |
| ------------- | ------------- | -------- | -------------- | ---------------- |
| **Random Forest** ✅ | **0.8631** | **76%**   | **86%**         | **65%**           |
| XGBoost       | 0.8580        | 75%      | 82%             | 63%              |
| CatBoost      | 0.8594        | 75%      | 83%             | 63%              |

> 🔎 **Best model**: Random Forest – highest ROC AUC, best balance of recall and F1-score

---

## 📈 Model Evaluation (Random Forest)

```
Train ROC AUC: 0.7921
Test ROC AUC: 0.8631
Accuracy: 76%
Recall (Churn): 86%
F1-score (Churn): 65%
```

---

## 🧪 Preprocessing Details

| Step                | Description |
|--------------------|-------------|
| Categorical Encoding | Label Encoding via `sklearn.preprocessing.LabelEncoder` |
| SMOTE   | Used **SMOTE for upsampling**, result gave better recall for non-churn but my aim was to better predict churn, so preferred **downsampling** |
| Class Imbalance     | Used **downsampling** to balance churn/no-churn classes |
| Hyperparameter Tuning | Used `RandomizedSearchCV` with `StratifiedKFold` |
| Feature Engineering | Tried binning `tenure` into groups, but found **negligible impact**, so reverted |

---

## 🚀 Streamlit Web App

A user-friendly web app is created using **Streamlit** to allow real-time predictions.

### 🔧 Features:
- Choose between **Random Forest**, **XGBoost**, or **CatBoost**
- Input form for customer attributes
- Shows prediction result and churn probability
- All models, encoders, and scalers are reused via `pickle`

### 📎 To Run the App:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the app
streamlit run app.py
```

---
### Link for streamlit app 

[Try mychurn predictor](https://mychurn.streamlit.app/)

---

## 🧰 Tech Stack

- **Python 3.11**
- **Scikit-learn** (Label Encoding, RandomForest, CV tools)
- **CatBoost**, **XGBoost**
- **Streamlit** – for web UI
- **Pickle** – for model serialization
- **Pandas**, **NumPy**, **Seaborn**, **Matplotlib**

---

## 🛠 File Structure

```
Churn_predictor/
├── app.py                          # Streamlit frontend
├── customer_churn_randomforest.pkl
├── customer_churn_catboost.pkl
├── customer_churn_xgboost.pkl
├── encoders.pkl
├── monthlycharges_scaler.pkl
├── totalcharges_scaler.pkl
├── requirements.txt
└── README.md
```

---

## 💡 Future Work

- Try to reduce overfitting
- Explore **stacking ensemble** (e.g. RF + XGB + CatBoost)
- Dockerize for scalable cloud deployment
- Add logging and analytics dashboard to track model predictions

---

## 🙌 Acknowledgements

- [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Scikit-learn, Streamlit, XGBoost, CatBoost

---

