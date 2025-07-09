# 📊 Customer Churn Prediction

A Machine Learning project to predict customer churn using the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## 🚀 Overview

This project focuses on building predictive models to identify customers likely to churn. The complete pipeline includes:

* Data preprocessing & feature engineering
* Handling class imbalance (SMOTE and downsampling)
* Model training, hyperparameter tuning (GridSearchCV)
* Model comparison (Random Forest, XGBoost, CatBoost)
* Streamlit-based deployment for user-friendly predictions

---

## 📊 Dataset

* Dataset: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* Target variable: `Churn` (Yes/No)

---

## 🎯 Model Comparison

| Metric               | Random Forest | CatBoost | XGBoost |
| -------------------- | ------------- | -------- | ------- |
| **Train ROC AUC**    | 0.7921        | 0.7937   | 0.7814  |
| **Test ROC AUC**     | **0.8631** ✅  | 0.8594   | 0.8580  |
| **Accuracy**         | **76%** ✅     | 75%      | 75%     |
| **Recall (Churn)**   | **86%** ✅     | 83%      | 82%     |
| **F1-score (Churn)** | **0.65** ✅    | 0.63     | 0.63    |

> ⭐ **Best Performing Model**: **Random Forest Classifier**

---

## 📅 Features Used

* Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* Services: `PhoneService`, `InternetService`, `OnlineSecurity`, `StreamingTV`, etc.
* Billing: `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

---

## 📆 Preprocessing

* Categorical Encoding using LabelEncoders (stored using `pickle`)
* Numeric Feature Scaling for `MonthlyCharges` and `TotalCharges`
* SMOTE + Downsampling used to balance class distribution

---

## ⚖️ Model Training

* Train/Test Split with stratification
* StratifiedKFold Cross Validation
* GridSearchCV for tuning:

  * `RandomForestClassifier`
  * `XGBoostClassifier`
  * `CatBoostClassifier`

---

## 🏙️ Deployment

The project includes a fully interactive **Streamlit Web App**:

* Select model: Random Forest, CatBoost, XGBoost
* Input customer details manually
* Prediction + Probability of churn

### Run the app locally:

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

* Python 3
* Pandas, NumPy
* Scikit-learn
* XGBoost, CatBoost
* SMOTE (imbalanced-learn)
* Streamlit (deployment)
* Pickle (model persistence)

---

## 📊 Future Enhancements

* SHAP/ELI5 explainability
* Try Voting Classifier or model stacking
* Host on Streamlit Cloud / Hugging Face Spaces
* Add feature selection or dimensionality reduction

---

## 🎉 Acknowledgements

* Dataset: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📦 Project Setup (Optional)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

> 🌟 Feel free to fork this project and experiment with new models, encoders, or app layouts!

---
