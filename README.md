# Telco-churn-prediction
Telco Customer Churn Prediction | EDA + XGBoost + SHAP + Streamlit App + Power BI Dashboard

## 🗂️ Project Structure

```
churn_prediction_app/
├── churn_app/
│   ├── app.py
│   ├── model.pkl
│   └── requirements.txt
└── Notebook/
    ├── churn_EDA_Prediction.ipynb
    └── Telco-Customer-Churn.csv
```

---

## 🔍 Overview

This project analyzes customer churn behavior for a telecom company using the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

It covers the full ML pipeline:
- Exploratory Data Analysis (EDA)
- Feature Selection using Mutual Information
- Predictive Modeling with XGBoost
- Model Explainability with SHAP
- Interactive Streamlit App
- Power BI Dashboard

---

## 📊 Power BI Dashboard

![Dashboard](dashboard.png)

---

## 🤖 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 74% |
| Churn Recall | 75% |
| Churn Precision | 51% |
| ROC AUC | 0.82 |

---

## 🔑 Key Findings

- Customers on **Month-to-month contracts** are at the highest churn risk
- **Short tenure** and **high MonthlyCharges** strongly increase churn probability
- **Fiber Optic** users show a higher tendency to churn
- **Electronic check** payment method correlates with higher churn
- **Two-year contracts** are the strongest protective factor against churn

---

## 🚀 Run the App

```bash
cd churn_app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-green)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![PowerBI](https://img.shields.io/badge/PowerBI-yellow)

---

## 👩‍💻 Author

**Monira Ashraf** — [GitHub](https://github.com/moniraahraf) | 
[Kaggle](https://kaggle.com/moniraashraf)
