"""Shared constants and helpers for the churn analytics dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = APP_DIR / "model.pkl"
DATA_PATH = PROJECT_DIR / "Notebook" / "Telco-Customer-Churn.csv"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

RISK_THRESHOLDS = {"low": 0.3, "high": 0.7}

RISK_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#ef4444",
}


def inject_dashboard_css() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; max-width: 1200px; }
        .dash-header {
            font-size: 1.75rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 0.25rem;
        }
        .dash-sub {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        .kpi-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            text-align: center;
            min-height: 110px;
        }
        .kpi-label {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .kpi-sub { color: #64748b; font-size: 0.78rem; margin-top: 0.3rem; }
        .section-label {
            color: #cbd5e1;
            font-size: 0.9rem;
            font-weight: 600;
            margin: 1rem 0 0.5rem 0;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid #334155;
        }
        .gauge-track {
            background: #334155;
            border-radius: 6px;
            height: 14px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .gauge-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, sub: str = "", color: str = "#f8fafc") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def render_gauge(probability: float, color: str) -> str:
    pct = min(max(probability * 100, 0), 100)
    return f"""
    <div class="gauge-track">
        <div class="gauge-fill" style="width:{pct:.1f}%; background:{color};"></div>
    </div>
    """


def get_risk_band(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    if probability <= RISK_THRESHOLDS["high"]:
        return "Medium"
    return "High"


def get_confidence(probability: float) -> float:
    return abs(probability - 0.5) * 2


def humanize_feature(name: str) -> str:
    if name.startswith("cat__"):
        return name.replace("cat__", "").replace("_", " ")
    if name.startswith("num__"):
        return name.replace("num__", "").replace("_", " ")
    return name.replace("_", " ")


def load_raw_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_features_dataset() -> pd.DataFrame:
    df = load_raw_dataset()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    return df


def build_customer_record(
    gender: str,
    senior: int,
    partner: str,
    dependents: str,
    tenure: int,
    phone: str,
    multiple_lines: str,
    internet: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    contract: str,
    paperless: str,
    payment: str,
    monthly: float,
    total_charges: float | None = None,
) -> dict:
    total = total_charges if total_charges is not None else monthly * tenure
    return {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
