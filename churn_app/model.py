"""Model loading and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pickle
import streamlit as st

from utils import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    get_confidence,
    get_risk_band,
    load_features_dataset,
)


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    prediction: int
    risk_band: str
    confidence: float


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_preprocessor(pipeline):
    return pipeline.named_steps["preprocesor"]


def get_xgb_model(pipeline):
    return pipeline.named_steps["model"]


def get_feature_names(pipeline) -> list[str]:
    return list(get_preprocessor(pipeline).get_feature_names_out())


def to_dense_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


@st.cache_resource(show_spinner="Preparing SHAP background…")
def get_background_encoded(_pipeline):
    sample = load_features_dataset().sample(n=min(100, 7043), random_state=42)
    preprocessor = get_preprocessor(_pipeline)
    return to_dense_array(preprocessor.transform(sample))


def encode_input(pipeline, df: pd.DataFrame) -> np.ndarray:
    preprocessor = get_preprocessor(pipeline)
    return to_dense_array(preprocessor.transform(df[FEATURE_COLUMNS]))


def predict_single(pipeline, customer_df: pd.DataFrame) -> PredictionResult:
    prob = float(pipeline.predict_proba(customer_df)[0][1])
    return PredictionResult(
        probability=prob,
        prediction=int(prob >= 0.5),
        risk_band=get_risk_band(prob),
        confidence=get_confidence(prob),
    )
