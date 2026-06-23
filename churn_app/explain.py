"""SHAP visual analytics — plots only, no text generation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from model import encode_input, get_background_encoded, get_feature_names, get_xgb_model, to_dense_array
from utils import humanize_feature


_explainer_cache: dict[int, shap.Explainer] = {}


def _get_explainer(pipeline) -> shap.Explainer:
    key = id(pipeline)
    if key not in _explainer_cache:
        model = get_xgb_model(pipeline)
        background = get_background_encoded(pipeline)

        def churn_proba(X: np.ndarray) -> np.ndarray:
            return model.predict_proba(to_dense_array(X))[:, 1]

        _explainer_cache[key] = shap.Explainer(churn_proba, background)
    return _explainer_cache[key]


def _to_explanation(values, data, base_values, feature_names) -> shap.Explanation:
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
    )


def _run_shap(explainer, matrix: np.ndarray, pipeline) -> shap.Explanation:
    result = explainer(matrix)
    names = [humanize_feature(f) for f in get_feature_names(pipeline)]
    result.feature_names = names
    return result


@st.cache_data(show_spinner="Computing global SHAP…")
def get_global_shap(_pipeline_id: int, _pipeline) -> tuple:
    background = get_background_encoded(_pipeline)
    sample = background[:50]
    explainer = _get_explainer(_pipeline)
    result = _run_shap(explainer, sample, _pipeline)
    return (
        np.asarray(result.values),
        np.asarray(result.data),
        np.asarray(result.base_values),
        list(result.feature_names),
    )


@st.cache_data(show_spinner="Computing local SHAP…")
def get_local_shap(_pipeline_id: int, _pipeline, customer_key: str, encoded_tuple: tuple) -> tuple:
    encoded = np.array(encoded_tuple).reshape(1, -1)
    explainer = _get_explainer(_pipeline)
    result = _run_shap(explainer, encoded, _pipeline)
    return (
        np.asarray(result.values),
        np.asarray(result.data),
        np.asarray(result.base_values),
        list(result.feature_names),
    )


def _unpack_global(cached: tuple) -> shap.Explanation:
    values, data, base, names = cached
    return _to_explanation(values, data, base, names)


def _unpack_local(cached: tuple) -> shap.Explanation:
    values, data, base, names = cached
    return _to_explanation(values, data, base, names)


def plot_summary_beeswarm(cached: tuple, max_display: int = 12) -> plt.Figure:
    plt.close("all")
    fig = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(_unpack_global(cached), max_display=max_display, show=False)
    fig.tight_layout()
    return fig


def plot_global_bar(cached: tuple, max_display: int = 12) -> plt.Figure:
    plt.close("all")
    fig = plt.figure(figsize=(10, 6))
    shap.plots.bar(_unpack_global(cached), max_display=max_display, show=False)
    fig.tight_layout()
    return fig


def plot_importance_ranking(cached: tuple, top_n: int = 12) -> plt.Figure:
    values, _, _, names = cached
    if values.ndim == 3:
        values = values[:, :, 1]
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    y = np.arange(len(order))
    ax.barh(y, mean_abs[order], color="#3b82f6", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_local_waterfall(cached: tuple, max_display: int = 12) -> plt.Figure:
    plt.close("all")
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(_unpack_local(cached)[0], max_display=max_display, show=False)
    fig.tight_layout()
    return fig


def plot_local_bar(cached: tuple, max_display: int = 12) -> plt.Figure:
    plt.close("all")
    fig = plt.figure(figsize=(10, 5))
    shap.plots.bar(_unpack_local(cached)[0], max_display=max_display, show=False)
    fig.tight_layout()
    return fig


def customer_cache_key(customer: dict) -> str:
    return "|".join(f"{k}={customer[k]}" for k in sorted(customer))
