"""Churn Analytics Dashboard — visual-first enterprise UI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from explain import (
    customer_cache_key,
    get_global_shap,
    get_local_shap,
    plot_global_bar,
    plot_importance_ranking,
    plot_local_bar,
    plot_local_waterfall,
    plot_summary_beeswarm,
)
from model import encode_input, load_model, predict_single
from utils import (
    RISK_COLORS,
    build_customer_record,
    inject_dashboard_css,
    load_raw_dataset,
    render_gauge,
    render_kpi,
)

st.set_page_config(
    page_title="Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_dashboard_css()

if "result" not in st.session_state:
    st.session_state.result = None
if "customer" not in st.session_state:
    st.session_state.customer = None


@st.cache_data
def overview_stats():
    df = load_raw_dataset()
    churn = (df["Churn"] == "Yes").mean() * 100
    return {
        "customers": len(df),
        "churn_rate": churn,
        "avg_tenure": df["tenure"].mean(),
        "avg_monthly": df["MonthlyCharges"].mean(),
    }


def render_prediction_outputs(result) -> None:
    color = RISK_COLORS[result.risk_band]
    prob_pct = result.probability * 100
    conf_pct = result.confidence * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            render_kpi("Churn Probability", f"{prob_pct:.1f}%", color=color),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_kpi("Risk Category", result.risk_band, color=color),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_kpi("Confidence", f"{conf_pct:.1f}%", color="#94a3b8"),
            unsafe_allow_html=True,
        )

    st.markdown(render_gauge(result.probability, color), unsafe_allow_html=True)
    st.caption("Churn probability gauge")


def render_input_panel() -> dict:
    with st.expander("Customer Profile", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        senior = c2.selectbox("Senior", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = c3.selectbox("Partner", ["Yes", "No"])
        dependents = c4.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 24)
        phone = st.selectbox("Phone service", ["Yes", "No"])

    with st.expander("Contract Info", expanded=True):
        c1, c2, c3 = st.columns(3)
        contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
        payment = c3.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with st.expander("Charges & Usage", expanded=True):
        c1, c2, c3 = st.columns(3)
        internet = c1.selectbox("Internet", ["Fiber optic", "DSL", "No"])
        multiple_lines = c2.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        monthly = c3.number_input("Monthly charges ($)", 0.0, 200.0, 65.0)
        c4, c5, c6, c7 = st.columns(4)
        online_security = c4.selectbox("Online security", ["Yes", "No", "No internet service"])
        online_backup = c5.selectbox("Online backup", ["Yes", "No", "No internet service"])
        device_protection = c6.selectbox("Device protection", ["Yes", "No", "No internet service"])
        tech_support = c7.selectbox("Tech support", ["Yes", "No", "No internet service"])
        c8, c9 = st.columns(2)
        streaming_tv = c8.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = c9.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

    return build_customer_record(
        gender=gender,
        senior=senior,
        partner=partner,
        dependents=dependents,
        tenure=tenure,
        phone=phone,
        multiple_lines=multiple_lines,
        internet=internet,
        online_security=online_security,
        online_backup=online_backup,
        device_protection=device_protection,
        tech_support=tech_support,
        streaming_tv=streaming_tv,
        streaming_movies=streaming_movies,
        contract=contract,
        paperless=paperless,
        payment=payment,
        monthly=monthly,
    )


def tab_overview(pipeline) -> None:
    stats = overview_stats()
    global_shap = get_global_shap(id(pipeline), pipeline)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{stats['customers']:,}")
    c2.metric("Churn Rate", f"{stats['churn_rate']:.1f}%")
    c3.metric("Avg Tenure", f"{stats['avg_tenure']:.0f} mo")
    c4.metric("Avg Monthly", f"${stats['avg_monthly']:.2f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.pyplot(plot_importance_ranking(global_shap, top_n=10))
    with col_r:
        st.pyplot(plot_global_bar(global_shap, max_display=10))


def tab_prediction(pipeline) -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        customer = render_input_panel()
        run = st.button("Run Prediction", type="primary", use_container_width=True)

    with right:
        st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)
        if run:
            df = pd.DataFrame([customer])
            st.session_state.result = predict_single(pipeline, df)
            st.session_state.customer = customer

        if st.session_state.result is None:
            st.empty()
            return

        render_prediction_outputs(st.session_state.result)

    if st.session_state.result and st.session_state.customer:
        customer = st.session_state.customer
        customer_df = pd.DataFrame([customer])
        encoded = tuple(encode_input(pipeline, customer_df).flatten().tolist())
        local_shap = get_local_shap(
            id(pipeline),
            pipeline,
            customer_cache_key(customer),
            encoded,
        )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_local_waterfall(local_shap, max_display=10))
        with c2:
            st.pyplot(plot_local_bar(local_shap, max_display=10))


def tab_feature_importance(pipeline) -> None:
    global_shap = get_global_shap(id(pipeline), pipeline)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_summary_beeswarm(global_shap, max_display=12))
    with c2:
        st.pyplot(plot_importance_ranking(global_shap, top_n=12))

    st.pyplot(plot_global_bar(global_shap, max_display=15))


def main() -> None:
    pipeline = load_model()

    st.markdown('<div class="dash-header">Churn Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-sub">Customer retention intelligence</div>',
        unsafe_allow_html=True,
    )

    overview_tab, prediction_tab, importance_tab = st.tabs(
        ["Overview", "Prediction", "Feature Importance"]
    )

    with overview_tab:
        tab_overview(pipeline)

    with prediction_tab:
        tab_prediction(pipeline)

    with importance_tab:
        tab_feature_importance(pipeline)


if __name__ == "__main__":
    main()
