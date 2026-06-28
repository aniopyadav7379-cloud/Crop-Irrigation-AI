"""
dashboard/app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit multi-page dashboard for the Crop-Irrigation-AI system.

Run:
    streamlit run dashboard/app.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when launched from dashboard/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from config import settings
from utils import logger
from dashboard.visualization import (
    render_kpi_cards,
    render_crop_distribution_chart,
    render_stress_donut,
    render_weather_forecast_chart,
    render_advisory_table,
    render_model_accuracy_bars,
)
from dashboard.maps import (
    render_crop_map,
    render_deficit_map,
    render_stress_map,
    render_zone_advisory_map,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crop Irrigation AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .stMetric { background: #F0FAF5; border-radius: 8px; padding: 0.5rem; }
      .stMetric label { font-size: 12px !important; color: #666 !important; }
      div[data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    st.sidebar.image(
        "https://img.icons8.com/color/96/satellite.png", width=60
    )
    st.sidebar.title("🌾 Crop Irrigation AI")
    st.sidebar.caption(f"v{settings.project_version}  ·  Kharif 2026")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🗺️ Maps", "💧 Advisory", "📈 Analytics", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    season = st.sidebar.selectbox("Season", ["Kharif 2026", "Rabi 2025-26"])
    zones = st.sidebar.multiselect(
        "Zones",
        ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"],
        default=["Zone 1", "Zone 2", "Zone 3"],
    )
    date_range = st.sidebar.date_input("Date range", [])

    st.sidebar.divider()
    if st.sidebar.button("🔄  Run Pipeline", use_container_width=True):
        with st.spinner("Running satellite data pipeline …"):
            try:
                from main import run_pipeline
                run_pipeline()
                st.sidebar.success("Pipeline complete!")
            except Exception as exc:
                st.sidebar.error(f"Pipeline error: {exc}")

    return {"page": page, "season": season, "zones": zones, "date_range": date_range}


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_dashboard():
    st.title("📊 Dashboard")
    st.caption("Real-time satellite-derived irrigation intelligence · Sentinel-2 & Sentinel-1")

    # KPI cards
    render_kpi_cards()
    st.divider()

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.subheader("Crop Distribution")
        render_crop_distribution_chart()

    with col_right:
        st.subheader("Moisture Stress")
        render_stress_donut()

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("7-Day Weather Forecast")
        render_weather_forecast_chart()
    with col_b:
        st.subheader("Model Accuracy")
        render_model_accuracy_bars()


def page_maps():
    st.title("🗺️ Maps")
    st.caption("Interactive satellite-derived raster maps")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌾 Crop Classification",
        "💧 Water Deficit",
        "🌡 Moisture Stress",
        "📍 Advisory Zones",
    ])

    with tab1:
        render_crop_map()
    with tab2:
        render_deficit_map()
    with tab3:
        render_stress_map()
    with tab4:
        render_zone_advisory_map()


def page_advisory():
    st.title("💧 Irrigation Advisory")
    st.caption("AI-generated zone-level irrigation recommendations")

    col1, col2, col3 = st.columns(3)
    col1.metric("Critical Zones", "4", delta="↑ 1 since yesterday", delta_color="inverse")
    col2.metric("Moderate Zones", "2")
    col3.metric("Adequate Zones", "0")

    st.divider()
    render_advisory_table()

    st.download_button(
        "⬇  Download Advisory Report (CSV)",
        data="zone_id,priority,water_deficit_mm\nZ1,critical,52.3\n",
        file_name="irrigation_advisory.csv",
        mime="text/csv",
    )


def page_analytics():
    st.title("📈 Analytics")
    st.caption("Historical trends and model performance")

    st.info(
        "Connect to your PostgreSQL time-series database to enable "
        "historical NDVI, deficit, and advisory trend charts.",
        icon="ℹ️",
    )

    with st.expander("Model Performance Summary"):
        render_model_accuracy_bars()

    with st.expander("Feature Importance (SHAP)"):
        st.caption(
            "Run the training pipeline to generate SHAP values. "
            "Results will appear here automatically."
        )
        shap_path = settings.reports_dir / "shap_feature_importance.csv"
        if shap_path.exists():
            import pandas as pd
            import plotly.express as px
            df = pd.read_csv(shap_path).head(15)
            fig = px.bar(
                df, x="mean_abs_shap", y="feature", orientation="h",
                color="mean_abs_shap", color_continuous_scale="Greens",
                title="Top 15 Features by Mean |SHAP|",
            )
            fig.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("SHAP report not found. Train models first.")


def page_settings():
    st.title("⚙️ Settings")

    with st.expander("Satellite Configuration", expanded=True):
        st.text_input("Copernicus Hub Username",
                      value=settings.satellite.sentinel_user or "",
                      type="password")
        st.number_input("Cloud Cover Threshold (%)",
                        value=settings.satellite.cloud_cover_threshold,
                        min_value=0.0, max_value=100.0)
        st.number_input("Spatial Resolution (m)",
                        value=settings.satellite.spatial_resolution_m)

    with st.expander("Irrigation Thresholds"):
        st.slider("CWSI Critical Threshold",
                  0.0, 1.0, settings.irrigation.cwsi_critical)
        st.slider("CWSI Moderate Threshold",
                  0.0, 1.0, settings.irrigation.cwsi_moderate)
        st.number_input("Water Deficit Critical (mm)",
                        value=settings.irrigation.water_deficit_critical_mm)
        st.number_input("Water Deficit Moderate (mm)",
                        value=settings.irrigation.water_deficit_moderate_mm)

    with st.expander("Database"):
        st.text_input("PostgreSQL DSN", value=settings.db.postgres_dsn, type="password")
        st.text_input("Redis URL", value=settings.db.redis_url)

    if st.button("Save Settings", type="primary"):
        st.success("Settings saved (update your .env file to persist).")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ctx = render_sidebar()
    page = ctx["page"]

    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "🗺️ Maps":
        page_maps()
    elif page == "💧 Advisory":
        page_advisory()
    elif page == "📈 Analytics":
        page_analytics()
    elif page == "⚙️ Settings":
        page_settings()


if __name__ == "__main__":
    main()
