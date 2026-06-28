"""
dashboard/visualization.py
─────────────────────────────────────────────────────────────────────────────
Plotly-based chart and KPI widgets rendered inside the Streamlit dashboard.
Each public function calls st.plotly_chart() or st.metric() directly.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Colour palette (matches dashboard CSS) ─────────────────────────────────────
GREEN   = "#1D9E75"
AMBER   = "#EF9F27"
RED     = "#D85A30"
BLUE    = "#378ADD"
LIGHT_G = "#EAF3DE"

CROP_COLORS = {
    "Paddy/Rice":    "#639922",
    "Maize":         "#BA7517",
    "Soybean":       GREEN,
    "Sugarcane":     "#D85A30",
    "Groundnut":     BLUE,
    "Cotton":        "#8E6BBF",
    "Other/Fallow":  "#AAAAAA",
}

STRESS_COLORS = {
    "No stress":      GREEN,
    "Mild":           "#90C25A",
    "Moderate":       AMBER,
    "Severe":         RED,
}


# ── KPI Cards ─────────────────────────────────────────────────────────────────

def render_kpi_cards():
    """Four top-level KPI metrics in a single row."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Command Area",       "12,840 ha",  "↑ 3.2% monitored")
    c2.metric("Irrigation Need",    "38.6 mm",    "Moderate deficit",  delta_color="inverse")
    c3.metric("Crop Health (NDVI)", "0.71",       "↑ +0.03 this week")
    c4.metric("Stress Alerts",      "247 pixels", "4 critical zones",  delta_color="inverse")


# ── Crop Distribution Bar Chart ────────────────────────────────────────────────

def render_crop_distribution_chart():
    """Horizontal bar chart of crop-wise area."""
    data = {
        "Crop":     ["Paddy/Rice", "Maize", "Soybean", "Groundnut", "Sugarcane", "Cotton"],
        "Area (ha)":[4210,          2940,    1860,       2600,        1230,         0],
        "Stage":    ["Tillering", "Tasseling", "Flowering",
                     "Pod Formation", "Grand Growth", "—"],
    }
    df = pd.DataFrame(data).sort_values("Area (ha)")
    df["color"] = df["Crop"].map(CROP_COLORS)

    fig = go.Figure(go.Bar(
        x=df["Area (ha)"],
        y=df["Crop"],
        orientation="h",
        marker_color=df["color"],
        text=df["Area (ha)"].apply(lambda v: f"{v:,} ha"),
        textposition="outside",
        customdata=df["Stage"],
        hovertemplate="<b>%{y}</b><br>Area: %{x:,} ha<br>Stage: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Area (ha)",
        margin=dict(l=0, r=40, t=10, b=30),
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Stress Donut ───────────────────────────────────────────────────────────────

def render_stress_donut():
    """Donut chart of moisture-stress class distribution."""
    labels  = ["No stress", "Mild", "Moderate", "Severe"]
    values  = [41, 28, 22, 9]
    colors  = [STRESS_COLORS[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value}% of area<extra></extra>",
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text="CWSI<br>0.61", x=0.5, y=0.5,
                          font_size=14, showarrow=False)],
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Weather Forecast Chart ─────────────────────────────────────────────────────

def render_weather_forecast_chart():
    """Combined bar + line chart for 7-day weather forecast."""
    today = date.today()
    days  = [(today + timedelta(days=i)).strftime("%a %d") for i in range(7)]

    temp_max  = [36, 34, 31, 30, 35, 37, 36]
    temp_min  = [27, 26, 25, 24, 27, 28, 27]
    precip    = [0,  2,  14, 8,  0,  0,  0]

    fig = go.Figure()

    # Rainfall bars
    fig.add_trace(go.Bar(
        x=days, y=precip,
        name="Rainfall (mm)",
        marker_color=BLUE,
        opacity=0.65,
        yaxis="y2",
    ))

    # Temp max line
    fig.add_trace(go.Scatter(
        x=days, y=temp_max,
        name="Max °C",
        line=dict(color=RED, width=2),
        mode="lines+markers",
    ))

    # Temp min line
    fig.add_trace(go.Scatter(
        x=days, y=temp_min,
        name="Min °C",
        line=dict(color=BLUE, width=2, dash="dot"),
        mode="lines+markers",
    ))

    fig.update_layout(
        yaxis=dict(title="Temperature (°C)", range=[15, 45]),
        yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right", range=[0, 50]),
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=0, r=0, t=10, b=40),
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    st.plotly_chart(fig, use_container_width=True)


# ── Model Accuracy Bars ────────────────────────────────────────────────────────

def render_model_accuracy_bars():
    """Horizontal bar chart of ML model accuracies."""
    models = [
        ("Crop Classification (RF)",    92.4),
        ("Moisture Stress (CNN)",        89.3),
        ("Water Deficit (XGBoost)",      87.1),
        ("Irrigation Advisory (Rules)",  94.0),
    ]
    df = pd.DataFrame(models, columns=["Model", "Accuracy (%)"])

    fig = go.Figure(go.Bar(
        x=df["Accuracy (%)"],
        y=df["Model"],
        orientation="h",
        marker=dict(
            color=df["Accuracy (%)"],
            colorscale=[[0, AMBER], [0.5, GREEN], [1, GREEN]],
            cmin=80, cmax=100,
        ),
        text=df["Accuracy (%)"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(range=[70, 100], title="Accuracy / F1-Macro (%)"),
        margin=dict(l=0, r=40, t=10, b=20),
        height=230,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Advisory Table ─────────────────────────────────────────────────────────────

def render_advisory_table():
    """Styled DataFrame of zone-level irrigation advisories."""
    data = {
        "Zone":           ["Zone 3", "Zone 6", "Zone 1", "Zone 5", "Zone 2", "Zone 4"],
        "Crop":           ["Maize", "Sugarcane", "Paddy", "Maize", "Soybean", "Groundnut"],
        "Priority":       ["🔴 Critical", "🔴 Critical", "🟡 Moderate",
                           "🟡 Moderate", "🟢 Adequate", "🟢 Adequate"],
        "Deficit (mm)":   [54.2, 51.8, 32.1, 28.4, 12.3, 9.7],
        "CWSI":           [0.82, 0.79, 0.61, 0.57, 0.28, 0.22],
        "SM (%)":         [17.3, 18.1, 34.2, 37.6, 52.4, 58.9],
        "Irrigate By":    ["Today", "Today", "In 3 days", "In 3 days", "—", "—"],
    }
    df = pd.DataFrame(data)

    def _color_priority(val: str) -> str:
        if "Critical" in val:
            return "background-color: #FCEBEB; color: #A32D2D"
        if "Moderate" in val:
            return "background-color: #FAEEDA; color: #854F0B"
        return "background-color: #EAF3DE; color: #3B6D11"

    styled = (
        df.style
        .applymap(_color_priority, subset=["Priority"])
        .format({"Deficit (mm)": "{:.1f}", "CWSI": "{:.2f}", "SM (%)": "{:.1f}"})
        .bar(subset=["Deficit (mm)"], color="#EF9F2780", vmin=0, vmax=60)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── NDVI Time-Series ───────────────────────────────────────────────────────────

def render_ndvi_timeseries(df: pd.DataFrame | None = None):
    """
    NDVI time-series line chart.

    Parameters
    ----------
    df : DataFrame with columns [date, ndvi_mean, ndvi_p25, ndvi_p75]
         If None a synthetic example is rendered.
    """
    if df is None:
        dates = pd.date_range("2026-04-01", periods=12, freq="5D")
        np.random.seed(42)
        ndvi_mean = 0.3 + np.cumsum(np.random.uniform(-0.02, 0.05, 12))
        df = pd.DataFrame({
            "date":      dates,
            "ndvi_mean": np.clip(ndvi_mean, 0.1, 0.9),
            "ndvi_p25":  np.clip(ndvi_mean - 0.08, 0.05, 0.9),
            "ndvi_p75":  np.clip(ndvi_mean + 0.08, 0.1,  0.95),
        })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([df["date"], df["date"][::-1]]),
        y=pd.concat([df["ndvi_p75"], df["ndvi_p25"][::-1]]),
        fill="toself",
        fillcolor="rgba(29,158,117,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IQR",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["ndvi_mean"],
        mode="lines+markers",
        name="Mean NDVI",
        line=dict(color=GREEN, width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="NDVI",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=0, r=0, t=10, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    st.plotly_chart(fig, use_container_width=True)


# ── Deficit Heatmap ────────────────────────────────────────────────────────────

def render_deficit_heatmap(matrix: np.ndarray | None = None):
    """
    Pseudo-colour heatmap of per-pixel water deficit values.

    Parameters
    ----------
    matrix : 2-D float array of deficit values in mm.
              If None, a synthetic 50×80 example is rendered.
    """
    if matrix is None:
        np.random.seed(7)
        matrix = np.random.uniform(0, 80, (50, 80)).astype(np.float32)
        # Add spatial structure
        matrix[:25, 40:] += 30
        matrix[25:, :40] -= 10
        matrix = np.clip(matrix, 0, 100)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        colorscale=[
            [0.0,  "#5DCAA5"],
            [0.35, "#EF9F27"],
            [0.70, "#D85A30"],
            [1.0,  "#7B0D0D"],
        ],
        zmin=0,
        zmax=80,
        colorbar=dict(
            title="Deficit (mm)",
            thickness=12,
            len=0.7,
        ),
        hovertemplate="Row %{y}, Col %{x}<br>Deficit: %{z:.1f} mm<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=0, r=60, t=10, b=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
