"""
charts/chart_builder.py
All chart rendering functions using Plotly.
Returns plotly figures — UI layer calls st.plotly_chart().
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from core.logger import get_logger

logger = get_logger(__name__)


def trend_with_anomalies(
    ts: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = None,
) -> go.Figure:
    """
    KPI trend line with anomaly scatter overlay.
    Replaces matplotlib chart from app.py v0.1.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ts[date_col], y=ts[value_col],
        mode="lines",
        name=value_col,
        line=dict(color="#3B82F6", width=2),
    ))

    if "rolling_mean" in ts.columns:
        fig.add_trace(go.Scatter(
            x=ts[date_col], y=ts["rolling_mean"],
            mode="lines",
            name="Rolling Mean",
            line=dict(color="#94A3B8", width=1, dash="dot"),
        ))

    if "upper_band" in ts.columns and "lower_band" in ts.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([ts[date_col], ts[date_col][::-1]]),
            y=pd.concat([ts["upper_band"], ts["lower_band"][::-1]]),
            fill="toself",
            fillcolor="rgba(148,163,184,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±2σ Band",
        ))

    if "anomaly" in ts.columns:
        anom = ts[ts["anomaly"] == True]
        if not anom.empty:
            fig.add_trace(go.Scatter(
                x=anom[date_col], y=anom[value_col],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#EF4444", size=9, symbol="circle-open", line=dict(width=2)),
            ))

    if "trend" in ts.columns:
        fig.add_trace(go.Scatter(
            x=ts[date_col], y=ts["trend"],
            mode="lines",
            name="Trend",
            line=dict(color="#F59E0B", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title=title or f"{value_col} over time",
        xaxis_title="Date",
        yaxis_title=value_col,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def driver_bar_chart(
    drivers: pd.DataFrame,
    title: str = "Driver Attribution",
    n: int = 10,
) -> go.Figure:
    """Horizontal bar chart of top positive and negative drivers."""
    if drivers.empty:
        return go.Figure()

    top_pos = drivers.nlargest(n // 2, "delta")
    top_neg = drivers.nsmallest(n // 2, "delta")
    plot_df = pd.concat([top_neg, top_pos]).drop_duplicates()
    plot_df = plot_df.sort_values("delta")
    plot_df["label"] = plot_df["dimension"] + ": " + plot_df["value"].astype(str)
    plot_df["color"] = plot_df["delta"].apply(lambda x: "#EF4444" if x < 0 else "#22C55E")

    fig = go.Figure(go.Bar(
        x=plot_df["delta"],
        y=plot_df["label"],
        orientation="h",
        marker_color=plot_df["color"],
        text=plot_df["delta"].apply(lambda x: f"{x:+,.0f}"),
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Delta",
        height=max(300, len(plot_df) * 32 + 80),
        margin=dict(l=20, r=60, t=50, b=40),
    )
    return fig


def funnel_chart(funnel_df: pd.DataFrame, title: str = "Funnel") -> go.Figure:
    """Funnel chart using plotly funnel trace."""
    if funnel_df.empty:
        return go.Figure()

    fig = go.Figure(go.Funnel(
        y=funnel_df["stage"],
        x=funnel_df["users"],
        textinfo="value+percent previous",
        marker=dict(color="#3B82F6"),
        connector=dict(line=dict(color="#CBD5E1", width=1)),
    ))
    fig.update_layout(title=title, height=420, margin=dict(l=20, r=20, t=50, b=30))
    return fig


def cohort_heatmap(matrix: pd.DataFrame, title: str = "Cohort Retention") -> go.Figure:
    """Retention matrix heatmap."""
    if matrix.empty:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=[str(p) for p in matrix.index],
        colorscale="Blues",
        text=matrix.values.round(1),
        texttemplate="%{text}%",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        height=max(300, len(matrix) * 36 + 100),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title="Period",
        yaxis_title="Cohort",
    )
    return fig


def contribution_bar(
    df: pd.DataFrame,
    dim_col: str,
    title: str = "Contribution by Segment",
) -> go.Figure:
    """Stacked or sorted bar of segment share."""
    if df.empty:
        return go.Figure()

    fig = px.bar(
        df.head(15),
        x=dim_col,
        y="share_pct",
        color="share_pct",
        color_continuous_scale="Blues",
        text="share_pct",
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=380,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=60),
        yaxis_title="Share %",
        coloraxis_showscale=False,
    )
    return fig


def kpi_comparison_bar(comparisons: list[dict], title: str = "Period Comparisons") -> go.Figure:
    """Bar chart for DoD / WoW / MoM comparisons."""
    if not comparisons:
        return go.Figure()

    labels = [c["comparison"] for c in comparisons]
    pcts = [c["pct_change"] for c in comparisons]
    colors = ["#22C55E" if p >= 0 else "#EF4444" for p in pcts]

    fig = go.Figure(go.Bar(
        x=labels, y=pcts,
        marker_color=colors,
        text=[f"{p:+.1f}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        yaxis_title="% Change",
        height=320,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig
