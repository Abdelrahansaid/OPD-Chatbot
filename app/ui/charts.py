"""
app/ui/charts.py — Enhanced Plotly chart generation for Andalusia OPD Chatbot.
Phase 5-6: Added 10+ chart types with consistent theming and color-coding.
"""
from typing import Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Consistent color palette ──────────────────────────────────
C = {
    "good":     "#22c55e",
    "warning":  "#f59e0b",
    "critical": "#ef4444",
    "blue":     "#3b82f6",
    "purple":   "#8b5cf6",
    "cyan":     "#06b6d4",
    "actual":   "#60a5fa",
    "target":   "#a78bfa",
    "bg":       "rgba(0,0,0,0)",
    "grid":     "rgba(255,255,255,0.06)",
    "text":     "#94a3b8",
    "title":    "#e2e8f0",
}

PALETTE = [C["blue"], C["purple"], C["cyan"], C["good"], C["warning"], C["critical"]]


def _base_layout(title: str, height: int = 320) -> dict:
    """Shared layout for all charts."""
    return dict(
        title=dict(text=title, font=dict(color=C["title"], size=14), x=0.01),
        plot_bgcolor=C["bg"],
        paper_bgcolor=C["bg"],
        font=dict(color=C["text"], size=12),
        margin=dict(l=12, r=12, t=44, b=12),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color=C["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
    )


def _status_color(value: float, good_threshold: float = 90, warn_threshold: float = 70) -> str:
    if value >= good_threshold:
        return C["good"]
    if value >= warn_threshold:
        return C["warning"]
    return C["critical"]


# ─────────────────────────────────────────────────────────────
def generate_chart(intent: str, data: Any) -> Optional[go.Figure]:
    """Main entry point. Returns a Plotly Figure or None."""
    try:
        if data is None:
            return None
        if isinstance(data, dict) and data.get("chart_data"):
            return _generic_chart_data(data)
        handler = _HANDLERS.get(intent)
        if handler:
            return handler(data)
    except Exception:
        pass
    return None


def _generic_chart_data(data: dict):
    rows = data.get("chart_data")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    x_col = data.get("x") or next((c for c in df.columns if c != data.get("y")), None)
    y_col = data.get("y") or data.get("metric")
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return None

    title = f"{data.get('metric', y_col)} by {data.get('group_by', x_col)}"
    chart_type = str(data.get("chart_type") or "bar").lower()
    if chart_type == "line":
        fig = go.Figure(go.Scatter(
            x=df[x_col], y=df[y_col], mode="lines+markers+text",
            marker=dict(size=8, color=C["blue"]),
            line=dict(color=C["blue"], width=2.5),
            text=[f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" for v in df[y_col]],
            textposition="top center",
        ))
    else:
        sort_df = df.sort_values(y_col, ascending=True) if len(df) > 3 else df
        horizontal = chart_type in {"horizontal_bar", "hbar"} or data.get("group_by") == "doctor"
        if horizontal:
            fig = go.Figure(go.Bar(
                x=sort_df[y_col], y=sort_df[x_col], orientation="h",
                marker_color=C["blue"],
                text=[f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" for v in sort_df[y_col]],
                textposition="outside",
            ))
        else:
            fig = go.Figure(go.Bar(
                x=sort_df[x_col], y=sort_df[y_col],
                marker_color=PALETTE[:len(sort_df)],
                text=[f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" for v in sort_df[y_col]],
                textposition="outside",
            ))
    fig.update_layout(**_base_layout(title, height=max(300, min(620, len(df) * 34 + 180))))
    return fig


# ── Revenue vs Target ─────────────────────────────────────────
def _revenue_vs_target(data: dict):
    if not isinstance(data, dict) or "total_revenue" not in data:
        return None
    actual = data.get("total_revenue", 0)
    target = data.get("target_revenue", 0)
    ach = data.get("revenue_achievement_pct", 0) or 0
    color = _status_color(ach)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Actual Revenue"], y=[actual],
        marker_color=color, name="Actual",
        text=[f"{ach:.1f}%"], textposition="outside",
        textfont=dict(color=C["title"], size=14, family="monospace"),
    ))
    fig.add_trace(go.Bar(
        x=["Target Revenue"], y=[target],
        marker_color=C["target"], name="Target",
        opacity=0.7,
    ))
    fig.add_hline(y=target, line_dash="dash", line_color=C["warning"],
                  annotation_text="Target", annotation_font_color=C["warning"])
    fig.update_layout(**_base_layout("Revenue vs Target"))
    fig.update_yaxes(tickformat=",.0f")
    return fig


# ── Executive Summary ─────────────────────────────────────────
def _executive_summary(data: dict):
    if not isinstance(data, dict) or "revenue" not in data:
        return None
    rev = data.get("revenue", {})
    qual = data.get("quality", {})
    ops = data.get("operational_issues", {})
    if not isinstance(rev, dict):
        return None

    indicators = [
        ("Revenue Ach%", rev.get("achievement_pct", 0), 90, 70, "%"),
        ("COE%", qual.get("avg_coe_pct", 0), 80, 60, "%"),
        ("PMS%", qual.get("avg_pms_pct", 0), 80, 60, "%"),
        ("No-Show%", ops.get("avg_noshow_pct", 0), 15, 25, "%"),
    ]

    fig = go.Figure()
    for i, (label, val, good_t, warn_t, suffix) in enumerate(indicators):
        if label == "No-Show%":
            color = C["good"] if val <= good_t else (C["warning"] if val <= warn_t else C["critical"])
        else:
            color = _status_color(val, good_t, warn_t)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(suffix=suffix, font=dict(size=18, color=color)),
            title=dict(text=label, font=dict(size=11, color=C["text"])),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=C["text"]),
                bar=dict(color=color),
                bgcolor=C["bg"],
                bordercolor="rgba(255,255,255,0.1)",
                steps=[
                    dict(range=[0, warn_t], color="rgba(239,68,68,0.1)"),
                    dict(range=[warn_t, good_t], color="rgba(245,158,11,0.1)"),
                    dict(range=[good_t, 100], color="rgba(34,197,94,0.1)"),
                ],
            ),
            domain=dict(row=0, column=i),
        ))

    fig.update_layout(
        grid=dict(rows=1, columns=4),
        height=240,
        plot_bgcolor=C["bg"],
        paper_bgcolor=C["bg"],
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="KPI Scorecard", font=dict(color=C["title"], size=14), x=0.01),
    )
    return fig


# ── PMS Ranking ───────────────────────────────────────────────
def _pms_ranking(data):
    rows = data if isinstance(data, list) else (data.get("ranking") if isinstance(data, dict) else None)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    col = next((c for c in ["Avg PMS Score (%)", "avg_pms_pct", "pms_score"] if c in df.columns), None)
    name_col = next((c for c in ["Doctor Name", "doctor_name"] if c in df.columns), None)
    if not col or not name_col:
        return None
    df = df.sort_values(col, ascending=True).tail(11)
    colors = [_status_color(v, 80, 60) for v in df[col]]
    fig = go.Figure(go.Bar(
        x=df[col], y=df[name_col], orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df[col]], textposition="outside",
        textfont=dict(color=C["title"], size=11),
    ))
    fig.add_vline(x=80, line_dash="dash", line_color=C["good"], annotation_text="Good (80%)")
    fig.add_vline(x=60, line_dash="dot", line_color=C["critical"], annotation_text="Critical (60%)")
    fig.update_layout(**_base_layout("Doctor PMS Score Ranking", height=max(280, len(df) * 28)))
    fig.update_xaxes(range=[0, 110])
    return fig


# ── Leakage Analysis ──────────────────────────────────────────
def _leakage_analysis(data: dict):
    if not isinstance(data, dict):
        return None
    rows = data.get("top_leakage_by_doctor")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "Doctor Name" not in df.columns or "leakage_losses" not in df.columns:
        return None
    df = df.sort_values("leakage_losses", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["leakage_losses"], y=df["Doctor Name"], orientation="h",
        marker=dict(
            color=df["leakage_losses"],
            colorscale=[[0, C["warning"]], [1, C["critical"]]],
            showscale=False,
        ),
        text=[f"{v:,.0f}" for v in df["leakage_losses"]], textposition="outside",
        textfont=dict(color=C["title"], size=11),
    ))
    fig.update_layout(**_base_layout(f"Top Leakage by Doctor  (Total: {data.get('total_leakage_losses', 0):,.0f})", height=max(280, len(df) * 30)))
    return fig


# ── COE Compliance ────────────────────────────────────────────
def _coe_compliance(data: dict):
    if not isinstance(data, dict):
        return None
    rows = data.get("by_doctor")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    col = next((c for c in ["Actual COE Compliance %", "avg_coe_pct"] if c in df.columns), None)
    name_col = next((c for c in ["Doctor Name", "doctor_name"] if c in df.columns), None)
    if not col or not name_col:
        return None
    df = df.sort_values(col, ascending=True)
    colors = [_status_color(v, 80, 60) for v in df[col]]
    fig = go.Figure(go.Bar(
        x=df[col], y=df[name_col], orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df[col]], textposition="outside",
        textfont=dict(color=C["title"], size=11),
    ))
    fig.add_vline(x=80, line_dash="dash", line_color=C["good"], annotation_text="Target 80%")
    fig.update_layout(**_base_layout("COE Compliance by Doctor", height=max(280, len(df) * 28)))
    fig.update_xaxes(range=[0, 115])
    return fig


# ── No-Show Analysis ──────────────────────────────────────────
def _noshow_analysis(data: dict):
    if not isinstance(data, dict):
        return None
    rows = data.get("by_doctor")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    col = next((c for c in ["No-Show %", "avg_noshow_pct"] if c in df.columns), None)
    name_col = next((c for c in ["Doctor Name"] if c in df.columns), None)
    if not col or not name_col:
        return None
    df = df.sort_values(col, ascending=False)
    colors = [C["critical"] if v > 25 else (C["warning"] if v > 15 else C["good"]) for v in df[col]]
    fig = go.Figure(go.Bar(
        x=df[name_col], y=df[col],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df[col]], textposition="outside",
        textfont=dict(color=C["title"], size=11),
    ))
    fig.add_hline(y=25, line_dash="dash", line_color=C["critical"], annotation_text="Critical (25%)")
    fig.add_hline(y=15, line_dash="dot", line_color=C["warning"], annotation_text="Warning (15%)")
    fig.update_layout(**_base_layout("No-Show Rate by Doctor"))
    fig.update_xaxes(tickangle=30)
    return fig


# ── Branch Comparison ─────────────────────────────────────────
def _branch_comparison(data):
    rows = data if isinstance(data, list) else (data.get("comparison_data") if isinstance(data, dict) else None)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "BU" not in df.columns:
        return None
    fig = go.Figure()
    metrics = [
        ("revenue_achievement_pct", "Revenue Ach%", C["blue"]),
        ("avg_coe_pct", "COE%", C["good"]),
        ("avg_pms_pct", "PMS%", C["purple"]),
    ]
    for col, label, color in metrics:
        if col in df.columns:
            fig.add_trace(go.Bar(name=label, x=df["BU"], y=df[col], marker_color=color))
    fig.update_layout(**_base_layout("Branch KPI Comparison"), barmode="group")
    fig.add_hline(y=80, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    return fig


# ── Doctor Multi-BU ───────────────────────────────────────────
def _doctor_multi_bu(data: dict):
    if not isinstance(data, dict):
        return None
    rows = data.get("per_bu_performance")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "bu" not in df.columns:
        return None
    fig = go.Figure()
    cols_map = [
        ("total_revenue", "Revenue", C["blue"]),
        ("avg_pms_score_pct", "PMS%", C["purple"]),
        ("avg_coe_compliance_pct", "COE%", C["good"]),
    ]
    agg_name = data.get("overall_aggregated", {}).get("doctor_name", "Doctor")

    # Revenue bars (left y-axis), percentages (right y-axis)
    if "total_revenue" in df.columns:
        fig.add_trace(go.Bar(
            name="Revenue", x=df["bu"], y=df["total_revenue"],
            marker_color=C["blue"], opacity=0.85, yaxis="y1",
            text=[f"{v:,.0f}" for v in df["total_revenue"]], textposition="outside",
        ))
    if "avg_pms_score_pct" in df.columns:
        fig.add_trace(go.Scatter(
            name="PMS%", x=df["bu"], y=df["avg_pms_score_pct"],
            mode="lines+markers+text", marker=dict(size=10, color=C["purple"]),
            line=dict(color=C["purple"], width=2), yaxis="y2",
            text=[f"{v:.0f}%" for v in df["avg_pms_score_pct"]], textposition="top center",
        ))

    fig.update_layout(
        **_base_layout(f"{agg_name} — Performance by Branch"),
        yaxis=dict(title="Revenue", gridcolor=C["grid"]),
        yaxis2=dict(title="% Score", overlaying="y", side="right", range=[0, 120], gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── Volume Analysis ───────────────────────────────────────────
def _volume_analysis(data: dict):
    if not isinstance(data, dict):
        return None
    rows = data.get("by_doctor")
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("cases", ascending=False).head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Cases", x=df["Doctor Name"], y=df["cases"], marker_color=C["blue"]))
    if "services" in df.columns:
        fig.add_trace(go.Bar(name="Services", x=df["Doctor Name"], y=df["services"], marker_color=C["cyan"]))
    fig.update_layout(**_base_layout("Case & Service Volume by Doctor"), barmode="group")
    fig.update_xaxes(tickangle=35)
    return fig


# ── Monthly Trend ─────────────────────────────────────────────
def _monthly_trend(data):
    rows = data if isinstance(data, list) else None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    kpi_col = next((c for c in df.columns if c not in ["Year", "Month No", "Month"]), None)
    if not kpi_col or "Month" not in df.columns:
        return None
    fig = go.Figure(go.Scatter(
        x=df["Month"], y=df[kpi_col],
        mode="lines+markers+text",
        marker=dict(size=8, color=C["blue"]),
        line=dict(color=C["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        text=[f"{v:,.0f}" for v in df[kpi_col]], textposition="top center",
        textfont=dict(size=10, color=C["title"]),
    ))
    fig.update_layout(**_base_layout(f"Monthly Trend — {kpi_col}"))
    return fig


# ── Booking Utilization ───────────────────────────────────────
def _booking_utilization(data: dict):
    if not isinstance(data, dict):
        return None
    rows = data.get("by_doctor")
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("utilization_pct", ascending=False).head(10)
    if "utilization_pct" not in df.columns:
        return None
    colors = [_status_color(v, 80, 60) for v in df["utilization_pct"]]
    fig = go.Figure(go.Bar(
        x=df["Doctor Name"], y=df["utilization_pct"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df["utilization_pct"]], textposition="outside",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color=C["good"], annotation_text="Target 80%")
    fig.update_layout(**_base_layout("Booking Utilization by Doctor"))
    fig.update_xaxes(tickangle=35)
    return fig


# ── Best / Worst Doctor (Ranking) ─────────────────────────────
def _ranking_chart(data: dict):
    if not isinstance(data, dict):
        return None
    rows = data.get("ranking") or data.get("full_ranking")
    if not rows:
        return None
    df = pd.DataFrame(rows[:10])
    name_col = next((c for c in ["Doctor Name", "doctor_name"] if c in df.columns), None)
    score_col = next((c for c in ["excellence_score", "score", "avg_pms_pct"] if c in df.columns), None)
    if not name_col or not score_col:
        return None
    df = df.sort_values(score_col, ascending=True)
    colors = [_status_color(v, 80, 65) for v in df[score_col]]
    fig = go.Figure(go.Bar(
        x=df[score_col], y=df[name_col], orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in df[score_col]], textposition="outside",
        textfont=dict(color=C["title"], size=11),
    ))
    fig.update_layout(**_base_layout("Composite Excellence Score Ranking", height=max(280, len(df) * 30)))
    return fig


# ── Operational Risks ─────────────────────────────────────────
def _operational_risks(data: dict):
    if not isinstance(data, dict):
        return None
    risks = data.get("top_3_risks")
    if not risks:
        return None
    labels = [r["risk"] for r in risks]
    severities = [r["severity"] for r in risks]
    colors = [C["critical"] if "Critical" in s else C["warning"] for s in severities]
    # Simple horizontal bar with rank values
    fig = go.Figure(go.Bar(
        y=labels, x=[3, 2, 1][:len(risks)],
        orientation="h",
        marker_color=colors,
        text=[r["value"] for r in risks], textposition="inside",
        textfont=dict(color="#fff", size=11),
    ))
    fig.update_layout(
        **_base_layout("Top Operational Risks"),
        xaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)"),
        height=220,
    )
    return fig


# ── Top Doctors by Revenue ────────────────────────────────────
def _top_doctors_revenue(data):
    rows = data if isinstance(data, list) else None
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "Doctor Name" not in df.columns or "Total Revenue" not in df.columns:
        return None
    df = df.sort_values("Total Revenue", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["Total Revenue"], y=df["Doctor Name"], orientation="h",
        marker_color=C["blue"],
        text=[f"{v:,.0f}" for v in df["Total Revenue"]], textposition="outside",
        textfont=dict(color=C["title"], size=11),
    ))
    fig.update_layout(**_base_layout("Top Doctors by Revenue", height=max(240, len(df) * 32)))
    return fig


# ── Handler dispatch map ──────────────────────────────────────
_HANDLERS = {
    "revenue_vs_target":         _revenue_vs_target,
    "executive_summary":         _executive_summary,
    "pms_ranking":               _pms_ranking,
    "leakage_analysis":          _leakage_analysis,
    "coe_compliance":            _coe_compliance,
    "noshow_analysis":           _noshow_analysis,
    "branch_comparison":         _branch_comparison,
    "doctor_multi_bu":           _doctor_multi_bu,
    "volume_analysis":           _volume_analysis,
    "monthly_trend":             _monthly_trend,
    "booking_utilization":       _booking_utilization,
    "best_doctor":               _ranking_chart,
    "worst_doctor":              _ranking_chart,
    "most_problematic_doctor":   _ranking_chart,
    "dynamic_ranking":           _ranking_chart,
    "top_operational_risks":     _operational_risks,
    "top_doctors_revenue":       _top_doctors_revenue,
    "revenue_breakdown":         _revenue_vs_target,   # reuse same chart
}
