# app/core/tools.py
import json
import logging
import re
from typing import Any, Dict
from typing import Optional

from langchain.tools import tool

_analytics = None
_kb = None

logger = logging.getLogger(__name__)


def set_analytics_kb(analytics, kb):
    global _analytics, _kb
    _analytics = analytics
    _kb = kb


def _analytics_ready() -> bool:
    return _analytics is not None and hasattr(_analytics, "df")


def _json_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _as_percent_if_needed(series, column: str):
    value = series.mean()
    if column in {"Doctor PMS %", "Actual COE Compliance %", "No-Show %"}:
        value = value * 100
    return round(float(value), 2)


@tool
def get_doctors_by_bu(bu: str) -> str:
    """Return list of doctors working in a specific branch (bu = ASH, SMH, HJH)."""
    if not _analytics:
        return json.dumps({"error": "Analytics not initialized"})
    df = _analytics.filter(bu=bu)
    doctors = sorted(df['Doctor Name'].unique().tolist())
    return json.dumps({"bu": bu, "doctors": doctors, "count": len(doctors)}, ensure_ascii=False)


@tool
def get_executive_summary(
    year: Optional[int] = None,
    month: Optional[int] = None,
    bu: Optional[str] = None,
) -> str:
    """Return the executive summary for OPD performance."""
    data = _analytics.executive_summary(year=year, month=month, bu=bu)
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_revenue_achievement(
    bu: Optional[str] = None,
    year: Optional[int] = None,
    doctor: Optional[str] = None,
) -> str:
    """Return revenue achievement, actual revenue, target, and gap."""
    data = _analytics.revenue_gap_analysis(bu=bu, year=year, doctor=doctor)
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_pms_ranking(
    bu: Optional[str] = None,
    year: Optional[int] = None,
    top_n: int = 5,
) -> str:
    """Rank doctors by PMS score. Optional filters: bu, year, top_n."""
    data = _analytics.pms_ranking(bu=bu, year=year, top_n=top_n)
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_leakage_analysis(
    bu: Optional[str] = None,
    year: Optional[int] = None,
    doctor: Optional[str] = None,
) -> str:
    """Analyze leakage losses and leakage percentage."""
    data = _analytics.leakage_analysis(bu=bu, year=year, doctor=doctor)
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_branch_comparison(year: Optional[int] = None) -> str:
    """Compare ASH, SMH, and HJH branches for the selected year."""
    data = _analytics.branch_comparison(year=year)
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_coe_compliance(
    bu: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """Return COE compliance analysis by branch or year."""
    data = _analytics.coe_compliance_analysis(bu=bu, year=year)
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_noshow_analysis(
    bu: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """Analyze no-show percentage by branch or year."""
    data = _analytics.noshow_analysis(bu=bu, year=year)
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_doctor_multi_branch(
    doctor_name: str,
    year: Optional[int] = None,
) -> str:
    """Return one doctor's performance across branches."""
    data = _analytics.doctor_multi_branch_summary(doctor_name=doctor_name, year=year)
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_kpi_drivers(kpi_name: str) -> str:
    """Return KPI drivers from the knowledge base."""
    drivers = _kb.get_drivers(kpi_name)
    return json.dumps(drivers, default=str, ensure_ascii=False)


@tool
def get_kpi_playbook(kpi_name: str) -> str:
    """Return KPI investigation playbook from the knowledge base."""
    playbook = _kb.get_playbook(kpi_name)
    return json.dumps(playbook, default=str, ensure_ascii=False)


@tool
def rank_doctors_by_column(
    column: str,
    bu: Optional[str] = None,
    year: Optional[int] = None,
    top_n: int = 5,
) -> str:
    """Rank doctors by any numeric column available in the OPD data."""
    if not _analytics_ready():
        return _json_error("Analytics not initialized")

    valid_columns = [
        "Total Revenue",
        "No. Cases",
        "No. Services",
        "Credit Revenue",
        "Cash Revenue",
        "Total Leakage Revenue Losses",
        "Doctor PMS %",
        "Actual COE Compliance %",
        "No-Show %",
    ]
    if column not in valid_columns:
        return _json_error(f"Column '{column}' is not valid. Choose from: {valid_columns}")

    df = _analytics.filter(bu=bu, year=year)
    if df.empty:
        return _json_error("No data found for the selected filters.")

    group_cols = ["Doctor Name"] if bu else ["BU", "Doctor Name"]
    if column in {"Doctor PMS %", "Actual COE Compliance %", "No-Show %"}:
        result = df.groupby(group_cols)[column].mean().sort_values(ascending=False).head(top_n).reset_index()
        result[column] = (result[column] * 100).round(1)
    else:
        result = df.groupby(group_cols)[column].sum().sort_values(ascending=False).head(top_n).reset_index()
        result[column] = result[column].round(2)

    if not bu and "BU" in result.columns:
        result["Doctor Name"] = result["Doctor Name"].astype(str) + " (" + result["BU"].astype(str) + ")"
    result = result[["Doctor Name", column]].rename(columns={column: "Value"})
    return json.dumps(result.to_dict(orient="records"), ensure_ascii=False)


@tool
def get_summary_statistic(kpi: str, bu: Optional[str] = None, year: Optional[int] = None) -> str:
    """Return a total or average KPI statistic across the filtered OPD data."""
    if not _analytics_ready():
        return _json_error("Analytics not initialized")

    df = _analytics.filter(bu=bu, year=year)
    if df.empty:
        return _json_error("No data found for the selected filters.")

    kpi_map = {
        "pms": ("Doctor PMS %", "mean"),
        "doctor pms": ("Doctor PMS %", "mean"),
        "coe": ("Actual COE Compliance %", "mean"),
        "compliance": ("Actual COE Compliance %", "mean"),
        "noshow": ("No-Show %", "mean"),
        "no-show": ("No-Show %", "mean"),
        "revenue": ("Total Revenue", "sum"),
        "total revenue": ("Total Revenue", "sum"),
        "cases": ("No. Cases", "sum"),
        "services": ("No. Services", "sum"),
        "leakage": ("Total Leakage Revenue Losses", "sum"),
    }
    key = kpi.lower().strip()
    if key not in kpi_map:
        return _json_error(f"KPI '{kpi}' is not recognized.")

    column, agg = kpi_map[key]
    value = _as_percent_if_needed(df[column], column) if agg == "mean" else round(float(df[column].sum()), 2)
    return json.dumps({"kpi": kpi, "column": column, "value": value, "bu": bu, "year": year}, ensure_ascii=False)


@tool
def flexible_data_query(query: str) -> str:
    """Run a constrained aggregation query for clear OPD data questions."""
    if not _analytics_ready():
        return _json_error("Analytics not initialized")

    forbidden = ["specialty", "patient", "patient-level", "forecast", "fraud"]
    if any(term in query.lower() for term in forbidden):
        return _json_error("This flexible query is limited to safe OPD aggregations and cannot answer that topic.")

    df = _analytics.df.copy()
    bu = re.search(r"\b(ASH|SMH|HJH)\b", query, re.I)
    if bu:
        df = df[df["BU"] == bu.group(1).upper()]

    year = re.search(r"\b(202[3-5])\b", query)
    if year:
        df = df[df["Year"] == int(year.group(1))]

    doctor = re.search(r"(?:dr\.?|doctor)\s+([\w\u0600-\u06FF]+)", query, re.I)
    if doctor:
        df = df[df["Doctor Name"].str.contains(doctor.group(1), case=False, na=False)]

    kpi_mapping = {
        "revenue": "Total Revenue",
        "cases": "No. Cases",
        "services": "No. Services",
        "pms": "Doctor PMS %",
        "coe": "Actual COE Compliance %",
        "compliance": "Actual COE Compliance %",
        "leakage": "Total Leakage Revenue Losses",
        "noshow": "No-Show %",
        "no-show": "No-Show %",
    }
    kpi = next((col for key, col in kpi_mapping.items() if re.search(rf"\b{re.escape(key)}\b", query, re.I)), None)
    if not kpi:
        return _json_error("Could not determine KPI. Specify revenue, cases, services, PMS, COE, leakage, or no-show.")

    if df.empty:
        return _json_error("No data found for the selected filters.")

    agg_func = "mean" if kpi in {"Doctor PMS %", "Actual COE Compliance %", "No-Show %"} else "sum"
    grouped = df.groupby("Doctor Name")[kpi].agg(agg_func).sort_values(ascending=False).reset_index()
    if agg_func == "mean":
        grouped[kpi] = (grouped[kpi] * 100).round(1)
    else:
        grouped[kpi] = grouped[kpi].round(2)

    top_match = re.search(r"top\s*(\d+)", query, re.I)
    top_n = min(int(top_match.group(1)), 50) if top_match else 5
    return json.dumps(grouped.head(top_n).to_dict(orient="records"), ensure_ascii=False)


@tool
def calculate_custom_metric(formula: str, group_by: str = "doctor", filters: Optional[Dict[str, Any]] = None) -> str:
    """Calculate a safe custom metric using basic arithmetic over numeric OPD columns."""
    if not _analytics_ready():
        return _json_error("Analytics not initialized")

    if not re.match(r"^[a-zA-Z0-9\s+\-*/().%]+$", formula):
        return _json_error("Formula contains invalid characters. Only basic arithmetic is allowed.")

    df = _analytics.filter(**(filters or {}))
    if df.empty:
        return _json_error("No data found for the selected filters.")

    group_column = {"doctor": "Doctor Name", "bu": "BU", "branch": "BU"}.get(group_by.lower())
    if not group_column:
        return _json_error("group_by must be 'doctor' or 'bu'.")

    numeric_columns = [col for col in df.columns if col != group_column and hasattr(df[col], "dtype") and df[col].dtype.kind in "if"]
    used_columns = [col for col in sorted(numeric_columns, key=len, reverse=True) if re.search(rf"\b{re.escape(col)}\b", formula)]
    if not used_columns:
        return _json_error("Formula must include at least one numeric OPD column.")

    safe_expr = formula
    safe_names = {}
    for idx, col in enumerate(used_columns):
        safe_name = f"c{idx}"
        safe_names[col] = safe_name
        safe_expr = re.sub(rf"\b{re.escape(col)}\b", safe_name, safe_expr)

    leftovers = re.sub(r"\bc\d+\b", "", safe_expr)
    if re.search(r"[A-Za-z_]", leftovers):
        return _json_error("Formula contains unknown column names.")

    grouped = df.groupby(group_column)[used_columns].sum().reset_index()
    eval_df = grouped.rename(columns=safe_names)
    try:
        grouped["value"] = eval_df.eval(safe_expr, engine="python")
    except Exception as exc:
        return _json_error(f"Could not calculate formula: {exc}")

    result = grouped[[group_column, "value"]].replace([float("inf"), float("-inf")], None)
    result["value"] = result["value"].round(2)
    return json.dumps(result.to_dict(orient="records"), ensure_ascii=False, default=str)


@tool
def generate_chart_data(
    chart_type: str,
    metric: str,
    group_by: str,
    year: Optional[int] = None,
    bu: Optional[str] = None,
    top_n: int = 10,
) -> str:
    """Return structured data for a chart from OPD data. group_by: bu, doctor, year, or month."""
    if not _analytics_ready():
        return _json_error("Analytics not initialized")

    valid_metrics = {
        "Total Revenue",
        "Target Revenue",
        "Credit Revenue",
        "Cash Revenue",
        "No. Cases",
        "No. Services",
        "Total Leakage Revenue Losses",
        "Doctor PMS %",
        "Actual COE Compliance %",
        "No-Show %",
    }
    if metric not in valid_metrics:
        return _json_error(f"Metric '{metric}' is not supported.")

    df = _analytics.filter(bu=bu, year=year)
    if df.empty:
        return _json_error("No data found for the selected filters.")

    percent_metrics = {"Doctor PMS %", "Actual COE Compliance %", "No-Show %"}
    agg = "mean" if metric in percent_metrics else "sum"
    group_key = group_by.lower().strip()

    if group_key in {"bu", "branch"}:
        grouped = df.groupby("BU")[metric].agg(agg).reset_index()
        x_col = "BU"
    elif group_key == "doctor":
        group_cols = ["Doctor Name"] if bu else ["BU", "Doctor Name"]
        grouped = df.groupby(group_cols)[metric].agg(agg).reset_index()
        if not bu and "BU" in grouped.columns:
            grouped["Doctor Name"] = grouped["Doctor Name"].astype(str) + " (" + grouped["BU"].astype(str) + ")"
        x_col = "Doctor Name"
    elif group_key == "year":
        grouped = df.groupby("Year")[metric].agg(agg).reset_index()
        x_col = "Year"
    elif group_key == "month":
        grouped = df.groupby(["Year", "Month No"])[metric].agg(agg).reset_index().sort_values(["Year", "Month No"])
        grouped["Month"] = grouped["Year"].astype(str) + "-" + grouped["Month No"].astype(int).astype(str).str.zfill(2)
        x_col = "Month"
    else:
        return _json_error("group_by must be bu, doctor, year, or month.")

    if metric in percent_metrics:
        grouped[metric] = (grouped[metric] * 100).round(1)
    else:
        grouped[metric] = grouped[metric].round(2)

    if group_key == "doctor":
        grouped = grouped.sort_values(metric, ascending=False).head(max(1, min(int(top_n or 10), 50)))

    return json.dumps({
        "chart_type": chart_type or "bar",
        "metric": metric,
        "group_by": group_key,
        "x": x_col,
        "y": metric,
        "chart_data": grouped.to_dict(orient="records"),
        "filters": {"year": year, "bu": bu},
    }, ensure_ascii=False, default=str)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the RAG knowledge base for KPI definitions, formulas, and playbooks."""
    try:
        from app.core.rag_engine import rag_engine

        if not rag_engine.is_loaded:
            rag_engine.load_and_index_kb()
        results = rag_engine.search(query, top_k=3, min_distance=1.5)
        if not results:
            return _json_error("No relevant information found in Knowledge Base.")
        return json.dumps(results, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.warning("Knowledge base search failed: %s", exc)
        return _json_error("Knowledge Base search is not available right now.")


@tool
def submit_feedback_request(question: str) -> str:
    """Submit a request to add a missing analysis, KPI, or answer."""
    from app.services.request_system import RequestSystem

    result = RequestSystem().submit(question)
    return result["message"]


@tool
def query_data(question: str) -> str:
    """Placeholder for unsupported direct dataframe questions."""
    return json.dumps({"info": "Please rephrase as a clear aggregation, ranking, KPI, or request."}, ensure_ascii=False)


ALL_TOOLS = [
    get_executive_summary,
    get_revenue_achievement,
    get_pms_ranking,
    get_leakage_analysis,
    get_branch_comparison,
    get_coe_compliance,
    get_noshow_analysis,
    get_doctor_multi_branch,
    get_kpi_drivers,
    get_kpi_playbook,
    get_doctors_by_bu,
    rank_doctors_by_column,
    get_summary_statistic,
    flexible_data_query,
    calculate_custom_metric,
    generate_chart_data,
    search_knowledge_base,
    submit_feedback_request,
    query_data,
]
