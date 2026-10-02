#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app/main.py – Andalusia OPD Chatbot Entry Point (Phase 3 Fixed + Memory Enhanced)
✅ Real Groq LLM integration via chatbot_formatter
✅ Conversation memory with pronoun resolution
✅ Fuzzy KPI/doctor matching
✅ Robust intent detection with fallback
"""
import os
import sys
import re
import json
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    OPD_PATH, KB_PATH, BU_LABELS, BU_REGIONS, THRESHOLDS,
    KPI_ALIASES, RANKING_KPI_MAP, DOCTOR_ALIASES, DEBUG_MODE
)
from app.analytics.base import OPDAnalytics
from app.knowledge.kb_helper import KBHelper
from app.utils.filters import (
    normalize_kpi_name, resolve_doctor, resolve_doctor_fuzzy, sanitize_filters,
    extract_year_month, extract_bu_from_question, extract_kpi_from_question
)
from app.utils.guards import check_guards, is_in_scope, SCOPE_MSG_EN, SCOPE_MSG_AR
from app.utils.llm_guard import llm_is_out_of_scope
from app.utils.language import detect_language
from app.utils.intent_classifier import classify_intent
# from app.core.analytics_router import route_analytics
from app.core.chatbot_formatter import format_answer, format_reasoning_answer
from app.prompts import get_empty_message
from app.core.memory import conversation_memory
from app.core.rag_engine import rag_engine
from app.core.rag_engine_v2 import HybridRAGEngine
from app.utils.kpi_similarity import get_kpi_similarity

# ════════════════════════════════════════════════════════════════════════════════
# FIX 1: Clean sheet names with Excel XML-encoded tabs (_x0009_)
# ════════════════════════════════════════════════════════════════════════════════
def clean_sheet_name(name: str) -> str:
    """Remove Excel XML-encoded tabs and strip whitespace."""
    cleaned = re.sub(r'_?x0009_?', '', name, flags=re.IGNORECASE)
    cleaned = cleaned.replace('\t', '').strip('_').strip()
    return cleaned

# ════════════════════════════════════════════════════════════════════════════════
# FIX 2: Convert numpy types for JSON serialization
# ════════════════════════════════════════════════════════════════════════════════
def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif pd.isna(obj):
        return None
    return obj

# ════════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ════════════════════════════════════════════════════════════════════════════════

def extract_question_threshold(question: str) -> dict:
    """Extract threshold value and direction from a conditional KPI question."""
    if not question:
        return {}

    q = question.lower()
    patterns = [
        (r"\b(below|less than|under|<)\s*([0-9]{1,3})\s*%?", "below"),
        (r"\b(above|greater than|over|>)\s*([0-9]{1,3})\s*%?", "above"),
        (r"\bأقل\s*من\s*([0-9]{1,3})\s*%?", "below"),
        (r"\bأكثر\s*من\s*([0-9]{1,3})\s*%?", "above"),
    ]

    for pattern, direction in patterns:
        match = re.search(pattern, q)
        if match:
            threshold = int(match.group(2))
            return {"threshold": threshold, "direction": direction}

    return {}


def _parse_threshold_value(value: str) -> int:
    """Parse the numeric threshold from strings like '< 80%' or '80%'."""
    if value is None:
        return None
    match = re.search(r"([0-9]{1,3})", str(value))
    return int(match.group(1)) if match else None


def is_follow_up_question(question: str) -> bool:
    """Determine whether the current question is a follow-up that should use memory context."""
    q_lower = question.lower()
    # Exclude explicit out-of-scope or definitional questions from memory reuse
    if re.search(r"\b(formula|definition|معادلة|تعريف|rejection|approval|رفض|قبول)\b", q_lower, re.I):
        return False

    follow_up_markers = [
        r'\b(he|him|his|she|her|it|its|they|them|their|that|this|those|these|about|what about|how about)\b',
        r'\b(الفرع ده|الدكتور ده|ده|دي|دول)\b',
        r'^what about\b|^how about\b|^what is his\b|^how is he\b|^his\b'
    ]
    return any(re.search(marker, q_lower, re.I) for marker in follow_up_markers)


def choose_playbook_row(playbook: list, threshold: dict | None = None) -> dict:
    """Choose the most relevant playbook row based on the question threshold."""
    if not playbook:
        return {}

    if not threshold or threshold.get("threshold") is None:
        return playbook[0]

    target = threshold["threshold"]
    scored = []
    for row in playbook:
        row_thr = _parse_threshold_value(row.get("Threshold"))
        if row_thr is None:
            continue
        scored.append((abs(target - row_thr), row))

    if not scored:
        return playbook[0]

    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def build_playbook_response(kpi: str, playbook: list, threshold: dict | None = None) -> str:
    """Build a direct playbook response for escalation / investigation questions."""
    row = choose_playbook_row(playbook, threshold)
    if not row:
        return ""

    escalation = row.get("Escalation") or row.get("escalation") or "not specified"
    action = row.get("Recommended_Action") or row.get("recommended_action") or ""
    scenario = row.get("Scenario") or row.get("scenario") or ""

    if threshold and threshold.get("direction") == "below":
        prefix = f"If {kpi} falls below {threshold.get('threshold')}%,"
    elif threshold and threshold.get("direction") == "above":
        prefix = f"If {kpi} rises above {threshold.get('threshold')}%,"
    else:
        prefix = f"For {kpi},"

    response = (
        f"{prefix} the best escalation path is to escalate to {escalation}. "
        f"Scenario: {scenario}. "
        f"Recommended action: {action}."
    )
    return response


def is_chart_request(question: str) -> bool:
    return bool(re.search(r"\b(chart|graph|plot|visuali[sz]ation|dashboard|رسم|مخطط)\b", question, re.I))


def extract_top_n(question: str, default: int = 5) -> int:
    match = re.search(r"\btop\s*(\d+)|\b(\d+)\s*(?:doctors?|branches?|bus?)", question, re.I)
    if not match:
        return default
    value = match.group(1) or match.group(2)
    return max(1, min(int(value), 50))


def _is_agent_response_unknown(answer: str) -> bool:
    if not answer:
        return True
    text = answer.lower()
    unknown_markers = [
        "i don't know",
        "i do not know",
        "couldn't find",
        "could not find",
        "no data found",
        "not available",
        "cannot answer",
        "can't answer",
        "does not contain",
        "does not include",
        "unable to provide",
        "does not have data",
        "not tracked",
        "not available in",
        "unable to answer",
        "unfortunately",
        "the provided data does not contain",
        "the provided data does not include",
        "the provided data does not have",
        "information is not available",
        "missing in the dataset",
        "not part of this dataset",
        "لم أتمكن",
        "لا أستطيع",
        "لا يوجد",
        "لا تحتوي",
        "لا توجد بيانات",
        "submit a request",
        "إرسال طلب",
    ]
    return any(marker in text for marker in unknown_markers)


def _unknown_request_prompt(lang: str = 'en') -> str:
    if lang == 'ar':
        return (
            "لم أتمكن من إيجاد إجابة لسؤالك باستخدام البيانات المتاحة وKPIs.\n\n"
            "هل تريد إرسال طلب لإضافة هذا التحليل؟ "
            "(أجب 'نعم' للإرسال، أو 'لا' للاستمرار)"
        )
    return (
        "I couldn't find an answer to your question using the available data and KPIs.\n\n"
        "Would you like to **submit a request** to add this analysis? "
        "(Reply 'yes' to submit, or 'no' to continue)"
    )


def _is_complex_analytical_question(question: str) -> bool:
    """Determine if a question requires complex analysis/reasoning via agent."""
    q = question.lower()
    
    # Questions requiring analysis, reasoning, or multiple-step thinking
    analytical_patterns = [
        r"\banalyze\b.*\b(revenue|leakage|pms|coe|noshow)",
        r"\bidentify.*top\s*(?:3|three|2|two).*drivers",
        r"\b(why|what\s+caused)\b.*\b(low|high|gap|decline|increase)",
        r"\b(drivers?\s+)(?:behind|of|for)\b.*\b(revenue|leakage|pms|achievement|gap)",
        r"\b(compare|comparison)\b.*\b(across|between)\b",
        r"\b(provide|give).*(?:concise|detailed|comprehensive).*(?:recommendation|analysis|summary)",
        r"\b(executive|detailed).*(?:summary|recommendation|analysis)\b",
        r"\b(what\s+are.*drivers|what\s+factors|what\s+causes)\b",
        r"\b(impact|effect|contribution)\b.*\b(of|on)\b",
        r"\b(explain|justify)\b.*\b(gap|difference|variance)",
    ]
    
    is_analytical = any(re.search(pattern, q, re.I) for pattern in analytical_patterns)
    # But exclude if it's just a definition/formula question
    is_definition = re.search(r"\b(formula|definition|meaning|what\s+is.*calculated|how.*calculated)\b", q, re.I)
    
    return is_analytical and not is_definition


def infer_chart_request(question: str) -> Dict[str, Any]:
    q = question.lower()
    metric = extract_kpi_from_question(question) or "Total Revenue"
    if metric in {"Service Leakage %", "leakage"}:
        metric = "Total Leakage Revenue Losses"
    if "pms" in q:
        metric = "Doctor PMS %"
    elif "coe" in q or "compliance" in q:
        metric = "Actual COE Compliance %"
    elif "credit" in q:
        metric = "Credit Revenue"
    elif "cash" in q:
        metric = "Cash Revenue"
    elif "case" in q:
        metric = "No. Cases"

    if re.search(r"\b(month|monthly|trend|over time)\b", q, re.I):
        group_by = "month"
        chart_type = "line"
    elif re.search(r"\b(year|yearly|annual)\b", q, re.I):
        group_by = "year"
        chart_type = "line"
    elif re.search(r"\bdoctor|physician|دكتور\b", q, re.I):
        group_by = "doctor"
        chart_type = "horizontal_bar"
    else:
        group_by = "bu"
        chart_type = "bar"

    return {"metric": metric, "group_by": group_by, "chart_type": chart_type, "top_n": extract_top_n(question, 10)}


def build_chart_data(
    analytics: OPDAnalytics,
    metric: str,
    group_by: str,
    chart_type: str = "bar",
    bu: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    d = analytics.filter(bu=bu, year=year, month=month)
    if d.empty:
        return {"error": "No data found for the selected filters."}
    if metric not in d.columns:
        return {"error": f"Metric '{metric}' is not available in the dataset."}

    percent_metrics = {"Doctor PMS %", "Actual COE Compliance %", "No-Show %"}
    agg = "mean" if metric in percent_metrics else "sum"
    group_key = group_by.lower().strip()

    if group_key in {"bu", "branch"}:
        grouped = d.groupby("BU")[metric].agg(agg).reset_index()
        x_col = "BU"
    elif group_key == "doctor":
        group_cols = ["Doctor Name"] if bu else ["BU", "Doctor Name"]
        grouped = d.groupby(group_cols)[metric].agg(agg).reset_index()
        if not bu and "BU" in grouped.columns:
            grouped["Doctor Name"] = grouped["Doctor Name"].astype(str) + " (" + grouped["BU"].astype(str) + ")"
        grouped = grouped.sort_values(metric, ascending=False).head(top_n)
        x_col = "Doctor Name"
    elif group_key == "year":
        grouped = d.groupby("Year")[metric].agg(agg).reset_index().sort_values("Year")
        x_col = "Year"
    elif group_key == "month":
        grouped = d.groupby(["Year", "Month No"])[metric].agg(agg).reset_index().sort_values(["Year", "Month No"])
        grouped["Month"] = grouped["Year"].astype(str) + "-" + grouped["Month No"].astype(int).astype(str).str.zfill(2)
        x_col = "Month"
    else:
        return {"error": "Unsupported chart grouping."}

    if metric in percent_metrics:
        grouped[metric] = (grouped[metric] * 100).round(1)
    else:
        grouped[metric] = grouped[metric].round(2)

    return {
        "chart_type": chart_type,
        "metric": metric,
        "group_by": group_key,
        "x": x_col,
        "y": metric,
        "chart_data": grouped.to_dict(orient="records"),
        "filters": {"bu": bu, "year": year, "month": month},
    }


def infer_rank_request(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()
    if not re.search(r"\b(rank|ranking|top|highest|lowest|best|worst)\b", q, re.I):
        return None

    column = None
    ascending = False
    intent = "dynamic_ranking"
    if "coe" in q or "compliance" in q:
        column = "Actual COE Compliance %"
    elif "credit" in q:
        column = "Credit Revenue"
    elif "cash" in q:
        column = "Cash Revenue"
    elif "leak" in q:
        column = "Total Leakage Revenue Losses"
    elif "case" in q:
        column = "No. Cases"
    elif "pms" in q or "best doctor" in q:
        column = "Doctor PMS %"
        intent = "pms_ranking"
    elif "revenue" in q or "revnue" in q or "revanue" in q or "revenu" in q:
        column = "Total Revenue"

    if "worst" in q or "lowest" in q or "poor" in q:
        ascending = True

    if not column:
        return None
    default_top = 1 if re.search(r"\bwho\b.*\b(highest|lowest|best|worst)\b|\b(best|worst|highest|lowest)\b.*\b(doctors?|doctoer|physician)\b|doctor.*\b(best|worst|highest|lowest)\b", q, re.I) else 5
    return {"intent": intent, "column": column, "top_n": extract_top_n(question, default_top), "ascending": ascending}


def rank_by_dataset_column(
    analytics: OPDAnalytics,
    column: str,
    bu: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    top_n: int = 5,
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    d = analytics.filter(bu=bu, year=year, month=month)
    if d.empty or column not in d.columns:
        return []
    group_cols = ["Doctor Name"] if bu else ["BU", "Doctor Name"]
    agg = "mean" if column in {"Doctor PMS %", "Actual COE Compliance %", "No-Show %"} else "sum"
    result = d.groupby(group_cols)[column].agg(agg).reset_index()
    if column in {"Doctor PMS %", "Actual COE Compliance %", "No-Show %"}:
        result[column] = (result[column] * 100).round(1)
    else:
        result[column] = result[column].round(2)
    if not bu and "BU" in result.columns:
        result["Doctor Name"] = result["Doctor Name"].astype(str) + " (" + result["BU"].astype(str) + ")"
    result = result.sort_values(column, ascending=ascending).head(top_n)
    return result[["Doctor Name", column]].rename(columns={column: "Value"}).to_dict(orient="records")

# ════════════════════════════════════════════════════════════════════════════════
# Load Data
# ════════════════════════════════════════════════════════════════════════════════
def load_data():
    """Load OPD dataset and Knowledge Base with cleaned sheet names."""
    print("📂 Loading data files...")
    
    # Load OPD Dataset
    opd_df = pd.read_excel(OPD_PATH)
    opd_df.columns = opd_df.columns.str.strip()
    print(f"  ✅ Loaded OPD Dataset: {len(opd_df)} records")

    # Load Knowledge Base with multiple sheets
    kb_raw = pd.read_excel(KB_PATH, sheet_name=None)
    kb = {}
    
    for sheet_name, df in kb_raw.items():
        clean_name = clean_sheet_name(sheet_name)
        kb[clean_name] = df
        print(f"  ✅ Loaded KB sheet: '{clean_name}' ({len(df)} rows)")
    
    print("📋 Available KB sheets:", list(kb.keys()))    
    return opd_df, kb

# ════════════════════════════════════════════════════════════════════════════════
# FIX 3: Enhanced rule-based intent detection with fuzzy matching
# ════════════════════════════════════════════════════════════════════════════════

def rule_based_intent(question: str, kpi_similarity=None) -> Optional[Dict]:
    """Rule-based intent detection with fuzzy KPI/doctor matching."""
    q = question.lower().strip()

    if is_chart_request(question):
        return {"intent": "chart_request", "filters": {}, "chart": infer_chart_request(question)}

    # ========== FIRST PRIORITY: KNOWLEDGE BASE (NO EXCEPTIONS) ==========
    kb_patterns = [
        r"according to the kb", r"knowledge base", r"drivers of total revenue",
        r"playbook", r"escalation", r"formula", r"definition",
        r"filter compatibility", r"scope", r"available at doctor level",
        r"معادلة", r"تعريف", r"سيناريو", r"تصعيد", r"خطوات التحقيق"
    ]
    if any(re.search(p, q, re.I) for p in kb_patterns):
        return {"intent": "knowledge_base", "filters": {}}

    # High-priority direct patterns before generic doctor/KPI matching.
    rank_request = infer_rank_request(question)
    if rank_request:
        if rank_request["intent"] == "pms_ranking":
            return {"intent": "pms_ranking", "filters": {"top_n": rank_request["top_n"], "ascending": rank_request["ascending"]}}
        return {**rank_request, "intent": "rank_doctors_by_column", "filters": {}}

    if re.search(r"\b(most\s+problematic|problematic\s+doctor)\b", q, re.I):
        return {"intent": "worst_doctor", "filters": {}}

    if re.search(r"\b(top\s*3\s+operational\s+risks?|operational\s+risks?)\b", q, re.I):
        return {"intent": "top_operational_risks", "filters": {}}

    if re.search(r"top\s+doctor\s+by\s+pms", q, re.I):
        return {"intent": "pms_ranking", "filters": {"top_n": 1}}

    if re.search(r"\b(list|show|get)\b.*\b(doctors|physicians)\b", q, re.I):
        bu_match = re.search(r"\b(ASH|SMH|HJH)\b", question, re.I)
        bu_value = bu_match.group(1).upper() if bu_match else extract_bu_from_question(question)
        if bu_value in ["ASH", "SMH", "HJH"]:
            return {"intent": "doctors_by_bu", "filters": {"bu": bu_value}}

    return None

# ════════════════════════════════════════════════════════════════════════════════
# Main Chatbot Function (AGENT-FIRST ROUTING)
# ════════════════════════════════════════════════════════════════════════════════
def chatbot_answer(
    question: str,
    analytics: OPDAnalytics,
    kb: KBHelper,
    debug: bool = False,
    kpi_similarity=None,
    role: str = 'General User',
    pre_filters: Optional[Dict[str, Any]] = None,
    return_data: bool = False,
    hybrid_rag=None,
    **kwargs
) -> Any:
    """Main chatbot entry point – with agent-first routing."""
    
    # 1. Language detection
    lang = detect_language(question)
    logger.info(f"[CHATBOT] Question (lang={lang}): {question[:80]}...")

    # 2. GUARD – FIRST: block out-of-scope or identity questions
    guard = check_guards(question, lang, available_years=analytics.available_years)
    if guard:
        return guard if not return_data else (guard, None, None)

    # 3. AGENT-FIRST ROUTING: كل الأسئلة تذهب إلى الـ Agent أولاً
    from app.core.real_agent import run_agent
    chat_history = [
        {"user": turn.user_message, "assistant": turn.bot_response}
        for turn in conversation_memory.turns[-3:]
    ]
    agent_response = run_agent(question, analytics, kb, chat_history)
    
    # دائماً نحفظ ونعود بإجابة الـ Agent
    if "would you like to submit a request" in agent_response.lower() or "إرسال طلب" in agent_response:
        conversation_memory.add_turn(question, agent_response, "agent_unknown", {"agent": "real_agent"})
        if return_data:
            return agent_response, {"suggest_request": True, "no_answer_found": True}, "agent_unknown"
        return agent_response
    
    # أي إجابة أخرى من الـ Agent - نعتبرها صحيحة ونعود بها
    conversation_memory.add_turn(question, agent_response, "agent_response", {"agent": "real_agent"})
    if return_data:
        return agent_response, {}, "agent_response"
    return agent_response


# ════════════════════════════════════════════════════════════════════════════════
# Terminal Chat Loop
# ════════════════════════════════════════════════════════════════════════════════
def run_terminal_chat():
    """Interactive terminal chat interface."""
    print("\n" + "=" * 70)
    print("  ANDALUSIA OPD CHATBOT – V2.0 (Phase 3 Fixed + Memory Enhanced)")
    print("  Commands: 'exit' | 'clear' | 'history' | 'reload kb'")
    print("=" * 70 + "\n")
    
    opd_df, kb_raw = load_data()
    analytics = OPDAnalytics(opd_df)
    kb = KBHelper(
        formulas=kb_raw.get('adx_kpi_formula_definition', pd.DataFrame()),
        knowledge_map=kb_raw.get('adx_kpi_knowledge_map', pd.DataFrame()),
        relationship=kb_raw.get('adx_kpi_relationship_map', pd.DataFrame()),
        playbook=kb_raw.get('adx_kpi_investigation_playbook', pd.DataFrame()),
        filter_compat=kb_raw.get('adx_kpi_filter_compatibility', pd.DataFrame()),
        scope=kb_raw.get('adx_dim_kpi_scope', pd.DataFrame())
    )

    print(f"✅ System Ready! {len(opd_df)} records indexed.")
    
    print("📚 Loading RAG Engine for semantic search...")
    hybrid_rag = None
    try:
        rag_engine.load_and_index_kb(KB_PATH)
        try:
            from app.core.rag_engine_v2 import HybridRAGEngine
            hybrid_rag = HybridRAGEngine(rag_engine.collection)
            print(f"✅ Hybrid RAG Engine initialized: {hybrid_rag.get_stats()}")
        except Exception as e:
            print(f"⚠️ Hybrid RAG Engine initialization failed: {e} (will use basic RAG)")
            hybrid_rag = None
        
        kpi_list = kb.list_all_kpis()
        kpi_similarity = get_kpi_similarity(kpi_list)
        print(f"✅ RAG Engine loaded: {rag_engine.get_stats()}\n")
    except Exception as e:
        print(f"⚠️ RAG Engine setup failed: {e} (will use pattern matching only)\n")
        kpi_similarity = None

    history = []
    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'خروج', 'اخرج']:
            print("\n👋 Goodbye!")
            if history:
                print("\n📜 Chat History:\n" + "-" * 60)
                for h in history[-5:]:
                    print(f"🧑 {h['user']}\n🤖 {h['assistant'][:200]}...\n")
            break
        if user_input.lower() == 'clear':
            history.clear()
            conversation_memory.clear()
            print("✅ History and conversation memory cleared.")
            continue
        if user_input.lower() == 'reload kb':
            print("🔄 Reloading knowledge base...")
            _, kb_raw = load_data()
            kb = KBHelper(
                formulas=kb_raw.get('adx_kpi_formula_definition', pd.DataFrame()),
                knowledge_map=kb_raw.get('adx_kpi_knowledge_map', pd.DataFrame()),
                relationship=kb_raw.get('adx_kpi_relationship_map', pd.DataFrame()),
                playbook=kb_raw.get('adx_kpi_investigation_playbook', pd.DataFrame()),
                filter_compat=kb_raw.get('adx_kpi_filter_compatibility', pd.DataFrame()),
                scope=kb_raw.get('adx_dim_kpi_scope', pd.DataFrame())
            )
            print("✅ KB reloaded.")
            continue
        
        ts = datetime.now().strftime("%H:%M:%S")
        answer = chatbot_answer(user_input, analytics, kb, debug=DEBUG_MODE, kpi_similarity=kpi_similarity, hybrid_rag=hybrid_rag)
        print(f"\n🤖 Assistant [{ts}]\n{'─' * 60}\n{answer}\n")
        history.append({"time": ts, "user": user_input, "assistant": answer})

if __name__ == "__main__":
    run_terminal_chat()
