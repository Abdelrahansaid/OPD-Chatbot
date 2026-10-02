"""
app/ui/app.py — Andalusia OPD Chatbot UI
Phase 5-6 Fixed: st.chat_input, st.chat_message, dynamic suggestions, better charts
"""
import sys
import inspect
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.main import load_data, chatbot_answer
from app.analytics.base import OPDAnalytics
from app.knowledge.kb_helper import KBHelper
from app.core.rag_engine import rag_engine
from app.core.memory import conversation_memory
from app.core.chart_agent import ChartAgent
from app.utils.kpi_similarity import get_kpi_similarity
from app.config import KB_PATH, ROLES, RAG_ENABLED
from app.ui.charts import generate_chart


# ─── cached startup ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_app():
    opd_df, kb_raw = load_data()
    analytics = OPDAnalytics(opd_df)
    kb = KBHelper(
        formulas=kb_raw.get('adx_kpi_formula_definition', pd.DataFrame()),
        knowledge_map=kb_raw.get('adx_kpi_knowledge_map', pd.DataFrame()),
        relationship=kb_raw.get('adx_kpi_relationship_map', pd.DataFrame()),
        playbook=kb_raw.get('adx_kpi_investigation_playbook', pd.DataFrame()),
        filter_compat=kb_raw.get('adx_kpi_filter_compatibility', pd.DataFrame()),
        scope=kb_raw.get('adx_dim_kpi_scope', pd.DataFrame()),
    )
    kpi_list = kb.list_all_kpis()
    kpi_similarity = None
    hybrid_rag = None
    try:
        if RAG_ENABLED:
            rag_engine.load_and_index_kb(KB_PATH)

            if rag_engine.is_loaded and rag_engine.collection is not None:
                try:
                    from app.core.rag_engine_v2 import HybridRAGEngine
                    hybrid_rag = HybridRAGEngine(rag_engine.collection)
                except Exception as e:
                    import logging
                    logging.debug(f"Hybrid RAG initialization failed: {e}")
                    hybrid_rag = None

            rag_status = f"✅ RAG: {rag_engine.get_stats().get('document_count', '?')} docs indexed"
        else:
            rag_status = "⚠️ RAG disabled: set RAG_ENABLED=1 and HF_TOKEN to enable"

        # KPI semantic matching is optional and may be skipped on low-memory environments
        kpi_similarity = None
    except Exception as e:
        kpi_similarity = None
        hybrid_rag = None
        rag_status = f"⚠️ RAG unavailable: {str(e)[:40]}"
    chart_agent = ChartAgent()
    return analytics, kb, kpi_similarity, rag_status, chart_agent, hybrid_rag


# ─── dynamic suggestions ──────────────────────────────────────
def get_suggestions(role: str, bu: str, year: str) -> list[str]:
    b = f" in {bu}" if bu != "All" else ""
    y = f" for {year}" if year != "All" else ""
    base = {
        "CEO": [
            f"Executive summary{b}{y}",
            f"Top 3 operational risks{b}",
            f"Revenue achievement vs target{y}",
            "Which branch is performing best?",
            "Worst performing doctor overall?",
        ],
        "OPD Manager": [
            f"Which doctor has highest leakage{b}?",
            f"COE compliance ranking{y}",
            f"Doctor PMS ranking{b}",
            f"Revenue vs target{b}{y}",
            f"No-show rate analysis{y}",
        ],
        "HR Analysis": [
            f"Doctor PMS ranking{b}",
            "Who has best overall performance score?",
            f"No-show rate by doctor{y}",
            f"Doctor performance breakdown{b}{y}",
        ],
        "OPD Coordinator": [
            f"Booking utilization{b}{y}",
            f"No-show rate{b}{y}",
            "Cancellation analysis",
            f"Missed opportunities{b}",
        ],
        "Doctor Staff": [
            "What is my PMS score?",
            "How is my COE compliance?",
            "What are the KPI drivers for revenue?",
            "How can I improve my performance?",
        ],
        "General User": [
            f"What is revenue achievement{y}?",
            "Which BU has highest revenue?",
            "Doctor performance analysis",
            "What are the top operational risks?",
            "Explain the leakage KPI",
        ],
    }
    return base.get(role, base["General User"])[:4]


# ─── CSS ──────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
/* ── Global ──────────────────────────── */
.stApp { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

/* ── Sidebar ─────────────────────────── */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

/* ── Chat messages ───────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 6px !important;
    padding: 4px 0 !important;
}

/* ── Chat input ──────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    padding: 14px 20px !important;
    font-size: 15px !important;
    border: 1.5px solid #334155 !important;
    background: #1e293b !important;
    color: #f1f5f9 !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
[data-testid="stChatInput"] button {
    background: #3b82f6 !important;
    border-radius: 50% !important;
    border: none !important;
}

/* ── Suggestion buttons ──────────────── */
.stButton > button {
    border-radius: 20px !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #94a3b8 !important;
    transition: all 0.2s !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    background: #3b82f6 !important;
    color: #fff !important;
    border-color: #3b82f6 !important;
    transform: translateY(-1px) !important;
}

/* ── Plotly charts ───────────────────── */
.js-plotly-plot {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Selectbox labels ────────────────── */
.stSelectbox label { 
    font-size: 13px !important; 
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Info/status boxes ───────────────── */
.status-badge {
    display: inline-block;
    background: #064e3b;
    color: #6ee7b7;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 4px;
}

/* ── Dividers ────────────────────────── */
hr { border-color: #1f2937 !important; }

/* ── Scrollbar ───────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* ── Main header ─────────────────────── */
.main-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 0 4px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 8px;
    position: sticky;
    top: 0;
    z-index: 999;
    background: #0f172a;
    backdrop-filter: blur(12px);
}
.main-header h1 { 
    font-size: 24px !important;
    font-weight: 700;
    margin: 0 !important;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Filter chip (active filter display) ── */
.filter-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #2563eb;
    border-radius: 12px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─── process a question ───────────────────────────────────────
def process_question(question: str, analytics, kb, kpi_similarity, chart_agent, hybrid_rag=None):
    """Run chatbot and append result to st.session_state.messages."""
    pre_filters = {
        "bu": None if st.session_state.selected_bu == "All" else st.session_state.selected_bu,
        "year": None if st.session_state.selected_year == "All" else int(st.session_state.selected_year),
    }

    msg_id = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question, "id": msg_id})

    with st.spinner("🤔 Analyzing…"):
        try:
            kwargs = {
                "debug": False,
                "kpi_similarity": kpi_similarity,
                "role": st.session_state.selected_role,
                "pre_filters": pre_filters,
                "return_data": True,
            }
            if "hybrid_rag" in inspect.signature(chatbot_answer).parameters:
                kwargs["hybrid_rag"] = hybrid_rag

            response, data, intent = chatbot_answer(
                question,
                analytics,
                kb,
                **kwargs,
            )
            if isinstance(data, dict) and data.get("suggest_request"):
                st.session_state.pending_unknown_question = question
                response = response + "\n\n---\n❓ **I couldn't find an answer. Would you like to submit a request for this question?**"

            # Try to render chart via ChartAgent (encapsulates generate_chart)
            chart = None
            if chart_agent:
                render_fn = getattr(chart_agent, 'render_chart', None)
                if callable(render_fn):
                    chart = render_fn(intent or "general_question", data)
                else:
                    # Fallback to app.ui.charts.generate_chart if render_chart unavailable
                    try:
                        chart = generate_chart(intent or "general_question", data)
                    except Exception:
                        chart = None

                if chart is None:
                    schema = chart_agent.suggest_chart_schema(question, intent)
                    if schema and schema.get("intent"):
                        render_fn = getattr(chart_agent, 'render_chart', None)
                        if callable(render_fn):
                            chart = render_fn(schema["intent"], data)
                        else:
                            try:
                                chart = generate_chart(schema["intent"], data)
                            except Exception:
                                chart = None
        except Exception as exc:
            response = f"⚠️ Error: {str(exc)[:120]}"
            chart = None
            intent = None

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "chart": chart, "intent": intent, "id": msg_id + 1}
    )


# ─── main ─────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Andalusia OPD Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # ── Session state defaults ────────────────────────────────
    defaults = {
        "messages": [],
        "selected_role": "General User",        "selected_bu": "All",
        "selected_year": "All",        "pending_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Load resources ────────────────────────────────────────
    analytics, kb, kpi_similarity, rag_status, chart_agent, hybrid_rag = initialize_app()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏥 Andalusia KPI")
        st.markdown("<small style='color:#64748b'>OPD Smart Analytics Assistant</small>", unsafe_allow_html=True)
        st.divider()

        st.markdown("**Role**")
        role_opts = ["General User", "OPD Manager", "CEO", "HR Analysis", "OPD Coordinator", "Doctor Staff"]
        st.session_state.selected_role = st.selectbox(
            "role", role_opts,
            index=role_opts.index(st.session_state.selected_role),
            label_visibility="collapsed",
        )

        st.markdown("**Business Unit**")
        bu_opts = ["All"] + analytics.available_branches
        st.session_state.selected_bu = st.selectbox(
            "bu", bu_opts,
            index=bu_opts.index(st.session_state.selected_bu),
            label_visibility="collapsed",
        )

        st.markdown("**Year**")
        yr_opts = ["All"] + [str(y) for y in analytics.available_years]
        st.session_state.selected_year = st.selectbox(
            "year", yr_opts,
            index=yr_opts.index(st.session_state.selected_year),
            label_visibility="collapsed",
        )

        st.divider()

        # RAG status
        st.markdown(f'<span class="status-badge">{rag_status}</span>', unsafe_allow_html=True)
        st.markdown(f"<small style='color:#475569'>{len(analytics.available_doctors)} doctors · {len(analytics.available_branches)} branches</small>", unsafe_allow_html=True)

        st.divider()

        # Clear chat
        if st.button("🗑️ Clear Chat", width='stretch'):
            st.session_state.messages = []
            conversation_memory.clear()
            st.rerun()


    # ── MAIN AREA ─────────────────────────────────────────────
    # Header
    st.markdown(
        '<div class="main-header"><span style="font-size:32px">🏥</span><h1>Andalusia OPD Chatbot</h1></div>',
        unsafe_allow_html=True,
    )

    # ── Chat messages ─────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            """
            <div style='text-align:center;padding:60px 20px;color:#475569;'>
                <div style='font-size:48px;margin-bottom:16px'>🏥</div>
                <div style='font-size:20px;font-weight:600;color:#64748b;margin-bottom:8px'>
                    Andalusia KPI Assistant
                </div>
                <div style='font-size:14px;color:#475569'>
                    Ask about revenue, doctors, branches, or any KPI.<br>
                    Supports English and Arabic 🇬🇧 🇸🇦
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for i, msg in enumerate(st.session_state.messages):
            avatar = "👤" if msg["role"] == "user" else "🏥"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                # Chart inline inside bot message
                if msg["role"] == "assistant" and msg.get("chart"):
                    st.plotly_chart(
                        msg["chart"],
                        width='stretch',
                        key=f"chart_{msg['id']}",
                    )


    # ── Suggested questions ───────────────────────────────────
    suggestions = get_suggestions(
        st.session_state.selected_role,
        st.session_state.selected_bu,
        st.session_state.selected_year,
    )

    if not st.session_state.messages:
        # Show larger suggestion cards when chat is empty
        st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
        st.markdown("**💡 Try asking:**", unsafe_allow_html=False)
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(sug, key=f"sug_empty_{i}", width='stretch'):
                    st.session_state.pending_question = sug
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Show compact suggestion row below chat
        st.markdown("<div style='margin:8px 0 4px;'><small style='color:#475569'>Quick questions:</small></div>", unsafe_allow_html=True)
        sug_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with sug_cols[i]:
                if st.button(sug, key=f"sug_chat_{i}", width='stretch'):
                    st.session_state.pending_question = sug
                    st.rerun()

    # ── Process pending question (from suggestion buttons) ────
    # Must happen BEFORE st.chat_input to avoid widget-order issues
    if st.session_state.get("pending_question"):
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        process_question(q, analytics, kb, kpi_similarity, chart_agent, hybrid_rag)
        st.rerun()

    # ── Chat input (native — auto-clears, no session state bug) ──
    if prompt := st.chat_input(
        "Ask about revenue, KPIs, doctors… (English or Arabic)",
        key="chat_input",
    ):
        process_question(prompt, analytics, kb, kpi_similarity, chart_agent, hybrid_rag)
        st.rerun()

    # ── Footer ─────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;padding:12px 0;color:#334155;font-size:11px;'>"
        "Uses OPD dataset (2023-2025) · Semantic RAG · Filter-aware · Role-adaptive responses"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()