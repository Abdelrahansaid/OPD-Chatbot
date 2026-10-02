# app/core/real_agent.py
import logging
import re
from typing import Optional, List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.core.tools import ALL_TOOLS, set_analytics_kb
from app.services.request_system import submit_request

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """
You are an intelligent OPD analytics assistant for Andalusia Hospitals.
Branches: ASH (El Shalalat, Alexandria), SMH (Smouha, Alexandria), HJH (Hay El Gamea, Saudi Arabia).
Data covers 2023-2025. You have 24 KPIs available.

**🚨 CRITICAL RULES (NEVER BREAK):**
1. **YOU MUST NOT INVENT** any data (numbers, doctor names, branches, KPIs). Use ONLY tool outputs.
2. **RESPONSE PATH (follow in order):**
   - Step A: Use a specialized tool (get_revenue_achievement, get_pms_ranking, get_branch_comparison, etc.)
   - Step B: If no specialized tool matches, try `flexible_data_query(query)` with the exact user question. This tool can safely compute sums, averages, and groupings by BU/doctor/year.
   - Step C: If `flexible_data_query` returns no data or an error, THEN respond with the request prompt:
     "I couldn't find an answer using available data. Would you like to submit a request? (Reply 'yes' to submit, or 'no' to continue)"
3. **Use `search_knowledge_base` ONLY for:**
   - KPI formulas or definitions (e.g., "what is the formula for No-Show %?")
   - Investigation playbooks or escalation paths (e.g., "If Service Leakage > 8%, what to do?")
   - KPI meanings or drivers explanations.
   **NEVER use it for analytical questions (revenue, leakage, PMS, COE, rankings, comparisons).**
4. **`flexible_data_query` is your safety net for simple aggregations not covered by specialized tools.** Examples of when to use it:
   - "Show me total revenue by branch for 2024"
   - "What is the average PMS score across all doctors in 2025?"
   - "List all doctors who worked in ASH in 2023"
   It will return real data from the OPD dataset, never invented numbers.

**TOOLS AVAILABLE (use in this priority order):**
- Specialized analytics tools: `get_revenue_achievement`, `get_pms_ranking`, `get_leakage_analysis`, `get_branch_comparison`, `get_coe_compliance`, `get_noshow_analysis`, `get_doctor_multi_branch`, `get_doctors_by_bu`, `rank_doctors_by_column`, `get_summary_statistic`, `calculate_custom_metric`, `generate_chart_data`
- Knowledge tools: `get_kpi_drivers`, `get_kpi_playbook`, `search_knowledge_base` (restricted to definitions/playbooks)
- **Last resort query tool:** `flexible_data_query(query)` → use this ONLY when no specialized tool applies and the question is a simple aggregation or filter.
- Request tool: `submit_feedback_request(question)` → use only after failing to get data from other tools.

**IMPORTANT RULES:**
1. For ANY analytics question, call the relevant tool FIRST.
2. For complex analysis (e.g., "identify top 3 drivers"), call BOTH `get_revenue_achievement` AND `get_kpi_drivers`.
3. For "list doctors in [branch]", call `get_doctors_by_bu(bu="SMH")`.
4. For "best/worst doctor", call `get_pms_ranking` or `rank_doctors_by_column`.
5. For "chart", call `generate_chart_data`.
6. NEVER invent numbers, doctor names, or branch data. Use ONLY what tools return.
7. Respond in the SAME LANGUAGE as the user. Arabic question = Arabic answer.
8. Format answers professionally: use bullet points, include exact numbers, end with one recommendation.
9. For "top 3 drivers", list exactly 3 numbered items with their impact values.
10. If asked for word limit (e.g., "max 250 words"), strictly respect it.
11. If a question is completely unrelated to healthcare KPIs (cooking, personal topics), say: "I can only answer OPD KPI analytics questions."

**EXAMPLES:**
- "Analyze the OPD revenue achievement for 2025, identify top 3 drivers":
  Step 1: Call `get_revenue_achievement(year=2025)` → get actual numbers.
  Step 2: Call `get_kpi_drivers(kpi_name="Total Revenue")` → get drivers.
  Step 3: Produce concise answer with numbers and recommendation.

- "Show me total revenue by branch for 2024":
  Step 1: No specialized tool for this exact aggregation.
  Step 2: Call `flexible_data_query(query="total revenue by branch for 2024")` → get real data.
  Step 3: Present results.

- "What is the average waiting time?" → No tool can answer (not a KPI, not in dataset). Respond with request prompt.
- "How many patients visited in 2027?" → Year out of range. `flexible_data_query` will return empty. Then respond with request prompt.

Always think step by step. If `flexible_data_query` returns an error or no data, use the request fallback immediately.
"""
_agent_app = None


def get_agent(analytics, kb):
    """Create or reuse the LangGraph ReAct agent."""
    global _agent_app

    set_analytics_kb(analytics, kb)

    if _agent_app is None:
        llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.1)
        _agent_app = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)

    return _agent_app


def _build_messages(question: str, chat_history: Optional[List[Any]] = None):
    messages = []

    for turn in (chat_history or [])[-3:]:
        if isinstance(turn, dict):
            user_text = turn.get("user") or turn.get("input") or turn.get("question")
            assistant_text = turn.get("assistant") or turn.get("output") or turn.get("answer")
        else:
            user_text = turn[0] if len(turn) > 0 else None
            assistant_text = turn[1] if len(turn) > 1 else None

        if user_text:
            messages.append(HumanMessage(content=str(user_text)))
        if assistant_text:
            messages.append(AIMessage(content=str(assistant_text)))

    messages.append(HumanMessage(content=question))
    return messages


def _is_temporary_api_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(keyword in text for keyword in (
        "429",
        "too many requests",
        "rate limit",
        "rate-limited",
        "quota exceeded",
        "502",
        "503",
        "504",
        "service unavailable",
        "gateway timeout",
    ))


def _is_waiting_for_request_confirmation(chat_history: Optional[List[Any]]) -> bool:
    if not chat_history:
        return False

    last_turn = chat_history[-1]
    if isinstance(last_turn, dict):
        last_assistant = last_turn.get("assistant") or last_turn.get("output") or last_turn.get("answer") or ""
    else:
        last_assistant = last_turn[1] if len(last_turn) > 1 else ""

    text = str(last_assistant).lower()
    return bool(re.search(
        r"(submit .*request|submit a request|send .*request|would you like.*request|هل.*ترغب.*في.*طلب|هل.*تريد.*طلب|هل.*أرسل.*طلب)",
        text
    ))


def _extract_original_question_from_context(chat_history: Optional[List[Any]]) -> Optional[str]:
    if not chat_history:
        return None

    for turn in reversed(chat_history):
        if isinstance(turn, dict):
            user_text = turn.get("user") or turn.get("input") or turn.get("question")
        else:
            user_text = turn[0] if len(turn) > 0 else None
        if user_text:
            return str(user_text)
    return None


def _handle_unknown_question(question: str, analytics, kb, chat_history: Optional[List[Any]], lang: str = "en") -> str:
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


def _handle_user_confirmation(
    question: str,
    analytics,
    kb,
    chat_history: Optional[List[Any]],
    lang: str = "en",
) -> str:
    q_lower = question.lower().strip()
    if q_lower in {"yes", "y", "نعم", "اه", "ايوه", "أيوه", "تمام", "ماشي"}:
        original_question = _extract_original_question_from_context(chat_history) or "Unknown question"
        result = submit_request(original_question, role="General User")
        return result["message"] if isinstance(result, dict) else str(result)

    if q_lower in {"no", "n", "لا", "كلا", "مش", "لأ"}:
        if lang == 'ar':
            return "حسنًا، لا مشكلة. اسأل سؤالًا آخر في تحليل OPD عندما تريد."
        return "Alright, no problem. Feel free to ask another OPD analytics question."

    # If not yes/no, treat as a normal question
    return None


def _looks_like_unknown_answer(answer: str) -> bool:
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
    ]
    return any(marker in text for marker in unknown_markers)


def run_agent(
    question: str,
    analytics,
    kb,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Run the OPD analytics agent on a user question."""
    confirmation_tokens = {"yes", "y", "نعم", "اه", "ايوه", "أيوه", "تمام", "ماشي", "no", "n", "لا", "كلا", "مش", "لأ"}
    lang = 'ar' if re.search(r'[ء-ي]', question) else 'en'
    if _is_waiting_for_request_confirmation(chat_history) and question.lower().strip() in confirmation_tokens:
        return _handle_user_confirmation(question, analytics, kb, chat_history, lang=lang)

    try:
        agent = get_agent(analytics, kb)
        result = agent.invoke({"messages": _build_messages(question, chat_history)})
        messages = result.get("messages", [])
        if not messages:
            return _handle_unknown_question(question, analytics, kb, chat_history, lang=lang)

        answer = str(messages[-1].content)

        # If the agent's answer looks like an unknown-answer fallback, redirect to the request handler
        if _looks_like_unknown_answer(answer) and ("submit a request" not in answer.lower() and "إرسال طلب" not in answer):
            return _handle_unknown_question(question, analytics, kb, chat_history, lang=lang)

        # If the agent explicitly asks to submit a request (EN or AR), return that prompt unchanged
        if "submit a request" in answer.lower() or "إرسال طلب" in answer:
            return answer

        return answer
    except Exception as e:
        if _is_temporary_api_error(e):
            logger.warning("Temporary API error from Groq/LLM service: %s", e)
            if lang == 'ar':
                return (
                    "الخدمة مشغولة حالياً بسبب حدود طلبات Groq أو مشكلة في الاتصال. "
                    "حاول مرة أخرى بعد قليل."
                )
            return (
                "The assistant is currently busy due to Groq rate limits or temporary service issues. "
                "Please try again in a few moments."
            )

        logger.warning("Agent execution failed, offering request fallback: %s", e)
        return _handle_unknown_question(question, analytics, kb, chat_history, lang=lang)
