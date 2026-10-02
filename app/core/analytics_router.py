"""
app/core/analytics_router.py

LLM-based analytics routing for Andalusia OPD Chatbot.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.llm_client import get_llm_client

logger = logging.getLogger(__name__)

VALID_INTENTS = {
    "revenue_vs_target",
    "leakage_analysis",
    "pms_ranking",
    "coe_compliance",
    "noshow_analysis",
    "booking_utilization",
    "volume_analysis",
    "executive_summary",
    "branch_comparison",
    "top_operational_risks",
    "doctor_multi_bu",
    "dynamic_ranking",
    "best_doctor",
    "worst_doctor",
    "cross_high_rev_low_coe",
    "cross_low_pms_high_leakage",
    "kpi_drivers",
    "kpi_playbook",
    "general_question",
}

ROUTING_PROMPT = """You are an analytics router for an OPD healthcare KPI chatbot.

Classify the user's question into exactly one canonical analytics intent from this list:
- revenue_vs_target
- leakage_analysis
- pms_ranking
- coe_compliance
- noshow_analysis
- booking_utilization
- volume_analysis
- executive_summary
- branch_comparison
- top_operational_risks
- doctor_multi_bu
- dynamic_ranking
- best_doctor
- worst_doctor
- cross_high_rev_low_coe
- cross_low_pms_high_leakage
- kpi_drivers
- kpi_playbook
- general_question

Return ONLY a JSON object with two keys:
{"intent": "<intent_name>", "params": { ... }}

Use the question context to populate params when helpful, such as year, month, bu, doctor, kpi_name, threshold, rank_by.
If you are unsure, return {"intent": "none", "params": {}}.

Available years: {available_years}

Question: {question}
"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def route_analytics(question: str, available_years: Optional[List[int]] = None) -> Tuple[str, Dict[str, Any]]:
    llm = get_llm_client()
    if not llm.is_ready:
        logger.warning("Analytics router unavailable: LLM client not ready")
        return "none", {}

    available_years_str = ", ".join(str(year) for year in available_years) if available_years else ""
    prompt = ROUTING_PROMPT.format(question=question, available_years=available_years_str)

    try:
        response = llm.chat(
            messages=[
                {"role": "system", "content": "You are a precise analytics router. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0
        )

        data = _extract_json(response)
        if not data:
            logger.warning(f"Analytics router could not parse JSON response: {response}")
            return "none", {}

        intent = str(data.get("intent", "none")).strip()
        params = data.get("params", {}) or {}

        if intent not in VALID_INTENTS:
            logger.warning(f"Analytics router returned unknown intent: {intent}")
            return "none", {}

        if not isinstance(params, dict):
            logger.warning("Analytics router params are not a JSON object")
            params = {}

        return intent, params

    except Exception as e:
        logger.warning(f"Analytics routing failed: {e}")
        return "none", {}
