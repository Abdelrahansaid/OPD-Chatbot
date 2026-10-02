import logging
from typing import Optional

from app.core.llm_client import get_llm_client

logger = logging.getLogger(__name__)


def llm_is_out_of_scope(question: str, lang: str = 'en') -> bool:
    """Use the Groq LLM to classify whether the question is out of scope for OPD KPI analytics."""
    try:
        llm_client = get_llm_client()
        if not llm_client.is_ready:
            logger.warning("LLM guard unavailable: client not ready")
            return False

        prompt = f"""
You are a classifier. Answer ONLY with YES or NO.
Question: {question}
Is this question out of scope for an OPD healthcare KPI chatbot focused on doctor/branch KPI analytics, revenue, leakage, PMS, COE, no-show, booking, cases, and investigation playbooks?
Out of scope topics include: insurance rejection or approval rates, patient-level data, specialty breakdowns, forecasting, manipulation/fraud, patient acquisition formulas, and unrelated billing metrics.
Respond with exactly YES or NO.
"""

        response = llm_client.chat(
            messages=[
                {"role": "system", "content": "You are a precise scope classifier for an OPD KPI chatbot."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        normalized = response.strip().upper()
        logger.debug(f"LLM guard classification response: {normalized}")
        return "YES" in normalized and "NO" not in normalized
    except Exception as e:
        logger.warning(f"LLM guard failed: {e}")
        return False
