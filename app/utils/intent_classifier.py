"""
app/utils/intent_classifier.py

Intent classifier using Groq LLM.
Replaces brittle rule-based intent routing with a single category decision.
"""
import json
import logging
import re
from app.core.llm_client import get_llm_client

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are an expert in OPD healthcare KPIs.
Classify the user question into exactly one category.

Categories:
- OUT_OF_SCOPE: Questions about insurance rejection rates, approval rates, patient-level details, forecasting, specialties, fraud, or anything not trackable in OPD dataset or Knowledge Base.
- ANALYTICS: Questions about revenue, achievement, leakage, PMS, COE, no-show, booking, cases, doctor performance, or branch comparison.
- KNOWLEDGE_BASE: Questions asking for formula, definition, drivers, investigation steps, escalation, playbook, or KPI metadata.

Return ONLY a JSON object: {{"category": "OUT_OF_SCOPE"}} or {{"category": "ANALYTICS"}} or {{"category": "KNOWLEDGE_BASE"}}.

Question: {question}
"""

VALID_CATEGORIES = {"OUT_OF_SCOPE", "ANALYTICS", "KNOWLEDGE_BASE"}

def classify_intent(question: str) -> str:
    """Return one of: OUT_OF_SCOPE, ANALYTICS, KNOWLEDGE_BASE."""
    
    # ========== FORCE KNOWLEDGE BASE FOR KB QUESTIONS (NO LLM) ==========
    kb_indicators = [
        r"according to the KB", r"knowledge base", r"drivers of Total Revenue",
        r"playbook", r"escalation", r"formula", r"definition", r"filter compatibility",
        r"scope", r"available at doctor level", r"معادلة", r"تعريف", r"سيناريو",
        r"تصعيد", r"خطوات التحقيق", r"If .*%", r"if .* below",
        r"main drivers of",          # ✅ جديد
        r"what are the main drivers", # ✅ جديد
        r"according to the knowledge base", # ✅ جديد
    ]
    if any(re.search(pattern, question, re.I) for pattern in kb_indicators):
        return "KNOWLEDGE_BASE"
    
    # ========== FORCE OUT_OF_SCOPE FOR APPROVAL/REJECTION RATES ==========
    if re.search(r"\b(approval rate|rejection %|rejection percentage|نسبة (الرفض|القبول))\b", question, re.I):
        return "OUT_OF_SCOPE"
    
    # ========== FALLBACK TO LLM ==========
    llm = get_llm_client()
    if not llm.is_ready:
        logger.warning("LLM classifier unavailable: falling back to ANALYTICS")
        return "ANALYTICS"

    prompt = CLASSIFICATION_PROMPT.format(question=question)
    try:
        response = llm.chat([
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt}
        ], max_tokens=50, temperature=0)

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            category = str(data.get("category", "ANALYTICS")).strip().upper()
            if category in VALID_CATEGORIES:
                return category
            logger.warning(f"Unknown classification category from LLM: {category}")
        else:
            logger.warning(f"LLM response did not contain valid JSON: {response[:200]}")
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")

    return "ANALYTICS"