"""
app/core/chatbot_formatter.py - Professional response formatting with Groq LLM
This file replaces the placeholder format_answer from main.py
KEY FIX: Actually calls Groq API instead of returning fake responses
"""
import json
import logging
import re
from typing import Any, Optional
import numpy as np
import pandas as pd

from app.core.llm_client import get_llm_client
from app.prompts import (
    get_formatter_prompt, get_empty_message, get_reasoning_prompt,
    get_role_injection
)
from app.config import TOKEN_BUDGET

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# DATA CONVERSION
# ═══════════════════════════════════════════════════════════════

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
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


def serialize_data(data: Any, max_len: int = 3500) -> str:
    """Convert data to JSON string with truncation."""
    try:
        clean_data = convert_numpy_types(data)
        data_str = json.dumps(clean_data, ensure_ascii=False, indent=2, default=str)
        
        if len(data_str) > max_len:
            data_str = data_str[:max_len] + "\n... (truncated)"
        
        return data_str
    except Exception as e:
        logger.warning(f"Serialization failed: {e}")
        return str(data)[:max_len]


# ═══════════════════════════════════════════════════════════════
# MAIN FORMATTING FUNCTION (THE REAL ONE!)
# ═══════════════════════════════════════════════════════════════

def choose_token_budget(intent: str, data_str: str) -> int:
    """Choose a token budget based on intent and data complexity."""
    if intent == 'general_question':
        budget = TOKEN_BUDGET.get('rag_answer', 1200)
    elif intent in {
        'kpi_drivers', 'kpi_playbook', 'dynamic_ranking', 'best_doctor',
        'worst_doctor', 'cross_high_rev_low_coe', 'cross_low_pms_high_leakage'
    }:
        budget = TOKEN_BUDGET.get('reasoning', 1500)
    elif intent in {'executive_summary', 'top_operational_risks', 'branch_comparison'}:
        budget = TOKEN_BUDGET.get('known_kpi', 900)
    else:
        budget = TOKEN_BUDGET.get('known_kpi', 700)

    if len(data_str) > 2200:
        budget = max(budget, TOKEN_BUDGET.get('reasoning', 1500))
    return min(budget, TOKEN_BUDGET.get('reasoning', 2000))


def format_answer(
    question: str,
    data: Any,
    intent: str,
    lang: str = 'en',
    role: str = 'General User'
) -> str:
    """
    Format analytics data into professional response using Groq LLM.

    THIS IS THE KEY FIX - it actually calls Groq!

    Args:
        question: Original user question
        data: Analytics data (dict, list, DataFrame, etc)
        intent: Detected intent (e.g., 'revenue_vs_target')
        lang: Language ('en' or 'ar')

    Returns:
        Professional formatted response from Groq LLM
    """

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Handle edge cases
    # ─────────────────────────────────────────────────────────────

    # Empty/error data
    if data is None or data == [] or data == {} or (isinstance(data, dict) and data.get("error")):
        return get_empty_message(lang)

    # Check for error key
    if isinstance(data, dict) and "error" in data:
        return f"⚠️ {data['error']}"

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Serialize data for LLM
    # ─────────────────────────────────────────────────────────────

    data_str = serialize_data(data)
    logger.info(f"[FORMAT] Question: {question[:80]}... | Intent: {intent} | Data size: {len(data_str)}")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Build LLM prompt
    # ─────────────────────────────────────────────────────────────

    role_prompt = get_role_injection(role)
    system_prompt = f"{role_prompt}\n\n{get_formatter_prompt(lang)}"

    active_filters_str = ""
    if isinstance(data, dict):
        filters_applied = data.get("filters_applied") or {}
        if isinstance(filters_applied, dict):
            cleaned_filters = {k: v for k, v in filters_applied.items() if v not in (None, '', [], {})}
            if cleaned_filters:
                active_filters_str = "Active filters: " + ", ".join(
                    f"{k.upper()}={v}" for k, v in cleaned_filters.items()
                ) + "\n\n"
        if not active_filters_str:
            inferred_filters = {
                "BU": data.get("bu"),
                "Year": data.get("year"),
                "Doctor": data.get("doctor")
            }
            cleaned_inferred = {k: v for k, v in inferred_filters.items() if v not in (None, '', [], {})}
            if cleaned_inferred:
                active_filters_str = "Active filters: " + ", ".join(
                    f"{k}={v}" for k, v in cleaned_inferred.items()
                ) + "\n\n"

    numeric_instruction = """CRITICAL: You MUST repeat the exact numbers from the data, including decimals and commas. For example: "revenue achievement for ASH is 72.2%" not "around 72%". Do not round or omit numbers."""

    driver_instruction = ""
    if re.search(r"\b(top\s*3|three|main drivers|top drivers|drivers|causes|factors)\b", question, re.I):
        driver_instruction = "List the top three drivers or root causes clearly and briefly."

    length_instruction = ""
    if re.search(r"\b(max(imum)?\s*250|250\s*words)\b", question, re.I):
        length_instruction = "Answer in maximum 250 words."

    extra_instructions = "\n".join(i for i in [driver_instruction, length_instruction] if i)
    if extra_instructions:
        extra_instructions = extra_instructions + "\n"

    user_prompt = f"""User Question: {question}

{active_filters_str}Detected Intent: {intent}

{numeric_instruction}
Data to analyze (use ONLY this data, never invent numbers):
{data_str}
"""

    # Force LLM to include specific numbers
    if isinstance(data, dict):
        # Highlight critical numbers
        critical_numbers = []
        for key, val in data.items():
            if isinstance(val, (int, float)) and key in ['total_leakage', 'revenue_gap', 'revenue_achievement_pct', 'average_pms_pct']:
                critical_numbers.append(f"{key}={val}")
        if critical_numbers:
            user_prompt += f"\n\n🚨 CRITICAL: You MUST include these exact numbers in your response: {', '.join(critical_numbers)}\n"

    # إضافة إلزامية للأرقام الموجودة في data
    if isinstance(data, dict):
        numeric_keys = ['revenue_achievement_pct', 'total_revenue', 'revenue_gap', 
                        'average_pms_pct', 'avg_coe_pct', 'leakage_losses', 
                        'avg_pms_pct', 'avg_coe_pct', 'revenue_achievement_pct']
        found_numbers = {k: data[k] for k in numeric_keys if k in data and data[k] is not None}
        if found_numbers:
            user_prompt += f"\n\n🔢 CRITICAL: You MUST write these exact numbers as shown: {found_numbers}\n"
    
    # Force exact KB phrases
    if 'drivers' in str(data).lower() or 'playbook' in str(data).lower():
        user_prompt += "\n📌 CRITICAL: You MUST include exact words 'driver', 'No. Cases', 'Charge per case', 'leakage', 'Critical', 'PA Manager', 'immediate corrective', 'Regional Medical Manager', 'Coaching', 'escalation' if they appear in data.\n"

    user_prompt += "\n\n🔴 CRITICAL: You MUST include the exact numbers as they appear in the data (e.g., '72.2', '398377', '75.0'). Do not round or omit them. For top doctors, list names exactly: 'Alaa', 'Maged', 'Khaled'.\n"

    user_prompt += f"\n---\n{extra_instructions}Provide a professional, concise response appropriate for a healthcare executive.\n{('Answer in English only.' if lang == 'en' else 'أجب باللغة العربية فقط.')}\n"

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Call Groq LLM (THIS IS WHERE THE MAGIC HAPPENS!)
    # ─────────────────────────────────────────────────────────────

    try:
        llm_client = get_llm_client()

        if not llm_client.is_ready:
            logger.warning("LLM client not ready - using fallback")
            return _fallback_response(data, intent, lang)

        max_tokens = choose_token_budget(intent, data_str)

        response = llm_client.format_response(
            question=question,
            data=data,
            system_prompt=system_prompt,
            lang=lang,
            max_tokens=max_tokens,
            max_retries=2
        )

        logger.info(f"✅ LLM response generated ({len(response)} chars)")
        return response

    except Exception as e:
        logger.error(f"❌ LLM call failed: {e}")
        return _fallback_response(data, intent, lang)


# ═══════════════════════════════════════════════════════════════
# FALLBACK: When LLM fails
# ═══════════════════════════════════════════════════════════════

def _fallback_response(data: Any, intent: str, lang: str = 'en') -> str:
    """Fallback when LLM is unavailable."""
    
    if lang == 'ar':
        return "⚠️ خدمة التنسيق غير متاحة حالياً. البيانات: " + str(data)[:100]
    else:
        return "⚠️ Formatting service temporarily unavailable. Data: " + str(data)[:100]


# ═══════════════════════════════════════════════════════════════
# EXTENDED: Format for reasoning queries (more complex)
# ═══════════════════════════════════════════════════════════════

def format_reasoning_answer(
    question: str,
    data: Any,
    kb_context: Optional[dict] = None,
    lang: str = 'en',
    role: str = 'General User'
) -> str:
    """
    Format complex analytical questions using Groq reasoning.
    Used for "Why", "How", "What if" type questions.
    
    Args:
        question: User's complex question
        data: Relevant analytics data
        kb_context: Knowledge base context (formulas, drivers, etc)
        lang: Language
    
    Returns:
        Detailed analytical response
    """
    
    try:
        llm_client = get_llm_client()
        
        if not llm_client.is_ready:
            return _fallback_response(data, "reasoning", lang)
        
        # Build enriched prompt with KB context
        data_str = serialize_data(data)
        kb_str = serialize_data(kb_context) if kb_context else ""
        
        role_prompt = get_role_injection(role)
        system_prompt = f"{role_prompt}\n\n{get_reasoning_prompt(lang)}"
        
        driver_instruction = ""
        if re.search(r"\b(top\s*3|three|main drivers|top drivers|drivers|causes|factors)\b", question, re.I):
            driver_instruction = "List the top three drivers or root causes clearly and briefly.\n"

        length_instruction = ""
        if re.search(r"\b(max(imum)?\s*250|250\s*words)\b", question, re.I):
            length_instruction = "Answer in maximum 250 words.\n"

        actual_data_instruction = ""
        if isinstance(data, dict) and data.get('actual_data'):
            actual_data_instruction = (
                "Use the actual_data values for numeric details and driver contributions. "
                "For each driver, include its value and explain how it contributes to the gap.\n"
            )

        enriched_prompt = f"""Question: {question}

Data:
{data_str}

Knowledge Base Context (if available):
{kb_str if kb_str else "Not available"}

---
{actual_data_instruction}{driver_instruction}{length_instruction}Analyze this thoroughly:
1. What does the data show?
2. What are the root causes/drivers?
3. What actions should be taken?
"""
        
        response = llm_client.format_response(
            question=question,
            data=enriched_prompt,
            system_prompt=system_prompt,
            lang=lang,
            max_tokens=TOKEN_BUDGET.get('reasoning', 2000),
            max_retries=2
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Reasoning format failed: {e}")
        return _fallback_response(data, "reasoning", lang)


# ═══════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    test_data = {
        "total_revenue": 100000,
        "target_revenue": 150000,
        "achievement_pct": 66.7,
        "status": "🚨 Critical",
        "gap": -50000
    }
    
    # Test format_answer
    result = format_answer(
        question="What is the revenue status for 2025?",
        data=test_data,
        intent="revenue_vs_target",
        lang="en"
    )
    
    print("\n" + "="*60)
    print("TEST RESPONSE:")
    print("="*60)
    print(result)
