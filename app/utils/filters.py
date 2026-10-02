"""
app/utils/filters.py - Filter helpers & KPI utilities
"""
import re
from difflib import get_close_matches
from typing import List
from app.config import KPI_ALIASES, DOCTOR_ALIASES

INVALID_VALS = {'UNKNOWN', 'ALL', 'NONE', 'N/A', 'NULL', ''}

def normalize_kpi_name(text: str) -> str:
    """Convert any KPI text to its canonical KB name."""
    if not text:
        return text
    cleaned = text.lower().strip()
    cleaned = re.sub(r'[.-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return KPI_ALIASES.get(cleaned, text)

def resolve_doctor(name: str) -> str:
    """Remove Arabic/English prefixes and resolve aliases."""
    if not name:
        return name
    name = str(name).strip()
    name = re.sub(r'^(الدكتور|دكتور|dr.?\s*)', '', name, flags=re.I).strip()
    return DOCTOR_ALIASES.get(name, name)


def resolve_doctor_fuzzy(name: str, available_doctors: List[str]) -> str:
    """Try exact match first, then fuzzy match doctor names."""
    if not name:
        return name
    resolved = resolve_doctor(name)
    candidates = [str(d).strip() for d in available_doctors if str(d).strip()]
    if resolved in candidates:
        return resolved

    matches = get_close_matches(resolved, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else resolved


def sanitize_filters(filters: dict) -> dict:
    """Remove null / invalid filter values."""
    if not isinstance(filters, dict):
        return {}
    return {
        k: v for k, v in filters.items()
        if v is not None and str(v).strip().upper() not in INVALID_VALS
    }

def extract_year_month(question: str) -> dict:
    """Extract year and month from question text."""
    filters = {}
    year_m = re.search(r'\b(202[3-5])\b', question)
    if year_m:
        filters['year'] = int(year_m.group(1))
    
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'يناير': 1, 'فبراير': 2, 'مارس': 3, 'أبريل': 4, 'مايو': 5,
        'يونيو': 6, 'يوليو': 7, 'أغسطس': 8, 'سبتمبر': 9,
        'أكتوبر': 10, 'نوفمبر': 11, 'ديسمبر': 12,
    }
    q_lower = question.lower()
    for mname, num in month_map.items():
        if re.search(r'\b' + re.escape(mname) + r'\b', q_lower):
            filters['month'] = num
            break
    return filters

def extract_bu_from_question(question: str):
    """Extract BU code from question."""
    if re.search(r'\bASH\b|شلالت|الشلالات', question, re.I):
        return 'ASH'
    if re.search(r'\bSMH\b|سموحة|سموحه', question, re.I):
        return 'SMH'
    if re.search(r'\bHJH\b|حي\s*الجامع|السعودي', question, re.I):
        return 'HJH'
    return None

def extract_kpi_from_question(question: str) -> str:
    """Extract a canonical KPI name from question text (pattern-based)."""
    q_lower = question.lower()

    # Early detection for conditional/threshold questions
    # "If Doctor PMS % is below 80%" → "Doctor PMS %"
    # "If No. Cases falls below 85%" → "No. Cases"
    conditional_patterns = [
        (r"if\s+(doctor\s+pms\s*%?)", "Doctor PMS %"),
        (r"if\s+(no[\.\s]+cases?\s*)", "No. Cases"),
        (r"if\s+(service\s+leakage\s*%?)", "Service Leakage %"),
        (r"if\s+(actual\s+coe|coe\s+compliance)", "Actual COE Compliance %"),
        (r"if\s+(no[\-\s]+show\s*%?)", "No-Show %"),
        (r"if\s+(total\s+revenue)", "Total Revenue"),
        (r"if\s+(no[\.\s]+booking)", "No. Booking"),
        (r"لو\s+(doctor\s+pms|بي\s*ام\s*اس)", "Doctor PMS %"),
        (r"لو\s+(الإيرادات|إيرادات)", "Total Revenue"),
    ]
    for pattern, kpi in conditional_patterns:
        if re.search(pattern, q_lower, re.I):
            return kpi

    kpi_patterns = [
        (r"no[\.\s]+booking\b", "No. Booking"),
        (r"no[\.\s]+planned\s+booking", "No. Planned booking Slots"),
        (r"no[\.\s]+follow[\s\-]+up", "No. follow-up visits"),
        (r"no[\.\s]+missed\s+opportunity", "No. Missed Opportunity"),
        (r"no[\.\s]+cancelled\s+clinic", "No. Cancelled Clinics"),
        (r"no[\.\s]+services?\b", "No. Services"),
        (r"target\s+no[\.\s]+cases?", "Target No. cases"),
        (r"no[\.\s]+cases?\b", "No. Cases"),
        (r"total\s+leakage\s+revenue", "Total Leakage Revenue Losses"),
        (r"service\s+leakage\s*%?", "Service Leakage %"),
        (r"leakage\s+losses?", "Total Leakage Revenue Losses"),
        (r"\bleakage\b", "Total Leakage Revenue Losses"),
        (r"doctor\s+pms\s*%?", "Doctor PMS %"),
        (r"\bpms\b", "Doctor PMS %"),
        (r"actual\s+coe\s+compliance", "Actual COE Compliance %"),
        (r"\bcoe\b", "Actual COE Compliance %"),
        (r"no[\-\s]+show\s*%?", "No-Show %"),
        (r"digital\s+actual\s+cr", "Digital Actual CR%"),
        (r"digital\s+target\s+cr", "Digital Target CR%"),
        (r"digital\s+(leads?|cr|conversion)", "Digital Actual CR%"),
        (r"charge\s+per\s+case", "Charge per case"),
        (r"cross\s+referral", "Cross Referral %"),
        (r"patient\s+retention", "Patient Retention %"),
        (r"patient\s+acquisition", "Patient Acquisition %"),
        (r"credit\s+revenue", "Credit Revenue"),
        (r"cash\s+revenue", "Cash Revenue"),
        (r"target\s+revenue", "Target Revenue"),
        (r"total\s+revenue", "Total Revenue"),
        (r"\brevenue\b", "Total Revenue"),
    ]
    for pattern, kpi in kpi_patterns:
        if re.search(pattern, q_lower):
            return kpi
    return ""