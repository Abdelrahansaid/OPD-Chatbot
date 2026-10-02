"""
app/config.py - All constants, paths, and shared configuration
Phase 1: Clean Architecture + Groq + Memory
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OPD_PATH = DATA_DIR / "OPD dataset.xlsx"
KB_PATH = DATA_DIR / "Knowledge base.xlsx"
VECTOR_DB_PATH = BASE_DIR / "vector_db"

# ═══════════════════════════════════════════════════════════════# HUGGING FACE / RAG CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
HUGGINGFACE_HUB_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
RAG_ENABLED = os.getenv("RAG_ENABLED", "0").strip().lower() not in ("0", "false", "no", "off")
if not HUGGINGFACE_HUB_TOKEN:
    RAG_ENABLED = False

# ═══════════════════════════════════════════════════════════════════════# GROQ LLM CONFIGURATION
# ═══════════════════════════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant" #llama-3.3-70b-versatile
GROQ_TIMEOUT = 30  # seconds

# Dataverse integration (optional). If any value is missing, unanswered-request
# capture falls back to data/pending_requests.json.
DATAVERSE_URL = os.getenv("DATAVERSE_URL", "").rstrip("/")
DATAVERSE_TENANT_ID = os.getenv("DATAVERSE_TENANT_ID", "")
DATAVERSE_CLIENT_ID = os.getenv("DATAVERSE_CLIENT_ID", "")
DATAVERSE_CLIENT_SECRET = os.getenv("DATAVERSE_CLIENT_SECRET", "")
DATAVERSE_REQUESTS_TABLE = os.getenv("DATAVERSE_REQUESTS_TABLE", "opd_chatbot_requests")

# Preferred request sync path: Power Automate HTTP trigger.
# The chatbot always saves requests locally first, then posts to this URL if set.
POWER_AUTOMATE_FLOW_URL = os.getenv("POWER_AUTOMATE_FLOW_URL", "").strip()

# ═══════════════════════════════════════════════════════════════
# TOKEN BUDGETS (لـ intelligent token allocation)
# ═══════════════════════════════════════════════════════════════
TOKEN_BUDGET = {
    "simple": 400,         # KPI definition / formula
    "known_kpi": 600,      # Known KPI from hardcoded analytics
    "rag_answer": 1200,    # RAG semantic search
    "analytical": 1500,    # Multi-step analysis
    "reasoning": 2000,     # Complex reasoning + drivers
}

# ═══════════════════════════════════════════════════════════════
# BUSINESS UNIT LABELS & REGIONS
# ═══════════════════════════════════════════════════════════════
BU_LABELS = {
    "ASH": "El Shalalat - Alexandria",
    "SMH": "Smouha - Alexandria",
    "HJH": "Hay El Gamea - Saudi Arabia",
}

BU_REGIONS = {
    "ASH": "Egypt",
    "SMH": "Egypt",
    "HJH": "Saudi Arabia",
}

BU_ALL = list(BU_LABELS.keys())

# ═══════════════════════════════════════════════════════════════
# KPI THRESHOLDS (لـ status indicators)
# ═══════════════════════════════════════════════════════════════
THRESHOLDS = {
    "revenue_warning": 90,
    "revenue_critical": 70,
    "coe_warning": 80,
    "coe_critical": 60,
    "pms_warning": 80,
    "pms_critical": 60,
    "noshow_warning": 15,
    "noshow_critical": 25,
    "leakage_pct_warning": 4,
}

# KPI Aliases (normalized name lookup)
KPI_ALIASES = {
    "total revenue": "Total Revenue",
    "revenue": "Total Revenue",
    "revenues": "Total Revenue",
    "target revenue": "Target Revenue",
    "credit revenue": "Credit Revenue",
    "credit": "Credit Revenue",
    "cash revenue": "Cash Revenue",
    "cash": "Cash Revenue",
    "no cases": "No. Cases",
    "number of cases": "No. Cases",
    "cases": "No. Cases",
    "num cases": "No. Cases",
    "target no cases": "Target No. cases",
    "target cases": "Target No. cases",
    "target no. cases": "Target No. cases",
    "no services": "No. Services",
    "services": "No. Services",
    "no booking": "No. Booking",
    "booking": "No. Booking",
    "bookings": "No. Booking",
    "planned slots": "No. Planned booking Slots",
    "planned booking slots": "No. Planned booking Slots",
    "service leakage": "Service Leakage %",
    "leakage": "Total Leakage Revenue Losses",
    "leakage losses": "Total Leakage Revenue Losses",
    "revenue leakage": "Total Leakage Revenue Losses",
    "total leakage": "Total Leakage Revenue Losses",
    "doctor pms": "Doctor PMS %",
    "pms": "Doctor PMS %",
    "pms score": "Doctor PMS %",
    "coe": "Actual COE Compliance %",
    "coe compliance": "Actual COE Compliance %",
    "compliance": "Actual COE Compliance %",
    "actual coe": "Actual COE Compliance %",
    "no show": "No-Show %",
    "noshow": "No-Show %",
    "no-show": "No-Show %",
    "no show %": "No-Show %",
    "digital cr": "Digital Actual CR%",
    "digital actual cr": "Digital Actual CR%",
    "digital target cr": "Digital Target CR%",
    "digital leads": "Digital Actual CR%",
    "cross referral": "Cross Referral %",
    "retention": "Patient Retention %",
    "acquisition": "Patient Acquisition %",
    "charge per case": "Charge per case",
    "charge": "Charge per case",
    "missed opportunity": "No. Missed Opportunity",
    "missed opportunities": "No. Missed Opportunity",
    "cancelled clinics": "No. Cancelled Clinics",
    "cancellation": "Total Losses Revenue_Cancellation_Modification",
    "follow up": "No. follow-up visits",
    "follow-up": "No. follow-up visits",
    "followup": "No. follow-up visits",
}

# Intent → Analytics method mapping
RANKING_KPI_MAP = {
    "revenue": ("total_revenue", True, "Total Revenue"),
    "credit": ("credit_revenue", True, "Credit Revenue"),
    "cash": ("cash_revenue", True, "Cash Revenue"),
    "pms": ("avg_pms_pct", True, "PMS Score %"),
    "coe": ("avg_coe_pct", True, "COE Compliance %"),
    "compliance": ("avg_coe_pct", True, "COE Compliance %"),
    "leakage": ("total_leakage", False, "Leakage Losses"),
    "noshow": ("avg_noshow_pct", False, "No-Show %"),
    "composite": ("excellence_score", True, "Composite Excellence Score"),
    "overall": ("excellence_score", True, "Composite Excellence Score"),
}

# Doctor name aliases (Arabic → English)
DOCTOR_ALIASES = {
    "محمود": "Mahmoud",
    "محمد": "Mohamed",
    "احمد": "Ahmed",
    "أحمد": "Ahmed",
    "خالد": "Khaled",
    "يوسف": "Youssef",
    "عمر": "Omar",
    "امانى": "Amany",
    "أماني": "Amany",
    "أمانى": "Amany",
    "علاء": "Alaa",
    "ماجد": "Maged",
    "مجد": "Maged",
    "علا": "Ola",
    "اولا": "Ola",
    "علي": "Aly",
    "على": "Aly",
}

# User Roles
ROLES = [
    "General User",
    "OPD Manager",
    "OPD Coordinator",
    "CEO",
    "Operational Manager",
    "HR Analysis",
    "Doctor Staff",
]

DEBUG_MODE = True
