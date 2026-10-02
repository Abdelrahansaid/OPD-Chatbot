"""
app/utils/guards.py - Guards, scope validation, identity responses
"""
import re
from app.config import BU_LABELS, ROLES

IDENTITY_PATTERNS = [
    r"^(انت|انت مين|مين انت|من انت)\b",
    r"^(who are you|what are you|hello\b|hi\b|hey\b|مرحبا|أهلا|هلو)\b",
    r"^(ازيك|عامل ايه|كيف حالك)\b",
]

IDENTITY_EN = (
    "👋 I'm the **Andalusia Hospital KPI Assistant**.\n\n"
    "I analyze 24 OPD KPIs across 3 branches (ASH, SMH, HJH) for 2023–2025.\n\n"
    "**I can help with:**\n"
    "• Doctor performance (PMS, COE, Revenue, Leakage)\n"
    "• Branch comparison and operational risks\n"
    "• KPI formulas, drivers, and investigation playbooks\n"
    "• Executive summaries\n\n"
    "**I cannot help with:** forecasting, patient-level data, insurance rejections, specialty breakdowns."
)

IDENTITY_AR = (
    "👋 أنا **مساعد KPI لمستشفيات أندلسيا**.\n\n"
    "أحلل 24 KPI لعيادات OPD في 3 فروع (ASH, SMH, HJH) للفترة 2023-2025.\n\n"
    "**أقدر أساعدك في:** أداء الدكاترة | مقارنة الفروع | معادلات KPI | أسباب الأداء | الملخصات التنفيذية\n\n"
    "**خارج نطاقي:** التنبؤ | بيانات المريض الفردي | رفض التأمين | تقسيم التخصصات"
)

SCOPE_MSG_EN = (
    "I'm specialized in OPD KPI analytics for Andalusia Hospitals.\n"
    "I can answer questions about:\n"
    "• Revenue, Leakage, PMS, COE, No-Show, Booking, Cases, Digital CR\n"
    "• Doctor and Branch performance\n"
    "• KPI formulas, drivers, and investigation scenarios\n\n"
    "Please ask about a specific KPI, doctor, or branch."
)

SCOPE_MSG_AR = (
    "أنا متخصص في تحليل KPIs لعيادات OPD في أندلسيا.\n"
    "أجيب عن أسئلة خاصة بـ:\n"
    "• الإيرادات، Leakage، PMS، COE، No-Show، الحجوزات، الحالات\n"
    "• أداء الدكاترة والفروع\n"
    "• معادلات KPI وأسبابها وسيناريوهات التحقيق\n\n"
    "من فضلك اسأل عن KPI أو دكتور أو فرع محدد."
)


def check_guards(question: str, lang: str, available_years=None) -> str | None:
    q = question.strip().lower()

    # Identity patterns -> short intro
    for p in IDENTITY_PATTERNS:
        if re.search(p, q, re.IGNORECASE | re.UNICODE):
            return IDENTITY_EN if lang == 'en' else IDENTITY_AR

    # Non-KPI chatter: allow most KPI questions through, block clearly unrelated topics
    non_kpi = r"\b(cook|sing|dance|movie|song|play|game|football|sports|weather|fuck|shit|damn)\b"
    if re.search(non_kpi, q, re.I):
        if lang == 'en':
            return "I'm specialized in OPD KPI analytics. Please ask about revenue, doctors, branches, or KPIs."
        else:
            return "أنا متخصص في تحليل KPIs. من فضلك اسأل عن الإيرادات، الدكاترة، أو الفروع."

    # Forecasting/out-of-range predictions are explicitly out-of-scope
    if re.search(r"\b(predict|forecast|next month|next year|تنبؤ|توقع)\b", q, re.I):
        return "⚠️ Forecasting is outside scope. I analyze historical data (2023–2025) only." if lang == 'en' else "⚠️ التنبؤ خارج نطاق هذا النظام."

    # Otherwise, allow the Agent to handle (do not block legitimate KPI questions)
    return None


def is_in_scope(question: str, extra_kpis: list = None) -> bool:
    return True
