import re
from typing import Optional, Dict, Any


class ChartAgent:
    """Simple chart intent agent for OPD analytics questions."""

    def __init__(self):
        self.intent_patterns = [
            (r"(compare|comparison|compare.*branches|branches.*compare|branch.*performance|bu.*performance|business unit.*performance)", "branch_comparison"),
            (r"(which|what).*(bu|branch|business unit).*(highest|best|top|largest)", "branch_comparison"),
            (r"(highest|top).*leakage|leakage.*(highest|top)|doctor.*leakage|which doctor.*leakage", "leakage_analysis"),
            (r"(pms ranking|doctor pms|pms score|performance ranking|doctor performance|performance analysis)", "pms_ranking"),
            (r"(coe compliance|compliance ranking|low coe|high coe)", "coe_compliance"),
            (r"(no[- ]show|noshow|no show)", "noshow_analysis"),
            (r"(revenue achievement|revenue vs target|revenue gap|target revenue|revenue achievement percentage)", "revenue_vs_target"),
            (r"(top operational risks|operational risks|risks.*operational)", "top_operational_risks"),
            (r"(which branch is performing best|best branch|top branch)", "branch_comparison"),
            (r"(monthly trend|trend over|over time|yearly trend|monthly revenue)", "monthly_trend"),
            (r"(top doctors|best doctors|worst doctors|doctor ranking)", "dynamic_ranking"),
        ]

    def suggest_chart_schema(
        self,
        question: str,
        current_intent: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a chart schema suggestion based on the question and intent."""
        if current_intent and current_intent not in {"general_question", "kpi_playbook", "kpi_drivers"}:
            return {
                "intent": current_intent,
                "chart_type": self._intent_to_chart_type(current_intent)
            }

        chart_intent = self._choose_intent(question)
        if not chart_intent:
            return None

        return {
            "intent": chart_intent,
            "chart_type": self._intent_to_chart_type(chart_intent)
        }

    def _choose_intent(self, question: str) -> Optional[str]:
        q = question.lower()
        for pattern, intent in self.intent_patterns:
            if re.search(pattern, q, re.I):
                return intent
        return None

    def _intent_to_chart_type(self, intent: str) -> str:
        mapping = {
            "branch_comparison": "bar",
            "leakage_analysis": "horizontal_bar",
            "pms_ranking": "horizontal_bar",
            "coe_compliance": "bar",
            "noshow_analysis": "bar",
            "revenue_vs_target": "gauge",
            "top_operational_risks": "bar",
            "monthly_trend": "line",
            "dynamic_ranking": "horizontal_bar",
            "best_doctor": "horizontal_bar",
            "worst_doctor": "horizontal_bar",
        }
        return mapping.get(intent, "bar")

    def render_chart(self, intent: str, data: Any):
        """Render a chart object using the existing `generate_chart` helper.

        This delegates to `app.ui.charts.generate_chart` when available.
        Returns whatever `generate_chart` returns (e.g., a Plotly figure) or None.
        """
        try:
            # Import lazily to avoid circular imports at module load time
            from app.ui.charts import generate_chart
            return generate_chart(intent, data)
        except Exception:
            return None
