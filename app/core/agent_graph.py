# app/core/agent_graph.py
from typing import Any, Dict, List, Optional

from app.core.real_agent import run_agent as _run_real_agent


def run_agent(
    question: str,
    analytics,
    kb,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Compatibility wrapper for older imports.

    The previous implementation used ToolExecutor, which was removed from the
    installed LangGraph version. Keep this module as a stable entry point and
    delegate to the current LangGraph ReAct agent.
    """
    return _run_real_agent(question, analytics, kb, chat_history)
