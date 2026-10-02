"""
app/core/memory.py — Conversation context & filter persistence
Supports Arabic/English pronoun resolution, filter inheritance, and context summarization.
"""
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from app.utils.filters import sanitize_filters  # Reuse your existing sanitizer


@dataclass
class ConversationTurn:
    """Immutable record of a single conversation exchange."""
    timestamp: datetime
    user_message: str
    bot_response: str
    intent: Optional[str] = None
    extracted_filters: Dict[str, Any] = field(default_factory=dict)
    resolved_context: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """
    Manages short-term conversation state for follow-up understanding.
    
    Capabilities:
    ✓ Tracks last N turns with timestamps
    ✓ Maintains active filter state (doctor, bu, year, month, kpi)
    ✓ Resolves Arabic/English pronouns: هو، هي، ده، دي، him, her, it, that
    ✓ Merges implicit context with explicit new filters
    ✓ Generates compact context summaries for LLM prompts
    """
    
    # Pronouns that refer to previously mentioned entities
    ARABIC_PRONOUNS = {'هو', 'هي', 'ده', 'دي', 'دول', 'اللي فات', 'التاني', 'الفرع ده', 'الدكتور ده'}
    ENGLISH_PRONOUNS = {'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'that', 'the previous', 'the last'}

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.active_filters: Dict[str, Any] = {}  # Currently applied filters
        self.last_entities: Dict[str, Optional[str]] = {
            'doctor': None,
            'branch': None,   # BU code: ASH/SMH/HJH
            'kpi': None,      # Canonical KPI name
            'year': None,
            'month': None
        }
    
    def add_turn(self, user_msg: str, bot_response: str, 
                 intent: Optional[str] = None, 
                 filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a new conversation turn and update context state.
        
        Args:
            user_msg: Raw user question
            bot_response: Generated answer
            intent: Detected intent label (e.g., 'revenue_vs_target')
            filters: Extracted filters from the question
        """
        # Sanitize and store filters
        clean_filters = sanitize_filters(filters or {})
        
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_msg.strip(),
            bot_response=bot_response.strip(),
            intent=intent,
            extracted_filters=clean_filters,
            resolved_context=self.last_entities.copy()
        )
        
        self.turns.append(turn)
        
        # Trim history to max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        
        # Update active filters and last-mentioned entities
        self.active_filters.update(clean_filters)
        self._update_last_entities(clean_filters)
    
    def _update_last_entities(self, filters: Dict[str, Any]) -> None:
        """Update the 'last mentioned' tracker from new filters."""
        entity_mapping = {
            'doctor': 'doctor',
            'bu': 'branch', 
            'kpi': 'kpi',
            'year': 'year',
            'month': 'month'
        }
        for filter_key, entity_key in entity_mapping.items():
            if filter_key in filters and filters[filter_key]:
                self.last_entities[entity_key] = filters[filter_key]
    
    def resolve_context(self, question: str) -> Dict[str, Any]:
        """
        Resolve implicit references in a new question using conversation history.
        
        Examples:
        - "How is he performing?" → {doctor: last_doctor}
        - "إزاي أداؤه في 2024؟" → {doctor: last_doctor, year: 2024}
        - "What about that branch?" → {bu: last_branch}
        
        Returns:
            Dictionary with resolved entities to merge with newly extracted filters.
        """
        resolved: Dict[str, Any] = {}
        q_lower = question.lower().strip()
        
        # 1. Determine whether the new question explicitly refers to prior context.
        def contains_any(phrases):
            return any(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q_lower, flags=re.I) for phrase in phrases)

        has_pronoun = contains_any(self.ARABIC_PRONOUNS | self.ENGLISH_PRONOUNS)
        explicit_doctor_ref = contains_any({
            'الدكتور ده', 'دكتور التاني', 'الدكتور اللي فات',
            'his', 'her', 'him', 'she', 'he', 'their', 'that doctor', 'the doctor'
        })
        explicit_branch_ref = contains_any({
            'الفرع ده', 'الفرع اللي فات', 'فرع التاني',
            'this branch', 'that branch', 'the branch', 'same branch'
        })

        if explicit_doctor_ref or explicit_branch_ref or has_pronoun:
            if self.last_entities['doctor'] and (explicit_doctor_ref or has_pronoun):
                resolved['doctor'] = self.last_entities['doctor']
            if self.last_entities['branch'] and (explicit_branch_ref or has_pronoun):
                resolved['bu'] = self.last_entities['branch']

            if self.last_entities['year'] and has_pronoun:
                resolved['year'] = self.last_entities['year']
            if self.last_entities['month'] and has_pronoun:
                resolved['month'] = self.last_entities['month']

        return resolved
    
    def get_context_summary(self, max_turns: int = 3) -> str:
        """
        Generate a compact text summary of recent conversation for LLM context injection.
        
        Format:
        [Turn 1] User: ... | Bot: ...
        [Turn 2] ...
        Active filters: doctor=X, bu=Y, year=Z
        """
        if not self.turns:
            return "No prior conversation context."
        
        lines = ["[Conversation Context]"]
        for i, turn in enumerate(self.turns[-max_turns:], 1):
            user_snippet = turn.user_message[:80] + ("..." if len(turn.user_message) > 80 else "")
            bot_snippet = turn.bot_response[:80] + ("..." if len(turn.bot_response) > 80 else "")
            lines.append(f"{i}. Q: {user_snippet} | A: {bot_snippet}")
        
        # Append active filters
        active = {k: v for k, v in self.active_filters.items() if v}
        if active:
            filters_str = ", ".join(f"{k}={v}" for k, v in active.items())
            lines.append(f"Active filters: {filters_str}")
        
        return "\n".join(lines)
    
    def get_active_filters(self) -> Dict[str, Any]:
        """Return a safe copy of currently active filters."""
        return self.active_filters.copy()
    
    def clear(self) -> None:
        """Reset all memory state."""
        self.turns.clear()
        self.active_filters.clear()
        self.last_entities = {k: None for k in self.last_entities}
    
    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self.turns)}, filters={self.active_filters})"


# Singleton instance for global use
conversation_memory = ConversationMemory(max_turns=8)