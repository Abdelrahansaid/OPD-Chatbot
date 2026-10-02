"""
app/core/llm_client.py - Groq LLM client wrapper with retry logic & error handling
This fixes the format_answer issue!
"""
import logging
from typing import Optional, List, Dict, Any
from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# GROQ CLIENT WRAPPER
# ═══════════════════════════════════════════════════════════════

class GroqLLMClient:
    """Wrapper around Groq API with retry logic and error handling."""
    
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model name (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        self.is_ready = False
        
        # Test connection
        try:
            self._test_connection()
            self.is_ready = True
            logger.info(f"✅ Groq client ready ({model})")
        except Exception as e:
            logger.error(f"❌ Groq connection failed: {e}")
            self.is_ready = False
    
    def _test_connection(self) -> bool:
        """Test if Groq API is accessible."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Reply with: OK"}],
            temperature=0,
            max_tokens=10,
            timeout=GROQ_TIMEOUT
        )
        return response.choices[0].message.content.strip() == "OK"
    
    def format_response(
        self,
        question: str,
        data: Any,
        system_prompt: str,
        lang: str = 'en',
        max_tokens: int = 700,
        max_retries: int = 2
    ) -> str:
        """
        Format data into a professional response using Groq LLM.
        
        Args:
            question: Original user question
            data: Data to format (dict, list, or string)
            system_prompt: System instructions for the LLM
            lang: Language ('en' or 'ar')
            max_tokens: Maximum tokens for response
            max_retries: Number of retries on failure
        
        Returns:
            Formatted response string
        """
        if not self.is_ready:
            return f"⚠️ LLM connection unavailable. Raw data: {str(data)[:200]}"
        
        # Convert data to string
        import json
        data_str = self._serialize_data(data)
        
        # Build user prompt
        user_prompt = f"""Question: {question}
Language: {lang.upper()}
Data to analyze:
{data_str}

Provide a professional response."""
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Low temperature for consistency
                    max_tokens=max_tokens,
                    timeout=GROQ_TIMEOUT
                )
                
                result = response.choices[0].message.content.strip()
                logger.info(f"✅ Groq response generated (attempt {attempt+1})")
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    # Last attempt failed
                    return self._fallback_response(data, lang)
                continue
        
        return "⚠️ Formatting service temporarily unavailable."
    
    def _serialize_data(self, obj: Any, max_len: int = 3000) -> str:
        """Convert any object to JSON string with truncation."""
        import json
        import numpy as np
        import pandas as pd
        
        def convert_types(o):
            if isinstance(o, (np.integer, np.int64)):
                return int(o)
            elif isinstance(o, (np.floating, np.float64)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif pd.isna(o):
                return None
            elif isinstance(o, (dict, list)):
                return o
            return str(o)
        
        try:
            data_converted = json.loads(
                json.dumps(obj, default=convert_types, ensure_ascii=False)
            )
            data_str = json.dumps(data_converted, ensure_ascii=False, indent=2)
            
            # Truncate if too long
            if len(data_str) > max_len:
                data_str = data_str[:max_len] + "\n... (truncated)"
            
            return data_str
        except Exception as e:
            return str(obj)[:max_len]
    
    def _fallback_response(self, data: Any, lang: str = 'en') -> str:
        """Fallback response when LLM fails."""
        if lang == 'ar':
            return "⚠️ خدمة التنسيق مؤقتاً غير متاحة. يرجى المحاولة لاحقاً."
        else:
            return "⚠️ Formatting service temporarily unavailable. Please try again later."
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 700,
        temperature: float = 0.3
    ) -> str:
        """
        Low-level chat interface.
        
        Args:
            messages: List of {'role': 'user'/'system', 'content': '...'} dicts
            max_tokens: Maximum response tokens
            temperature: Response randomness (0-1)
        
        Returns:
            Response text
        """
        if not self.is_ready:
            raise RuntimeError("Groq client not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=GROQ_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Chat call failed: {e}")
            raise


# ═══════════════════════════════════════════════════════════════
# GLOBAL CLIENT INSTANCE
# ═══════════════════════════════════════════════════════════════

_llm_client: Optional[GroqLLMClient] = None


def get_llm_client() -> GroqLLMClient:
    """Get or create the global Groq LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = GroqLLMClient()
    return _llm_client


def initialize_llm(api_key: str = None, model: str = None) -> GroqLLMClient:
    """Initialize the LLM client with custom settings."""
    global _llm_client
    _llm_client = GroqLLMClient(
        api_key=api_key or GROQ_API_KEY,
        model=model or GROQ_MODEL
    )
    return _llm_client
