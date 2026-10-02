"""
app/utils/language.py - Language detection (Arabic / English)
"""

def detect_language(text: str) -> str:
    """Detects whether text is Arabic or English."""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
    if arabic_chars / max(len(text), 1) > 0.2:
        return 'ar'
    try:
        from langdetect import detect
        return 'ar' if detect(text) == 'ar' else 'en'
    except Exception:
        return 'en'