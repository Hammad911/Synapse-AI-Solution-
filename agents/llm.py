from __future__ import annotations

from langchain_groq import ChatGroq

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_llm(*, temperature: float = 0.0) -> ChatGroq:
    """Groq chat model (same ID everywhere for consistent behavior)."""
    return ChatGroq(model=DEFAULT_MODEL, temperature=temperature)
