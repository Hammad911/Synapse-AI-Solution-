from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    """Graph state for the multi-agent business research assistant."""

    messages: list[BaseMessage]
    query: str
    clarity_status: str
    research_findings: str
    confidence_score: int
    validation_result: str  # "sufficient" | "insufficient"
    validation_reason: str  # validator explanation; steers research retries
    research_attempts: int
    final_response: str
    clarification_question: str
