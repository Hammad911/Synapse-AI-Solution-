from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.context import transcript_block
from agents.llm import get_llm
from json_utils import parse_llm_json
from state import ResearchState
from tools.search import run_tavily_search


def _craft_search_plan(state: ResearchState) -> dict[str, Any]:
    """Ask the LLM for a tight Tavily query + topic; improves hit rate vs raw user text."""
    llm = get_llm(temperature=0)
    prior = int(state.get("research_attempts") or 0)
    feedback = (state.get("validation_reason") or "").strip()
    query = state.get("query") or ""
    history = transcript_block(state, tail=14)

    retry_hint = ""
    if prior > 0 and feedback:
        retry_hint = (
            f"\nPrior research was judged incomplete. Validator note: {feedback}\n"
            "Craft a DIFFERENT or broader search query (new angles, synonyms, or "
            "add 'annual report', 'SEC filing', 'earnings', 'CEO', etc. as appropriate).\n"
        )
    elif prior > 0:
        retry_hint = (
            "\nThis is a follow-up research pass — broaden keywords or add time-sensitive "
            "terms (e.g. '2024', 'latest') if useful.\n"
        )

    system = SystemMessage(
        content=(
            "You write optimal web search queries for Tavily about businesses.\n"
            "Output ONLY raw JSON with keys:\n"
            '  "search_query": string (under 380 chars, concrete entity names, disambiguation when needed),\n'
            '  "topic": one of "general", "news", "finance" — pick "news" for breaking/recent events, '
            '"finance" for earnings, stock, filings, valuation; else "general".\n'
            "No markdown, no explanation."
        )
    )
    human = HumanMessage(
        content=(
            f"Conversation context:\n{history}\n\n"
            f"Current user focus:\n{query}\n"
            f"{retry_hint}"
        )
    )
    raw = llm.invoke([system, human]).content
    if isinstance(raw, list):
        raw = str(raw)
    plan = parse_llm_json(raw)
    sq = str(plan.get("search_query") or query).strip()[:400]
    topic = plan.get("topic")
    if topic not in ("general", "news", "finance"):
        topic = "general"
    return {"search_query": sq or query, "topic": topic}


def research_agent(state: ResearchState) -> dict:
    print("[Research Agent] Planning search, then retrieving sources...")
    prior_attempts = int(state.get("research_attempts") or 0)
    plan = _craft_search_plan(state)
    search_q = plan["search_query"]
    topic = plan["topic"]

    depth = "basic" if prior_attempts == 0 else "advanced"
    raw_findings = run_tavily_search(
        search_q,
        max_results=7 if depth == "advanced" else 6,
        search_depth=depth,
        topic=topic,
        include_answer=True,
    )

    llm = get_llm(temperature=0)
    query = state.get("query") or ""
    history_tail = transcript_block(state, tail=8)

    system = SystemMessage(
        content=(
            "You evaluate search results for a business Q&A assistant.\n"
            "Score how well the SNIPPETS (and Tavily summary if present) let you answer the user: "
            "coverage, recency where relevant, factual specificity, alignment with the question.\n"
            "Return ONLY raw JSON: "
            '{"confidence_score": <int 0-10>, '
            '"summary": "<bullet-style notes of the strongest facts; cite years/units when present>"}\n'
            "Penalize empty, off-topic, or purely generic hits. No markdown."
        )
    )
    human = HumanMessage(
        content=(
            f"User query:\n{query}\n\n"
            f"Recent conversation (for follow-up context):\n{history_tail}\n\n"
            f"Planned search string used:\n{search_q}\n\n"
            f"Raw findings:\n{raw_findings[:14000]}"
        )
    )
    raw = llm.invoke([system, human]).content
    if isinstance(raw, list):
        raw = str(raw)
    parsed = parse_llm_json(raw)
    score = int(parsed.get("confidence_score", 0))
    score = max(0, min(10, score))
    summary = (parsed.get("summary") or "").strip()
    compiled = f"{summary}\n\n--- Sources ---\n{raw_findings}"

    attempts = prior_attempts + 1
    print(f"[Research Agent] Completed pass {attempts}/3 (model confidence {score}/10).")
    return {
        "research_findings": compiled,
        "confidence_score": score,
        "research_attempts": attempts,
    }
