from __future__ import annotations

import os
from typing import Literal

from tavily import TavilyClient

SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


def get_tavily_client() -> TavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment.")
    return TavilyClient(api_key=api_key)


def run_tavily_search(
    query: str,
    *,
    max_results: int = 6,
    search_depth: SearchDepth | None = "basic",
    topic: Literal["general", "news", "finance"] | None = None,
    include_answer: bool = True,
) -> str:
    """Run Tavily search; returns answer (if any) plus ranked snippets (MCP-free REST API)."""
    depth = search_depth or "basic"
    print(f"[Search] Tavily ({depth}, topic={topic or 'general'})...")
    client = get_tavily_client()
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=depth,
            topic=topic,
            include_answer=include_answer,
        )
    except Exception as e:
        return f"Tavily search error: {e}"

    lines: list[str] = []
    ans = (response.get("answer") or "").strip()
    if ans:
        lines.append(f"Tavily summary:\n{ans}\n")

    for i, r in enumerate(response.get("results") or [], start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        lines.append(f"{i}. {title}\n   {url}\n   {content}\n")

    return "\n".join(lines) if lines else "No results returned."
