"""
Web API for the research assistant. Run from this directory:

  uvicorn server:app --reload --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from graph import build_graph

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@dataclass
class ChatSession:
    messages: list = field(default_factory=list)
    thread_id: Optional[str] = None
    awaiting_clarification: bool = False


sessions: dict[str, ChatSession] = {}
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError("Set GROQ_API_KEY and TAVILY_API_KEY in .env")
        _graph = build_graph()
    return _graph


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=8)
    message: str = Field(..., max_length=32000)


class ChatResponse(BaseModel):
    ok: bool
    kind: str  # "reply" | "clarify" | "error"
    text: Optional[str] = None
    question: Optional[str] = None
    detail: Optional[str] = None
    # Filled on kind=="reply" — final graph state for this turn
    clarity_status: Optional[str] = None
    confidence_score: Optional[int] = None
    validation_result: Optional[str] = None
    validation_reason: Optional[str] = None
    research_attempts: Optional[int] = None


def _run_metrics_from_result(result: dict) -> dict:
    """Extract scoring / routing fields from graph output for the UI."""
    out: dict = {
        "clarity_status": None,
        "confidence_score": None,
        "validation_result": None,
        "validation_reason": None,
        "research_attempts": None,
    }
    cs = (result.get("clarity_status") or "").strip()
    if cs:
        out["clarity_status"] = cs
    vr = (result.get("validation_result") or "").strip()
    if vr:
        out["validation_result"] = vr
    reason = (result.get("validation_reason") or "").strip()
    if reason:
        out["validation_reason"] = reason
    try:
        if result.get("confidence_score") is not None:
            out["confidence_score"] = int(result["confidence_score"])
    except (TypeError, ValueError):
        pass
    try:
        if result.get("research_attempts") is not None:
            out["research_attempts"] = int(result["research_attempts"])
    except (TypeError, ValueError):
        pass
    return out


def _initial_state(messages: list, query: str) -> dict:
    return {
        "messages": messages,
        "query": query,
        "clarity_status": "",
        "clarification_question": "",
        "research_findings": "",
        "confidence_score": 0,
        "validation_result": "",
        "validation_reason": "",
        "research_attempts": 0,
        "final_response": "",
    }


@app.get("/")
async def serve_index():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "static/index.html missing — run from research_assistant/")
    return FileResponse(index)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message is empty")

    try:
        graph = get_graph()
    except RuntimeError as e:
        return ChatResponse(ok=False, kind="error", detail=str(e))

    sid = req.session_id
    s = sessions.setdefault(sid, ChatSession())

    try:
        if s.awaiting_clarification:
            s.messages.append(HumanMessage(content=text))
            cfg = {"configurable": {"thread_id": s.thread_id}}
            result = graph.invoke(
                Command(
                    resume=text,
                    update={
                        "messages": s.messages,
                        "query": text,
                        "validation_reason": "",
                    },
                ),
                config=cfg,
            )
            if result.get("__interrupt__"):
                q = result["__interrupt__"][0].value
                return ChatResponse(ok=True, kind="clarify", question=str(q))

            s.awaiting_clarification = False
            s.messages = result.get("messages") or s.messages
            final = (result.get("final_response") or "").strip()
            metrics = _run_metrics_from_result(result)
            return ChatResponse(
                ok=True,
                kind="reply",
                text=final or "(No response text produced.)",
                **metrics,
            )

        # New graph turn
        s.thread_id = uuid.uuid4().hex
        cfg = {"configurable": {"thread_id": s.thread_id}}
        s.messages.append(HumanMessage(content=text))
        result = graph.invoke(_initial_state(s.messages, text), config=cfg)

        if result.get("__interrupt__"):
            s.awaiting_clarification = True
            q = result["__interrupt__"][0].value
            return ChatResponse(ok=True, kind="clarify", question=str(q))

        s.messages = result.get("messages") or s.messages
        final = (result.get("final_response") or "").strip()
        metrics = _run_metrics_from_result(result)
        return ChatResponse(
            ok=True,
            kind="reply",
            text=final or "(No response text produced.)",
            **metrics,
        )

    except Exception as e:
        if s.messages and isinstance(s.messages[-1], HumanMessage):
            s.messages.pop()
        s.awaiting_clarification = False
        return ChatResponse(ok=False, kind="error", detail=str(e))


@app.post("/api/session/new")
async def new_session():
    sid = uuid.uuid4().hex
    sessions[sid] = ChatSession()
    return {"session_id": sid}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
