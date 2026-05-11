# Synapse-AI-Solution-

## Multi-Agent Business Research Assistant

A **LangGraph** pipeline with four specialized agents that help users gather and summarize **public business information** (news, financial context, leadership, competitors, and similar). It supports **multi-turn follow-ups**, **conditional routing** based on model confidence and validation, and **human-in-the-loop clarification** when a query is ambiguous.

Search is powered by [Tavily](https://tavily.com/) (via the official `tavily-python` client). LLM inference uses [Groq](https://groq.com/) through [LangChain’s Groq integration](https://python.langchain.com/docs/integrations/chat/groq) (`llama-3.3-70b-versatile`).

---

## Features

- **Clarity agent** — Decides if the request is specific enough (company identifiable from the message or prior chat). Triggers an **interrupt** so the user can clarify when needed.
- **Research agent** — Builds a search plan, calls Tavily, then scores coverage with a **confidence score** (0–10).
- **Validator agent** — Judges whether findings are **sufficient** or **insufficient** to answer the question; can steer another research pass using a short **validation reason**.
- **Synthesis agent** — Produces a structured answer that respects **conversation history**.
- **Web UI** (optional) — FastAPI server + static chat front end.
- **CLI** — Terminal loop with the same graph behavior.

---

## Prerequisites

- Python **3.9+** (3.10+ recommended)
- API keys (free tiers available):
  - [Groq Console](https://console.groq.com) — create an API key
  - [Tavily](https://app.tavily.com) — create an API key

---

## Setup

From the **repository root** (this folder after you clone):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env` in the repo as a template):

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Do not commit real keys. This repo’s `.gitignore` ignores `.env` if you use the provided one.

---

## Run (CLI)

```bash
source .venv/bin/activate
python main.py
```

Type your question; if the assistant needs a company name or disambiguation, it will ask and then **resume** the graph after you answer. Use **Ctrl+C** to exit.

---

## Run (Web UI)

```bash
source .venv/bin/activate
python server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.

Alternatively:

```bash
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Sessions are stored **in memory** on the server (cleared when the process restarts). The browser keeps a `session_id` in `localStorage`.

---

## Project layout

```
├── README.md
├── requirements.txt
├── .env                 # your keys (not committed)
├── main.py              # CLI entry point
├── server.py            # FastAPI + static UI
├── graph.py             # LangGraph wiring & routing
├── state.py             # Shared TypedDict state
├── json_utils.py        # Robust JSON parsing for LLM outputs
├── agents/
│   ├── llm.py           # Shared ChatGroq factory
│   ├── context.py       # Conversation → prompt text
│   ├── clarity.py
│   ├── research.py
│   ├── validator.py
│   └── synthesis.py
├── tools/
│   └── search.py        # Tavily search helper
└── static/              # Front end (served by FastAPI)
    ├── index.html
    ├── styles.css
    └── app.js
```

---

## Routing summary

1. **START** → **Clarity** → (interrupt if unclear) → **Research**
2. **Research** → if **confidence ≥ 6** → **Synthesis**; else → **Validator**
3. **Validator** → if **sufficient** → **Synthesis**; if **insufficient** and **research attempts &lt; 3** → **Research** again; else → **Synthesis** (best effort)
4. **Synthesis** → **END**

---

## Notes

- **Assignment wording (“Tavily MCP”)** — This project uses Tavily’s **HTTP API** through `tavily-python`, which is the same data source many MCP integrations call. It is not wired through the MCP wire protocol; functionally it still satisfies “search-backed research.”
- **Limits** — Free API tiers on Groq and Tavily apply; long or very broad questions may hit length or rate limits.
- **Accuracy** — Answers are only as good as retrieved snippets; the synthesis step is instructed not to invent facts beyond the findings.

---

## License

Provided as sample / assignment code; adjust or add a license as needed for your submission.
