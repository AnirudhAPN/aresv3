# ARES — Adaptive Research Ecosystem Suite

An AI-powered multi-agent research assistant that turns a single research
topic into a grounded research proposal — Planner → Literature → Gap
Analysis → Methodology → Reviewer — where every citation, dataset, and
gap claim is bounded by real, verifiable evidence rather than hallucinated.

## Why this exists

Most "AI research assistant" demos let an LLM invent papers, datasets, and
absolute claims about the state of a field. ARES is built around three
hard rules instead:

1. **Literature is retrieved, never generated.** The Literature Agent
   queries Semantic Scholar and arXiv *before* any LLM call; the model is
   only ever used to summarize real, already-fetched abstracts.
2. **Gap claims are evidence-bounded.** The Gap Agent can only say a topic
   "wasn't addressed in the papers reviewed here," never "no one has
   studied X" — and it must cite which retrieved paper each claim is based
   on.
3. **Methodology recommendations are whitelist-first.** Datasets and
   algorithms are checked against a curated whitelist before being marked
   `verified`; anything else is clearly separated into
   `unverified_suggestions`.

On top of this, an orchestrator (LangGraph) routes execution adaptively
based on an **Adaptive Collaboration Score (ACS)** computed independently
per agent, retrying a weak output once before flagging it "low confidence"
and moving on — never looping indefinitely.

## Architecture

```
User enters Research Topic
        ↓
Planner Agent         → objectives, questions, keywords, scope
        ↓
Literature Agent       → real papers (Semantic Scholar + arXiv) + LLM synthesis
        ↓
Gap Analysis Agent      → evidence-bounded gaps, current trends, future directions
        ↓
Methodology Agent       → whitelist-grounded datasets/algorithms
        ↓
Reviewer Agent          → Part A: code rule-checks. Part B: separate-model critique
        ↓
Final Report (Markdown / Text / PDF export)
```

Each agent is a LangGraph node. Two independent retry mechanisms exist:

- **Structural retry** (`agents/base.py`): if an LLM response fails to
  parse into its Pydantic schema, retry once with the raw invalid output +
  validation error. If that also fails, fall back to an empty/default
  output with `parse_failed: true` rather than crashing the session.
- **ACS quality retry** (`services/collaboration.py` +
  `orchestrator.py`): after a structurally valid output is obtained, its
  Adaptive Collaboration Score is computed. If ACS < 0.6, the agent is
  retried once with an augmented prompt describing its weakest dimension.
  If still below threshold, the pipeline proceeds and flags that section
  "low confidence" in the UI and exports.

### Adaptive Collaboration Score

```
ACS = 0.40 * confidence + 0.35 * evidence_quality + 0.25 * task_completion
```

- `confidence`: a **separate** lightweight LLM-as-judge call (default:
  Claude, not Gemini) scoring the output in isolation — never the
  generating call self-reporting its own confidence.
- `evidence_quality`: computed programmatically per agent (URL
  resolvability for Literature, embedding overlap with retrieved
  literature for Gap, whitelist match rate for Methodology, rule-check /
  critique agreement for Reviewer).
- `task_completion`: rule-based structural completeness against the
  Pydantic schema.

Weights are configurable via environment variables, not hardcoded.

### Conversational refinement

The pipeline isn't one-shot. Once a session completes, `POST
/api/research/{id}/chat` accepts follow-up messages and keeps the
conversation going:

- **Questions** ("what does the literature review cover?") are answered
  directly from the stored proposal — nothing is mutated.
- **Change requests** ("add more datasets," "focus the gap analysis on
  privacy risks") are routed to the single best-matching agent, which is
  re-run with the feedback folded into its original prompt, re-validated,
  and re-scored via ACS exactly as during the initial pipeline run.
- Because the Reviewer's rule-checks and critique depend on every other
  agent's output, refining any non-Reviewer section automatically triggers
  a Reviewer refresh afterward, so the report stays internally consistent
  without a full re-run.

All messages and which agent (if any) they updated are persisted per
session (`ChatMessage` table) and reloaded on revisiting the Results page.

## Project structure

```
backend/
    app.py                 FastAPI app (async background-task + polling API)
    orchestrator.py         LangGraph wiring, adaptive routing
    config.py               ACS weights, retry thresholds, API config

    agents/                 One module per agent
    prompts/                 Prompt templates per agent
    services/
        llm.py               Gemini (generation) + separate judge model
        memory.py            SQLite session persistence
        collaboration.py     ACS computation, pure routing functions
        literature_search.py Semantic Scholar + arXiv retrieval
        cache.py             30-day TTL literature cache
        export.py            Markdown / Text / PDF export

    data/
        known_datasets.json
        known_algorithms.json

    database/sqlite.py
    models/schemas.py         All Pydantic models
    tests/                    pytest suite (see Testing below)

frontend/
    src/pages/                Home, Results, History, Settings
    src/components/           Navbar, Sidebar, AgentCard, ResearchForm,
                               ProgressTimeline, OutputSection,
                               ConfidenceMeter, CollaborationGraph
```

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- A Gemini API key (required)
- An Anthropic API key (recommended — powers the independent
  confidence/reviewer judge model; without it the judge falls back to
  Gemini, which reintroduces same-family bias)
- A Semantic Scholar API key (optional — raises the shared free-tier rate
  limit of 100 req/5 min)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add GEMINI_API_KEY (and ideally ANTHROPIC_API_KEY)
uvicorn app:app --reload
```

The API is now live at `http://localhost:8000` (interactive docs at
`/docs`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI is now live at `http://localhost:5173` (proxies `/api` to
`localhost:8000`).

### Docker (both services)

```bash
cp backend/.env.example backend/.env   # fill in your keys
docker compose up --build
```

Frontend: `http://localhost:3000`. Backend: `http://localhost:8000`.

## API reference

| Method | Path | Description |
|---|---|---|
| POST | `/api/research` | Start a session. Body: `{"topic": "..."}`. Returns `{session_id, status}`. |
| GET | `/api/research/{id}/status` | Poll for progress: `current_agent`, `completed_agents`, `acs_scores`. |
| GET | `/api/research/{id}` | Full report once `status == "completed"`. |
| GET | `/api/history` | All sessions (single-user, unfiltered — see Limitations). |
| GET | `/api/history/{id}` | Same as `/api/research/{id}`. |
| GET | `/api/research/{id}/export/markdown` | Markdown export. |
| GET | `/api/research/{id}/export/text` | Plain-text export, links inline. |
| GET | `/api/research/{id}/export/pdf` | PDF export (WeasyPrint). |
| POST | `/api/research/{id}/chat` | Send a follow-up message: ask a question or request a change to one section. |
| GET | `/api/research/{id}/chat` | Full chat history for the session. |

The research endpoint is intentionally async (background task + polling)
rather than a single blocking call: five sequential agent calls, each
potentially retried, plus external literature API calls would risk client
timeouts on a single request.

## Testing

```bash
cd backend
pytest tests/ -v
```

- `test_acs.py` — ACS formula, known inputs → expected score.
- `test_schemas.py` — Pydantic validation for every agent output model.
- `test_pipeline_smoke.py` — full orchestrator run, all external calls
  (Gemini, judge model, Semantic Scholar, arXiv, sentence-transformers)
  mocked, asserting a valid `ResearchSession` is produced.
- `test_structural_retry.py` — malformed LLM output triggers exactly one
  retry, then falls back to `parse_failed: true` instead of crashing.

## Evaluation methodology

`orchestrator.py` supports three collaboration conditions for controlled
comparison, all logged to the `Metrics` table keyed by topic and
condition:

1. **`single_agent`** — one Gemini call producing the whole proposal
   directly (`run_single_agent_baseline`).
2. **`static`** — agents run in a fixed sequence, ACS computed for logging
   only, retries disabled.
3. **`adaptive`** — full ACS + retry logic (the default for
   `POST /api/research`).

Run all three conditions across the 10 fixed evaluation topics (AI in
Agriculture, Cybersecurity, Healthcare, Robotics, Education, Climate,
Blockchain, IoT, Forensics, Finance) and compare via the `Metrics` table.
Use a model/provider different from the generator as judge to avoid
circularity, and validate the automated judge against a small human-rated
sample.

## Known limitations (documented, not hidden)

- **Single-user, no authentication (v1).** `GET /api/history` returns all
  sessions in the local SQLite DB unfiltered. A `user_id` column already
  exists (hardcoded to `"default"`) so multi-user support can be added
  later without a breaking migration.
- **SQLite serializes concurrent writes.** Run evaluation sessions
  sequentially, or migrate to a server-grade DB for concurrent load.
- **`low_coverage`, `low_confidence_gaps`, and `parse_failed` flags are
  surfaced, not silently dropped** — visible as badges in the UI and
  carried into every export format.
- Literature retrieval depends on Semantic Scholar's and arXiv's public
  APIs being reachable; on failure the agent degrades gracefully (fewer
  papers, `low_coverage: true`) rather than fabricating results.
