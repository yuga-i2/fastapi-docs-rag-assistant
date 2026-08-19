# FastAPI Docs Assistant

A Retrieval-Augmented Generation (RAG) app that answers developer questions
about the [FastAPI](https://fastapi.tiangolo.com/) web framework, grounded
in its official documentation — with citations, retrieval evaluation, and
a monitoring dashboard.

Built as a course capstone project, but designed to be a genuinely useful,
standalone tool: point it at a different doc set and it becomes a Q&A
assistant for *your* documentation.

## Table of contents

- [Problem](#problem)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Configuration](#configuration-llm-providers)
- [How it works, step by step](#how-it-works-step-by-step)
- [Evaluation results](#evaluation-results)
- [Monitoring](#monitoring)
- [Project structure](#project-structure)
- [Design tradeoffs](#design-tradeoffs--limitations)
- [Reproducing from scratch](#reproducing-from-scratch)

## Problem

FastAPI's documentation is thorough but large (dozens of pages across
tutorials, advanced usage, and deployment). A developer with a specific
question ("how do I add a background task", "how do I deploy with
Docker") has to search or skim multiple pages to find the answer.

This app lets you ask a question in plain English and get a short,
grounded answer with links back to the exact doc sections it came from
— instead of a generic LLM answer that might hallucinate FastAPI details
that don't exist.

## Architecture

```
                    ┌─────────────────┐
  GitHub (raw docs) │   app/ingest.py  │  clean + chunk markdown
  ───────────────►  │                  │ ───────────► data/chunks.json
                    └─────────────────┘

  data/chunks.json  ┌─────────────────┐
  ───────────────►  │  app/search.py  │  BM25 + TF-IDF/SVD, RRF fusion
                    │  HybridSearch    │ ───────────► data/search_index.pkl
                    └─────────────────┘

  user question      ┌────────────────┐   retrieved     ┌───────────┐
  ───────────────►   │   app/rag.py   │ ─── chunks ───►  │ app/llm.py │
                     │  answer_question│                  │  (OpenAI /  │
                     │                │ ◄─── answer ────  │ Anthropic / │
                     └────────────────┘                  │ Ollama/demo)│
                            │                              └───────────┘
                            ▼
                     ┌────────────────┐
                     │   app/db.py    │  logs every Q&A + feedback
                     │  SQLite         │ ───────────► data/monitoring.db
                     └────────────────┘
                            │
                            ▼
                 app/streamlit_app.py
              (Chat tab + Monitoring tab)
```

## Quickstart

### Option A — Docker (recommended)

```bash
git clone <your-fork-url> fastapi-docs-assistant
cd fastapi-docs-assistant
docker compose up --build
```

Open http://localhost:8501. On first run, the container automatically
ingests the docs and builds the search index (takes ~30 seconds), then
launches the app.

### Option B — Local Python

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m app.ingest                                       # build the knowledge base
python -c "from app.search import build_index; build_index()"  # build the search index

streamlit run app/streamlit_app.py
```

Open http://localhost:8501.

By default the app runs in **demo mode**: no LLM key needed, and it shows
the top retrieved passage instead of a generated answer, so you can try
the whole pipeline (retrieval, citations, feedback, monitoring) with zero
setup. This is a fallback when no key is set. If you want real generated
answers, Groq is the easiest free, no-credit-card option; otherwise set
OpenAI, Anthropic, or Ollama as needed. See
[Configuration](#configuration-llm-providers) to enable real generated
answers.

## Configuration (LLM providers)

Set environment variables (or a `.env` file, or `docker-compose` env
vars) to switch providers:

| Variable | Values | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai`, `anthropic`, `ollama`, `groq`, `demo` | auto-detects `openai`/`anthropic` from whichever API key is set if unspecified |
| `OPENAI_API_KEY` | your key | needed if `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | your key | needed if `LLM_PROVIDER=anthropic` |
| `GROQ_API_KEY` | your key | needed if `LLM_PROVIDER=groq` |
| `LLM_MODEL` | e.g. `gpt-4o-mini`, `claude-sonnet-4-6`, `llama3.1`, `llama-3.1-70b-versatile` | optional override |

Example:

```bash
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai
streamlit run app/streamlit_app.py
```

```powershell
$env:GROQ_API_KEY = "gsk_..."
$env:LLM_PROVIDER = "groq"
streamlit run app/streamlit_app.py
```

A free Groq key can be created at https://console.groq.com/keys. Groq is
an easy no-credit-card option for real generation without paying for a
separate API plan.

`ollama` mode expects a local Ollama server at `localhost:11434` — no API
key needed, fully offline generation.

## How it works, step by step

1. **Ingestion** (`app/ingest.py`) — fetches 49 pages of official FastAPI
   docs (tutorial, advanced, deployment sections) from GitHub, strips
   HTML/template noise, splits each page into ~1200-character chunks
   along heading boundaries (with overlap for chunks that span a
   heading). Produces `data/chunks.json` (611 chunks). Fully automated —
   one command, no manual steps.

2. **Indexing** (`app/search.py`) — builds two indexes over the chunks:
   - **Keyword**: BM25 (`rank_bm25`), the standard lexical-overlap
     baseline.
   - **Vector**: TF-IDF → Truncated SVD → cosine similarity. This is a
     lightweight, fully local "poor man's embedding" — no model
     downloads, no API calls, so the whole project is reproducible
     offline. (See [Design tradeoffs](#design-tradeoffs--limitations)
     for how to upgrade this to real embeddings.)
   - **Hybrid**: both rankings are fused with Reciprocal Rank Fusion
     (RRF), which is more robust than either signal alone (see
     [Evaluation results](#evaluation-results)).

3. **Retrieval + generation** (`app/rag.py`) — given a question, retrieves
   the top-k chunks (configurable retrieval method), assembles them into
   a prompt with a system instruction to answer *only* from the provided
   context, and calls the LLM. Returns the answer plus the exact source
   files/headings used, so answers are auditable.

4. **Bonus: query rewriting** (`app/rewrite.py`) — optionally expands the
   question into 2 alternative phrasings via the LLM (helps when a
   user's wording doesn't match the docs' vocabulary), retrieves for
   each, and merges results. No-ops to the original question if no LLM
   key is set.

5. **Bonus: reranking** (`app/rerank.py`) — optionally retrieves a larger
   candidate set via hybrid search, then has the LLM score each
   candidate's relevance to the exact question and reorders before
   truncating to top-k. A common two-stage "retrieve broad, rerank
   narrow" pattern. No-ops (keeps hybrid-search order) if no LLM key is
   set.

6. **Interface** (`app/streamlit_app.py`) — a Streamlit app with a chat
   tab (ask, get answer + sources, thumbs up/down) and a monitoring tab.

7. **Monitoring** (`app/db.py`) — every question, the retrieval method
   used, response time, and any feedback is logged to SQLite. The
   monitoring tab reads this and renders 6 charts: questions/day,
   retrieval method usage, response time distribution, feedback
   breakdown, response time over time, and a recent-questions table.

## Evaluation results

### Retrieval evaluation

Ground truth: 240 question/chunk pairs, generated automatically from a
random sample of 120 chunks (`app/generate_ground_truth.py` — 2 questions
per chunk). Measured Hit Rate@5 and MRR for each retrieval method
(`app/evaluate_retrieval.py`):

| Method | Hit Rate@5 | MRR |
|---|---|---|
| Keyword (BM25) | 0.6667 | 0.4675 |
| Vector (TF-IDF/SVD) | 0.7417 | 0.5047 |
| **Hybrid (RRF fusion)** | **0.7542** | **0.5413** |

**Hybrid retrieval was selected** as the default: it has the best hit
rate of the three, and — more importantly than these specific numbers —
it's structurally more robust than either signal alone, because keyword
and vector search fail on different kinds of queries (exact API names vs.
paraphrased/conceptual questions). Full results in
`data/retrieval_eval_results.json`; regenerate with:

```bash
python -m app.generate_ground_truth
python -m app.evaluate_retrieval
```

### LLM output evaluation

Compared two system-prompt variants on 20 sampled questions
(`app/evaluate_llm.py`):
- **concise** — terse instructions, answer only
- **grounded_cited** — explicit instruction to cite sources and say
  "not covered" rather than guess

Scored 1-5 on relevance and groundedness by an LLM-as-judge (or, in demo
mode, a documented heuristic word-overlap proxy — see
[Design tradeoffs](#design-tradeoffs--limitations)). Regenerate with:

```bash
python -m app.evaluate_llm
```

| Variant | Relevance | Groundedness |
|---|---:|---:|
| concise | 4.75 | 4.62 |
| grounded_cited | 4.38 | 4.50 |

Results land in `data/llm_eval_results.json`. **The `concise` variant was
selected as the winner** because it achieved the best relevance and
groundedness scores, while also being simpler and shorter. This
comparison is only meaningful with a real LLM key configured — in demo
mode both variants produce identical extractive output, since demo mode
doesn't actually read the system prompt. With a real key configured
(e.g. `GROQ_API_KEY` or `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`), rerun this
to get a real relevance/groundedness comparison and pick the
better-performing prompt.

## Monitoring

The Monitoring tab in the Streamlit app is populated from real logged
interactions (`data/monitoring.db`) and shows:

1. Questions per day
2. Retrieval method usage
3. Response time distribution
4. Feedback breakdown (👍/👎/no feedback)
5. Response time over time
6. Recent questions table

Plus 4 top-line metrics: total questions, avg response time, feedback
rate, and satisfaction rate (of rated answers).

## Project structure

```
.
├── app/
│   ├── doc_sources.py          # list of FastAPI doc pages to ingest
│   ├── ingest.py                # fetch, clean, chunk -> data/chunks.json
│   ├── search.py                # BM25 + TF-IDF/SVD hybrid search (RRF)
│   ├── llm.py                   # provider-agnostic LLM wrapper
│   ├── rag.py                   # retrieve -> prompt -> generate
│   ├── rewrite.py                # bonus: query rewriting
│   ├── rerank.py                 # bonus: LLM reranking
│   ├── db.py                     # SQLite interaction/feedback logging
│   ├── generate_ground_truth.py  # builds retrieval eval question set
│   ├── evaluate_retrieval.py     # hit-rate/MRR across methods
│   ├── evaluate_llm.py           # LLM-as-judge prompt comparison
│   └── streamlit_app.py          # chat UI + monitoring dashboard
├── data/                          # generated at runtime (gitignored except .gitkeep)
├── tests/
│   └── test_pipeline.py          # pytest smoke tests, no API key required
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
└── requirements.txt
```

## Design tradeoffs & limitations

Documented honestly, since these are exactly the kind of decisions a
reviewer will want explained:

- **Vector search uses TF-IDF/SVD, not a neural embedding model.** This
  was a deliberate choice to keep the project runnable with zero API
  keys and zero large model downloads (some sandboxed/CI environments
  can't reach model hubs). It's a real vector-search implementation —
  documents are embedded into a dense space and compared by cosine
  similarity — just a weaker one than e.g. OpenAI's
  `text-embedding-3-small` or a sentence-transformers model. To upgrade:
  swap the body of `HybridSearch.__init__`/`vector_search` in
  `app/search.py` to call your embedding model of choice; nothing else
  in the pipeline (RRF fusion, evaluation scripts) needs to change.
- **Demo mode.** Demo mode is a fallback when no LLM key is set:
  `app/llm.py` returns the top retrieved passage instead of a generated
  answer, clearly labeled `[DEMO MODE]`. This keeps the entire app —
  retrieval, citations, feedback, monitoring — runnable and demoable with
  no setup. If you want real generated answers, Groq is the easiest free,
  no-credit-card option; generation quality (and the LLM-eval comparison)
  becomes meaningful once a real key is configured.
- **Ground truth for retrieval eval is heuristic-template-based in demo
  mode** (`app/generate_ground_truth.py`), and LLM-paraphrased when a key
  is set. The LLM-paraphrased version produces harder, more realistic
  queries and is the recommended way to run the evaluation for a
  submission.
- **SQLite for monitoring**, not Postgres — fine for a single-instance
  demo/course project; swap `app/db.py`'s connection logic for a
  multi-user production deployment.

## Reproducing from scratch

```bash
git clone <your-fork-url> fastapi-docs-assistant
cd fastapi-docs-assistant
pip install -r requirements.txt

python -m app.ingest
python -c "from app.search import build_index; build_index()"
python -m app.generate_ground_truth
python -m app.evaluate_retrieval
python -m app.evaluate_llm            # more meaningful with an LLM key set
pytest tests/ -v

streamlit run app/streamlit_app.py
```

All dependencies are pinned to minimum versions in `requirements.txt`.
Tested with Python 3.11/3.12.
