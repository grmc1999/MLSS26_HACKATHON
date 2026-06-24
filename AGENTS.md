# AGENTS.md — MLSS26_HACKATHON

## Setup

- Activate: `source .venv/bin/activate`
- Install: `pip install -e .` (namespace package `MLAgentBench`)

## Entry Points

| Command | Purpose |
|---------|---------|
| `python scripts/run_medmnist.py` | Standalone MedMNIST train/eval |
| `python scripts/run_exp.py` | Standalone flu train/eval |
| `python scripts/run_orchestrator.py --task medmnist\|flu` | Autonomous pipeline using autoresearch_pipeline.md phases |

## Experiment Loop Protocol

Canonical protocol is **`autoresearch_pipeline.md`** (`.opencode/commands/autoresearch_pipeline.md`). Key rules:
- Only modify `{ENV_DIR}/train.py`. Do NOT modify eval/data files.
- Run: `{VERIFY_CMD} > run.log 2>&1`
- Parse metric from stdout (task-dependent)
- Log to `experiments/loop-{task}-{YYMMDD}-{HHMM}/results.tsv`
- Time budget: 5 min per experiment. Kill at 10 min.

## Architecture

- `MLAgentBench/benchmarks/medmnist/env/train.py` — MedMNIST training code
- `env/train.py` — flu training code
- `scripts/run_medmnist.py` — MedMNIST CLI wrapper
- `scripts/run_flu_pipeline.py` — flu CLI wrapper
- `scripts/run_orchestrator.py` — autonomous pipeline: 8 phases per iteration
- `MLAgentBench/agents/orchestrator.py` — `ScientificAutoResearch` + `ExperimentManager`
- `MLAgentBench/agents/agent_specialized.py` — agent prompts + RAG functions
- `MLAgentBench/LLM.py` — LLM interface (all calls handled by opencode)

## 2 RAG Systems

### MedMNIST RAG (Visual FAISS)
- **Embedding model**: Qwen3-VL-Embedding-2B (local)
- **Index**: `index_output/` — 525 tile embeddings, 2048-dim IVF, built from 28 PDFs rendered as screenshots
- **Search**: `search_medical_literature(query, k, task="medmnist")`
- **Use**: retrieves relevant chest X-ray / OOD detection papers before modifying `train.py`

### Flu RAG (Vector + FalkorDB Graph)
- **Vector (FAISS)**: `index_output_flu/` — all-MiniLM-L6-v2 embeddings from 22 papers, 384-dim IVF
- **Graph (FalkorDB)**: Docker-backed knowledge graph, built via `scripts/build_flu_graph.py` using
  the **local Qwen2.5-Coder-7B model** for entity/relationship extraction (not OpenRouter — LLM
  calls in this project run through opencode, see `MLAgentBench/LLM.py`).
  - **Entity dedup**: each extracted entity is embedded (`sentence-transformers/all-MiniLM-L6-v2`).
    Embedding distance alone is *not* a reliable same-entity signal for short names (measured: "GRU"
    vs "Gated Recurrent Unit" ~0.19 cosine similarity, while "WMT 2014 English-to-German" vs
    "...English-to-French", different entities, ~0.83 — the ranges overlap) — so embeddings only
    shortlist *candidates* (Euclidean distance ≤ `DEDUP_CANDIDATE_MAX_DIST`, capped at
    `DEDUP_MAX_CANDIDATES`), and a second call to the same already-loaded local model makes the
    actual same/different judgment per candidate.
- **Search**: `search_flu_context_rag(query, k=5)` — returns `{"vector_hits", "graph_context", "graph_summary", "combined_context"}`
  - Vector hits: FAISS semantic search
  - Graph context: **semantic** matching — embeds the query with the same MiniLM model, compares
    against every node's stored `embedding` property (Euclidean distance, top 5 within
    `_FLU_GRAPH_MATCH_MAX_DIST`), then expands 1 hop. Catches synonyms/paraphrasing a literal
    substring match would miss. No LLM call needed for this part. Falls back to the older
    substring/keyword Cypher match (`_query_flu_graph_substring()`) for a graph built before this
    change (no `embedding` property on its nodes yet).
  - Graph summary: a 1-2 sentence narration of `graph_context`, via a **local Ollama** model called
    directly (`_call_ollama()` in `agent_specialized.py`) — Ollama is free/local, not a paid external
    API, so this doesn't reintroduce what the OpenRouter cleanup removed. Purely additive: skipped
    (no call) if `graph_context` is empty, degrades to `""` if Ollama is unreachable. Start with
    `scripts/start_ollama.sh`; configure via `OLLAMA_HOST`/`OLLAMA_MODEL` in `.env`.
  - Degrades gracefully to vector-only if FalkorDB, the embedding model, or Ollama is unavailable —
    never raises.
- **Graph schema**: nodes `{Model, Dataset, Country, Metric, Method, Paper}`, rels `{EVALUATED_ON, ACHIEVES, USES_METHOD, CITES, COMPARED_TO}`

## 2 Available Slash Commands

Defined in `.opencode/commands/`:

| Command | What it does |
|---------|-------------|
| `/autoresearch` | Simple modify → run → keep/discard against a single metric |
| `/autoresearch_pipeline` | Full multi-expert pipeline: research → plan → code → jury → commit → run → decide → log, with adaptive RAG and research reset |
