# AGENTS.md — MLSS26_HACKATHON

## Setup

- Activate: `source .venv/bin/activate`
- Install: `pip install -e .` (namespace package `MLAgentBench`)

## Entry Points

| Command | Purpose |
|---------|---------|
| `python scripts/run_flu_pipeline.py` | Standalone flu train/eval |
| `python scripts/run_orchestrator.py --task flu` | Autonomous pipeline using autoresearch_pipeline.md phases |

## Experiment Loop Protocol

Canonical protocol is **`autoresearch_pipeline.md`** (`.opencode/commands/autoresearch_pipeline.md`). Key rules:
- Only modify `env/train.py`. Do NOT modify eval/data files.
- Run: `{VERIFY_CMD} > run.log 2>&1`
- Parse metric from stdout
- Log to `experiments/loop-flu-{YYMMDD}-{HHMM}/results.tsv`
- Time budget: 5 min per experiment. Kill at 10 min.

## Architecture

- `env/train.py` — Flu training code (diffusion model)
- `scripts/run_flu_pipeline.py` — Flu CLI wrapper
- `scripts/run_orchestrator.py` — autonomous pipeline: 8 phases per iteration
- `MLAgentBench/agents/orchestrator.py` — `ScientificAutoResearch` + `ExperimentManager`
- `MLAgentBench/agents/agent_specialized.py` — agent prompts + RAG functions
- `MLAgentBench/LLM.py` — LLM interface (all calls handled by opencode)

## Flu RAG (Vector + FalkorDB Graph)

- **Vector (FAISS)**: `index_output_flu/` — all-MiniLM-L6-v2 embeddings from 22 papers, 384-dim IVF
- **Graph (FalkorDB)**: Docker-backed knowledge graph, built via `scripts/build_flu_graph.py` (uses OpenRouter LLM for entity extraction)
- **Search**: `search_flu_context_rag(query, k=5)` — returns `{"vector_hits", "graph_context", "combined_context"}`
  - Vector hits: FAISS semantic search
  - Graph context: keyword-matched Cypher (no LLM needed at query time). Matches query tokens against node names, returns 1-hop neighborhood. Degrades gracefully to vector-only if FalkorDB unavailable.
- **Graph schema**: nodes `{Model, Dataset, Country, Metric, Method, Paper}`, rels `{EVALUATED_ON, ACHIEVES, USES_METHOD, CITES, COMPARED_TO}`
