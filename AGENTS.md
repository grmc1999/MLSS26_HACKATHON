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
- `scripts/run_exp.py` — flu CLI wrapper
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
- **Graph (FalkorDB)**: Docker-backed knowledge graph, built via `scripts/build_flu_graph.py` (uses OpenRouter LLM for entity extraction)
- **Search**: `search_flu_context_rag(query, k=5)` — returns `{"vector_hits", "graph_context", "combined_context"}`
  - Vector hits: FAISS semantic search
  - Graph context: keyword-matched Cypher (no LLM needed at query time). Matches query tokens against node names, returns 1-hop neighborhood. Degrades gracefully to vector-only if FalkorDB unavailable.
- **Graph schema**: nodes `{Model, Dataset, Country, Metric, Method, Paper}`, rels `{EVALUATED_ON, ACHIEVES, USES_METHOD, CITES, COMPARED_TO}`

## 2 Available Slash Commands

Defined in `.opencode/commands/`:

| Command | What it does |
|---------|-------------|
| `/autoresearch` | Simple modify → run → keep/discard against a single metric |
| `/autoresearch_pipeline` | Full multi-expert pipeline: research → plan → code → jury → commit → run → decide → log, with adaptive RAG and research reset |
