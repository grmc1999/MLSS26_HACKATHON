# AGENTS.md — MLSS26_HACKATHON

## Setup

- Activate: `source .venv/bin/activate`
- Install: `pip install -e .` (namespace package `MLAgentBench`)

## Slash Commands

| Command | What it does |
|---------|-------------|
| `/autoresearch` | Simple flu iteration loop: modify → verify → keep/discard |
| `/autoresearch_final` | Full pipeline with RAG + code jury + logging |
| `/autoresearch_pipeline` | Full pipeline (alias for autoresearch_final) |

## Entry Points

| Command | Purpose |
|---------|---------|
| `python scripts/run_flu_pipeline.py --pretrain-epochs 30 --finetune-epochs 10` | Standalone flu train/eval |
| `python scripts/code_jury.py --task flu --env-dir env --train-py env/train.py --input-shape "(4, 5, 1)" --expected-output-shape "(4, 10, 1)" --out jury.json` | Code validation |
| `python scripts/run_rag_search.py --task flu --iteration N --query "..." --k 5 --out rag.json` | RAG literature search |

## Experiment Loop Protocol

- Only modify `env/train.py`. Do NOT modify eval/data files.
- Run: `python scripts/run_flu_pipeline.py --pretrain-epochs 30 --finetune-epochs 10 > run.log 2>&1`
- Parse: `grep "Test MAE" run.log | awk '{print $NF}'`
- Log to `experiments/loop-flu-{YYMMDD}-{HHMM}/results.tsv`
- Time budget: 5 min per experiment. Kill at 10 min.

## Architecture

```
env/
├── train.py              # Diffusion model (DiffusionForecaster + ConditionalDenoiser)
├── data.py               # CDC ILINet + WHO FluID loaders (read-only)
└── eval.py               # Evaluation metrics (read-only)

scripts/
├── run_flu_pipeline.py   # Flu CLI wrapper
├── code_jury.py          # Pre-commit code validation
├── run_rag_search.py     # RAG literature search
├── make_iter_log.py      # Per-iteration JSON logging
├── append_results_tsv.py # TSV result logging
├── build_flu_graph.py    # FalkorDB knowledge graph builder
└── start_falkordb.sh     # FalkorDB Docker launcher

dashboard/
├── backend/main.py       # FastAPI API server
└── frontend/             # Next.js interactive dashboard

MLAgentBench/agents/
├── orchestrator.py       # ScientificAutoResearch + ExperimentManager
├── agent_specialized.py  # Agent prompts + RAG functions
└── LLM.py               # LLM interface

index_output_flu/         # FAISS index (MiniLM, 22 papers)
literature_flu/           # 22 flu forecasting PDFs
literature_flu_md/        # PDFs converted to markdown
```

## Flu RAG (Vector + FalkorDB Graph)

- **Vector (FAISS)**: `index_output_flu/` — all-MiniLM-L6-v2 embeddings from 22 papers, 384-dim IVF
- **Graph (FalkorDB)**: Docker-backed knowledge graph with nodes `{Model, Dataset, Country, Metric, Method, Paper}` and relations `{EVALUATED_ON, ACHIEVES, USES_METHOD, CITES, COMPARED_TO}`
- **Search**: `search_flu_context_rag(query, k=5)` → `{"vector_hits", "graph_context", "combined_context"}`
  - Vector hits: FAISS semantic search
  - Graph context: keyword-matched Cypher (no LLM at query time)

## Dashboard

Start the interactive dashboard:

```bash
cd dashboard/backend && python main.py          # API on :8000
cd dashboard/frontend && npm run dev             # UI on :3000
```

The dashboard shows MAE curves, iteration reasoning tables, and experiment leaderboard from `experiments/loop-flu-*/`.

## Quick Start

```bash
# Run the pipeline
python scripts/run_flu_pipeline.py --pretrain-epochs 30 --finetune-epochs 10

# Or automate with opencode
# /autoresearch Goal: "improve Test MAE" Iterations: 5 RAG: yes
```
