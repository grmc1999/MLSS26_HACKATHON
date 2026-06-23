# MLSS26 Hackathon — Scientific AI AutoResearch (Multi-Task)

An autonomous **Scientific AI AutoResearch** system with 2 RAG-augmented experiment pipelines: chest X-ray OOD detection (MedMNIST) and cross-country ILI forecasting (flu).

## Available Tasks

| Task | Domain | Runner | Primary Metric |
|------|--------|--------|----------------|
| **medmnist** | Chest X-ray OOD detection | `scripts/run_medmnist.py` | OOD F1 / ID Test Acc |
| **flu** | ILI forecasting (CDC → WHO) | `scripts/run_exp.py` | Test MAE |

Pass `Task: medmnist` or `Task: flu` when invoking `/autoresearch_pipeline`.

## Quick Start

```bash
source .venv/bin/activate

# Run a standalone experiment
python scripts/run_medmnist.py --epochs 25   # medmnist
python scripts/run_exp.py --epochs 50         # flu

# Launch the autonomous pipeline
python scripts/run_orchestrator.py --task medmnist --iterations 25
python scripts/run_orchestrator.py --task flu --iterations 25
```

## 2 RAG Systems

| | MedMNIST RAG | Flu RAG |
|---|-------------|---------|
| **Index** | `index_output/` | `index_output_flu/` |
| **Papers** | 28 (OOD detection, chest X-ray) | 22 (influenza forecasting) |
| **Embedding** | Qwen3-VL-Embedding-2B (vision, 2048-dim) | all-MiniLM-L6-v2 (text, 384-dim) |
| **Type** | Visual — PDF screenshot tiles | Text — markdown chunks |
| **Graph** | ❌ | ✅ FalkorDB knowledge graph |
| **Search** | `search_medical_literature(query, k, task="medmnist")` | `search_flu_context_rag(query, k)` → vector hits + graph context (keyword-matched Cypher, no LLM) |

The flu RAG adds a **FalkorDB knowledge graph** for relational queries ("which model on which country with which metric"). Querying uses keyword-matched Cypher — no LLM call, deterministic, instant. The graph is built offline via `python scripts/build_flu_graph.py`.

## 2 Slash Commands

Defined in `.opencode/commands/`:

| Command | What it does |
|---------|-------------|
| `/autoresearch` | Simple modify → run → keep/discard against a single metric |
| `/autoresearch_pipeline` | Full multi-expert pipeline: research → plan → code → jury → commit → run → decide → log, with adaptive RAG and research reset |

## Architecture

The system follows **`autoresearch_pipeline.md`** (`.opencode/commands/autoresearch_pipeline.md`), an 8-phase iteration loop driven by opencode. The Research phase uses a **4-tier RAG search strategy**:

```
                                ┌──────────────────────────┐
                                │         User             │
                                │  (/autoresearch_pipeline)│
                                └──────────┬───────────────┘
                                           │
                                           v
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: RESEARCH                                                         │
│                                                                           │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                   4-Tier Search Strategy                         │    │
│   │                                                                  │    │
│   │  Tier 1:  Local FAISS (task-specific) ─── fastest, no API        │    │
│   │           medmnist → search_medical_literature()                  │    │
│   │           flu      → search_flu_context_rag() (+ FalkorDB graph) │    │
│   │                                                                  │    │
│   │  Tier 2:  Tavily web search ── broader / post-index methods      │    │
│   │           tvly search --depth=basic --max-results=5              │    │
│   │                                                                  │    │
│   │  Tier 3:  paper-navigator (Semantic Scholar) ── academic papers  │    │
│   │           Requires S2_API_KEY for rubric-based discovery         │    │
│   │                                                                  │    │
│   │  Tier 4:  research-ideation (evo-memory) ── generate from        │    │
│   │           scratch when search fails; avoids prior dead ends      │    │
│   └──────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: PLAN                                                             │
│   - Synthesize research into experiment hypothesis                        │
│   - If needed: tvly search for recent work on proposed approach           │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 3: IMPLEMENT                                                        │
│   - Make ONE focused change to {ENV_DIR}/train.py                         │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 4: CODE JURY                                                        │
│   - Syntax check → forward pass → loss → backward pass                    │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           v
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 5: COMMIT  │  Phase 6: RUN  │  Phase 7: DECIDE  │  Phase 8: LOG    │
│  git add + commit  │ {VERIFY_CMD}   │ keep or discard   │ results.tsv      │
│                    │ > run.log      │ (revert if worse) │                  │
└────────────────────────────────────────────────────────────────────────────┘

Every 10 iterations → Phase 9: Adaptive RAG Refresh (Tavily + arxiv discovery, rebuild index)
Every 20 iterations → Phase 10: Research Reset (paradigm shift on plateau)
```

## Experiment Protocol

Canonical protocol: **`autoresearch_pipeline.md`** (`.opencode/commands/autoresearch_pipeline.md`).

- Only modify `{ENV_DIR}/train.py`. Do NOT modify eval/data files.
- Run: `{VERIFY_CMD} > run.log 2>&1`
- Parse metric from stdout, log to `experiments/loop-*/results.tsv`
- Time budget: 5 min per experiment. Kill at 10 min.

## Project Structure

```
MLSS26_HACKATHON/
├── AGENTS.md
├── README.md
├── .opencode/commands/
│   ├── autoresearch.md               # Simple loop
│   └── autoresearch_pipeline.md      # Multi-expert pipeline
├── env/                           # Flu forecasting task
│   ├── data.py                    # CDC ILINet + WHO FluID loaders
│   ├── train.py                   # Forecasting models
│   └── eval.py                    # Evaluation metrics
├── MLAgentBench/benchmarks/medmnist/env/
│   ├── train.py                   # MedMNIST training code
│   └── loader.py                  # PneumoniaMNIST + ChestMNIST loader
├── MLAgentBench/agents/
│   ├── orchestrator.py            # ScientificAutoResearch + ExperimentManager
│   └── agent_specialized.py       # Agent prompts + RAG functions
├── MLAgentBench/LLM.py            # LLM interface (handled by opencode)
├── scripts/
│   ├── run_medmnist.py            # MedMNIST CLI wrapper
│   ├── run_exp.py                 # Flu CLI wrapper
│   ├── run_orchestrator.py        # Autonomous pipeline
│   ├── build_flu_graph.py         # FalkorDB knowledge graph builder
│   └── start_falkordb.sh          # FalkorDB Docker launcher
├── index_output/                  # MedMNIST FAISS index (Qwen3-VL, 525 tiles)
├── index_output_flu/              # Flu FAISS index (MiniLM, 475 chunks)
├── literature/                    # 28 OOD/chest X-ray PDFs
├── literature_flu/                # 22 flu forecasting PDFs
└── experiments/                   # Run logs and results
```

## FalkorDB (Flu Knowledge Graph)

Populate once (uses an LLM for entity extraction via `scripts/build_flu_graph.py`):

```bash
bash scripts/start_falkordb.sh
python scripts/build_flu_graph.py
```

Querying uses keyword-matched Cypher — no LLM needed at retrieval time:

```python
from MLAgentBench.agents.agent_specialized import search_flu_context_rag
result = search_flu_context_rag("GRU model France evaluation", k=5)
# "vector_hits" → FAISS semantic search
# "graph_context" → deterministic keyword-matched Cypher
# "combined_context" → both combined
```
