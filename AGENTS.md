# AGENTS.md — MLSS26_HACKATHON

## Setup

- Activate: `source .venv/bin/activate`
- `OPENROUTER_API_KEY` required (from `.env`) — LLM calls route through `https://openrouter.ai/api/v1`
- Install: `pip install -e .` (namespace package `MLAgentBench`)

## Entry Points (3)

| Command | Purpose |
|---------|---------|
| `python scripts/run_medmnist.py` | Standalone train/eval. **What the orchestrator calls.** Chdirs to env dir internally. |
| `python -m MLAgentBench.runner --task medmnist --agent-role <role>` | Full ReAct agent loop (MLAgentBench pipeline). |
| `python -m MLAgentBench.agents.orchestrator --agent <role> --iterations N` | Scientific AutoResearch loop (commit → run → eval → keep/discard). |

## Experiment Loop Protocol

Canonical protocol is **`program.md`** (read it). Key rules:
- Only modify `MLAgentBench/benchmarks/medmnist/env/train.py`. Do NOT modify eval/encode files.
- Run: `python scripts/run_medmnist.py > run.log 2>&1` (redirect only, no `tee`)
- Parse stdout for `Test 3-class Accuracy:` and `OOD F1 Score:` lines
- Log to `experiments/results.tsv` (tab-separated, 6 cols: commit, test_acc, ood_f1, memory_gb, status, description)
- Time budget: 5 min per experiment. Kill at 10 min. Never stop looping.
- Baseline: 2-layer CNN (`SimpleCNN`), softmax threshold OOD detection (default `--ood-threshold 0.7`)

## Architecture

- `MLAgentBench/benchmarks/medmnist/env/train.py` — training code (what gets modified)
- `MLAgentBench/benchmarks/medmnist/env/loader.py` — data loader (reads PneumoniaMNIST + `data/medmnist_subset/chestmnist_3class.npz`)
- `scripts/run_medmnist.py` — CLI wrapper (sets `CUDA_VISIBLE_DEVICES=0`, chdirs to env)
- `MLAgentBench/agents/orchestrator.py` — `ScientificAutoResearch` class: commits changes, runs experiment, parses metric, reverts on failure
- `MLAgentBench/agents/agent_specialized.py` — 8 agents with role-specific system prompts
- `MLAgentBench/LLM.py` — routes to OpenRouter; free model IDs in `OPENROUTER_FREE_MODELS` set (line 11)
- `configs/agents.yaml` — agent→model mappings
- `configs/models.yaml` — OpenRouter model catalog

## 8 Agent Roles

Configured in `configs/agents.yaml`, system prompts in `agent_specialized.py`, routing keywords in `orchestrator.py:43-66`:
`research_literature`, `autoresearch`, `cv_expert`, `dl_expert`, `llm_expert`, `medical_expert`, `continual_learning`, `robustness_expert`

## Dashboard

- Backend: `uvicorn dashboard.backend.main:app --port 8000`
- Frontend: `cd dashboard/frontend && npm run dev` (port 3000)
- Experiment data from `experiments/runs.jsonl` and `experiments/loop-*/results.tsv`

## Utilities

- `scripts/setup.sh` — full install from scratch
- `scripts/run_hackathon.sh <agent_role> medmnist` — launch MLAgentBench.runner
- `scripts/run_autoresearch_scientific.sh <agent> <iterations>` — launch orchestrator loop
- `scripts/start_dashboard.sh` — starts both backend + frontend

## Literature

- `literature/` — 28 PDF papers on OOD detection, chest X-ray classification, domain adaptation, contrastive learning, uncertainty estimation
- `literature_md/` — same papers as markdown (text-extracted via PyMuPDF) for RAG indexing
- `index_output/` — FAISS index (525 tiles, 2048 dim, IVF) + articles.json for retrieval

## FAISS Index

Built from rendered PDF tiles using Qwen3-VL-Embedding-2B. 525 tile embeddings across 28 papers.

```python
import faiss, json
index = faiss.read_index("index_output/index.faiss")
articles = json.load(open("index_output/articles.json"))
# Query with an embedding from the model
dist, idx = index.search(query_embedding, k=5)
results = [articles[i] for i in idx[0]]
```

## PixelRAG — Visual RAG for Medical Literature

Integrated from [`github.com/StarTrail-org/PixelRAG`](https://github.com/StarTrail-org/PixelRAG).

### Model

- **Qwen3-VL-Embedding-2B** — vision-language embedding model (2.1B params)
- LoRA adapters: `lora_vit` (ViT-only), `dora_ls005` (DoRA), `hyper3` (hypernetwork)
- Pre-trained LoRA: `Chrisyichuan/wiki-screenshot-embedding-lora`
- Local path: `models/Qwen3-VL-Embedding-2B/`
- LoRA adapters path: `models/Qwen3-VL-Embedding-LoRA/`

### Usage

```python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("models/Qwen3-VL-Embedding-2B", dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("models/Qwen3-VL-Embedding-2B")
```

Fine-tune via `train/` sub-project (separate uv env, LoRA on `Qwen3-VL-Embedding-2B`).

### RAG Pipeline in Agentic Workflow

The PixelRAG model is used to retrieve relevant medical literature (PDF screenshots) during the autoresearch loop:
1. **Render** — papers are rendered as screenshot tiles via `pixelshot`
2. **Embed** — tiles are embedded with Qwen3-VL-Embedding-2B
3. **Index** — embeddings stored in FAISS index
4. **Retrieve** — agents query the index for relevant papers before modifying train.py

This gives agents visual context from medical literature (tables, charts, radiograph comparisons) during the experiment loop.

## Flu Literature Context RAG — FalkorDB Knowledge Graph (Hybrid)

Separate from PixelRAG above. The flu literature (`literature_flu_md/`, 22 papers) already has a
text-based vector index (`index_output_flu/`, built by `scripts/build_flu_rag.py`,
`sentence-transformers/all-MiniLM-L6-v2` + FAISS). This is queried via
`search_medical_literature(query, k, task="flu")`.

To answer *relational* questions a vector index can't ("which model was evaluated on which
country with which method, and what metric did it achieve") this is complemented — not
replaced — by a knowledge graph in [FalkorDB](https://www.falkordb.com/), built and queried via
LangChain (`langchain-community`, `langchain-experimental`).

### Pipeline

1. **Start FalkorDB**: `bash scripts/start_falkordb.sh` (Docker; graph on `localhost:6379`, browser UI on `localhost:3001`)
2. **Build the graph** (offline, run once / re-run after literature changes): `python scripts/build_flu_graph.py`
   - Chunks `literature_flu_md/*.md` the same way as `build_flu_rag.py`
   - Extracts entities/relationships per chunk with `LLMGraphTransformer`, LLM routed through local Qwen2.5-3B
   - Schema: nodes `{Model, Dataset, Country, Metric, Method, Paper}`,
     relationships `{EVALUATED_ON, ACHIEVES, USES_METHOD, CITES, COMPARED_TO}`
   - Clears and rebuilds the graph each run (`--no-reset` to skip)
3. **Query (hybrid)**: `search_flu_context_rag(query, k=5)` in `MLAgentBench/agents/agent_specialized.py`
   - Returns `{"vector_hits": [...], "graph_context": "...", "combined_context": "..."}`
   - `vector_hits` — same FAISS search as `search_medical_literature(..., task="flu")`
   - `graph_context` — answer from `GraphCypherQAChain` over the FalkorDB graph
   - Degrades to vector-only (empty `graph_context`) if FalkorDB/langchain are unreachable

### Call sites

- `MLAgentBench/agents/agent_specialized.py` — `AGENT_PROMPTS["time_series_expert"]` references this
- `scripts/run_orchestrator.py` — uses it (instead of `search_medical_literature`) when `--task flu`
- `consult_agent.py --role time_series_expert --rag --graph` — CLI bridge, `--graph` opts into the hybrid path

### Config

`FALKORDB_HOST`, `FALKORDB_PORT`, `FALKORDB_GRAPH_NAME` in `.env` (see `.env.example`).

## No CI / No tests / No lint / No typecheck

This project has no CI pipeline, test suite, linter, or type checker.

## Slash Commands (15 autoresearch modes)

Defined in `.opencode/commands/autoresearch_*.md`:

| Command | What it does |
|---------|-------------|
| `/autoresearch` | Modify train.py → run → keep/discard against metric |
| `/autoresearch_plan` | Convert goal into experiment config |
| `/autoresearch_debug` | Hunt bugs via hypothesis testing |
| `/autoresearch_fix` | Fix errors one-by-one to zero |
| `/autoresearch_security` | Security audit of pipeline |
| `/autoresearch_ship` | Lock best model, final eval |
| `/autoresearch_scenario` | Explore edge cases and sensitivity |
| `/autoresearch_predict` | 5-expert debate before changing code |
| `/autoresearch_learn` | Extract cross-iteration lessons |
| `/autoresearch_reason` | Adversarial debate with blind judges |
| `/autoresearch_probe` | Surface hidden constraints |
| `/autoresearch_improve` | Research SOTA methods, generate PRDs |
| `/autoresearch_evals` | Analyze trends across all runs |
| `/autoresearch_regression` | Baseline vs candidate stability gate |
| `/autoresearch_scientific` | Full loop + 8 specialized agents |
