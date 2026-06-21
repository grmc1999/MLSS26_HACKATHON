# AGENTS.md — MLSS26_HACKATHON Multi-Agent System

## Overview

This project implements a multi-agent system for ML experimentation, forked from [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) with improvements from [MLRC-Bench](https://github.com/yunx-z/MLRC-Bench). The system uses **free OpenRouter models** to power specialized agents that collaborate on the `identify-contrails` satellite imagery task.

The autonomous experiment loop is inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — agents modify code, train, evaluate, keep/discard, and repeat indefinitely. The `program.md` file provides the baseline instructions for the autoresearch loop.

Domain skills (computer-vision, deep-learning, imaging-algorithms) are integrated directly into the agents' system prompts, giving them access to library-specific code patterns and best practices.

## Architecture

```
User / Dashboard
       |
       v
  Orchestrator
       |
  +----+----+----+----+----+----+----+----+
  |    |    |    |    |    |    |    |    |
  R    A    CV   DL  LLM  SAT  CL   PHY
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Orchestrator** | `MLAgentBench/agents/orchestrator.py` | Routes subproblems to agents, manages iterations |
| **Specialized Agents** | `MLAgentBench/agents/agent_specialized.py` | 8 role-specific agents extending ResearchAgent |
| **Continual Learning** | `MLAgentBench/agents/continual_learning.py` | EWC + replay buffer + checkpoint versioning |
| **LLM Router** | `MLAgentBench/LLM.py` | Routes calls to OpenRouter (free models) |
| **Model Config** | `configs/models.yaml` | 22 free OpenRouter models with metadata |
| **Agent Config** | `configs/agents.yaml` | Agent → model mappings + system prompts |

---

## Agents

### 1. Research Literature Agent (`research_literature`)
- **Model**: `qwen/qwen3-coder:free` (1M context, strong text)
- **Upgrade**: `openai/gpt-4o`
- **Skills**: paper search, citation generation, method summarization, literature review
- **Focus**: Contrail detection, satellite image segmentation, U-Net architectures

### 2. AutoResearch Agent (`autoresearch`)
- **Model**: `nvidia/nemotron-3-ultra-550b-a55b:free` (1M context, largest free model)
- **Upgrade**: `anthropic/claude-sonnet-4`
- **Skills**: experiment planning, hypothesis generation, result analysis, iteration strategy
- **Focus**: Iterative improvement of contrail detection models

### 3. Computer Vision Expert (`cv_expert`)
- **Model**: `google/gemma-4-26b-a4b-it:free` (multimodal: text+image+video)
- **Upgrade**: `openai/gpt-4o`
- **Skills**: image preprocessing, data augmentation, architecture design, segmentation, transfer learning
- **Focus**: CNN/transformer architectures for GOES-16 satellite imagery

### 4. Deep Learning Expert (`dl_expert`)
- **Model**: `nousresearch/hermes-3-llama-3.1-405b:free` (405B params)
- **Upgrade**: `anthropic/claude-sonnet-4`
- **Skills**: training loop design, loss function engineering, optimizer config, LR scheduling, regularization, diffusion models
- **Focus**: Efficient training with mixed precision, Dice/Focal/Tversky losses

### 5. LLM Expert (`llm_expert`)
- **Model**: `qwen/qwen3-next-80b-a3b-instruct:free` (strong instruction following)
- **Upgrade**: `openai/gpt-4o`
- **Skills**: prompt engineering, multimodal reasoning, agent coordination, few-shot design, chain-of-thought
- **Focus**: Inter-agent coordination and prompt optimization

### 6. Satellite Image Expert (`satellite_expert`)
- **Model**: `nvidia/nemotron-nano-12b-v2-vl:free` (multimodal: text+image+video)
- **Upgrade**: `openai/gpt-4o`
- **Skills**: remote sensing, spectral analysis, geospatial transforms, satellite image interpretation, atmospheric correction
- **Focus**: GOES-16 ABI bands (8-16), false color composites (band 11/14/15), ERA5 data

### 7. Continual Learning Expert (`continual_learning`)
- **Model**: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (multimodal + reasoning)
- **Upgrade**: `anthropic/claude-sonnet-4`
- **Skills**: catastrophic forgetting prevention, EWC, experience replay, model versioning, checkpoint management, parameter drift monitoring
- **Focus**: Balancing plasticity and stability across training iterations

### 8. Physics Expert (`physics_expert`)
- **Model**: `nvidia/nemotron-3-super-120b-a12b:free` (1M context, strong math)
- **Upgrade**: `openai/gpt-4o`
- **Skills**: PINNs, advection-continuity equations, atmospheric physics, metric computation, physical consistency checks
- **Focus**: CSI metrics, ERA5 wind fields, physics-informed constraints

---

## OpenRouter Integration

### How It Works
All LLM calls go through OpenRouter's OpenAI-compatible API:
```
POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer $OPENROUTER_API_KEY
```

The `LLM.py` module checks if a model ID is in `OPENROUTER_FREE_MODELS` or matches a paid prefix, then routes accordingly.

### Free Models Available (22 total)

| Provider | Model ID | Context | Multimodal |
|----------|----------|---------|------------|
| Meta | `meta-llama/llama-3.3-70b-instruct:free` | 131K | No |
| Meta | `meta-llama/llama-3.2-3b-instruct:free` | 131K | No |
| NVIDIA | `nvidia/nemotron-3-ultra-550b-a55b:free` | 1M | No |
| NVIDIA | `nvidia/nemotron-3-super-120b-a12b:free` | 1M | No |
| NVIDIA | `nvidia/nemotron-3-nano-30b-a3b:free` | 256K | No |
| NVIDIA | `nvidia/nemotron-nano-9b-v2:free` | 128K | No |
| NVIDIA | `nvidia/nemotron-nano-12b-v2-vl:free` | 128K | **Yes** (text+image+video) |
| NVIDIA | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 256K | **Yes** (text+image+audio+video) |
| Qwen | `qwen/qwen3-coder:free` | 1M | No |
| Qwen | `qwen/qwen3-next-80b-a3b-instruct:free` | 262K | No |
| Google | `google/gemma-4-26b-a4b-it:free` | 262K | **Yes** (text+image+video) |
| Google | `google/gemma-4-31b-it:free` | 262K | **Yes** (text+image+video) |
| OpenAI | `openai/gpt-oss-120b:free` | 131K | No |
| OpenAI | `openai/gpt-oss-20b:free` | 131K | No |
| Nous | `nousresearch/hermes-3-llama-3.1-405b:free` | 131K | No |
| Cohere | `cohere/north-mini-code:free` | 256K | No |
| Liquid | `liquid/lfm-2.5-1.2b-instruct:free` | 32K | No |
| Poolside | `poolside/laguna-m.1:free` | 262K | No |
| Poolside | `poolside/laguna-xs.2:free` | 262K | No |
| Cognitive | `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` | 32K | No |
| Nex AGI | `nex-agi/nex-n2-pro:free` | 262K | **Yes** (text+image) |

### Swapping Models
Models can be swapped at runtime via:
1. **CLI**: `--llm-name "qwen/qwen3-coder:free"`
2. **Config**: Edit `configs/agents.yaml` → change `model` field
3. **Dashboard**: Use the Model Swap Panel in the web UI (POST `/agents/{agent}/model`)

### Upgrade Path
When budget allows, upgrade to premium models:
- `openai/gpt-4o` — best general reasoning + multimodal
- `anthropic/claude-sonnet-4` — best code generation
- `anthropic/claude-3.5-sonnet` — strong analysis

---

## Continual Learning Loop

The continual learning system prevents catastrophic forgetting across training iterations:

```
Iteration N:
  1. Load best previous checkpoint
  2. Compute Fisher Information on old data
  3. Train on new data with EWC penalty: L = L_task + (lambda/2) * sum(F_i * (theta_i - theta*_i)^2)
  4. Evaluate new score and forgetting measure
  5. Decision:
     - If improvement >= 0.01 AND forgetting < 0.05 → commit new version
     - Otherwise → rollback to best previous version
  6. Update replay buffer with exemplar samples
```

### Configuration (`configs/agents.yaml`)
```yaml
orchestrator:
  continual_learning:
    enabled: true
    checkpoint_dir: "checkpoints"
    improvement_threshold: 0.01
    forgetting_threshold: 0.05
    ewc_lambda: 100.0
    replay_buffer_size: 1000
```

---

## Task: Identify Contrails

### Dataset
- **Source**: Kaggle competition `google-research-identify-contrails-reduce-global-warming`
- **Data**: GOES-16 ABI satellite imagery (bands 8-16), 256x256 patches
- **Task**: Binary segmentation — detect aviation contrails in satellite images
- **Metric**: Dice Score
- **Baseline**: `nn.Conv2d(3, 2, 1)` (single conv layer)

### Files
- `MLAgentBench/benchmarks/identify-contrails/env/train.py` — Starter training script
- `MLAgentBench/benchmarks/identify-contrails/env/data_description.txt` — Dataset description
- `MLAgentBench/benchmarks/identify-contrails/scripts/eval.py` — Evaluation script
- `MLAgentBench/benchmarks/identify-contrails/scripts/prepare.py` — Data download script

### Running the Task
```bash
# Activate venv
source .venv/bin/activate

# Set environment variables
export OPENROUTER_API_KEY=sk-or-v1-...
export KAGGLE_CONFIG_DIR=.kaggle

# Run with a specific specialized agent
python -m MLAgentBench.runner \
  --task identify-contrails \
  --device 0 \
  --log-dir logs/contrails_run1 \
  --work-dir workspace \
  --agent-role cv_expert \
  --llm-name "google/gemma-4-26b-a4b-it:free" \
  --fast-llm-name "openai/gpt-oss-20b:free" \
  --agent-max-steps 50 \
  --max-time 18000
```

---

## ERA5 Data

Pressure-level data for the Amazon region has been downloaded:
- **Variables**: Relative humidity, Temperature, U-wind, V-wind
- **Pressure levels**: 500, 700, 850, 1000 hPa
- **Region**: 5°N to 20°S, 80°W to 35°W (Amazon basin)
- **Years**: 2023, 2024
- **Times**: 00:00, 12:00 UTC (every 12 hours)
- **Format**: NetCDF4

Files:
- `30769e617fc7d3011ae57470218bd134/data_stream-oper_stepType-instant.nc` — 2023 data (~209MB)
- `e20ab2936c0bf78f7b82871c72d77fc/data_stream-oper_stepType-instant.nc` — 2024 data (~207MB)

---

## Dashboard

### Backend (FastAPI)
- **Port**: 8000
- **Endpoints**:
  - `GET /experiments` — List all experiment runs
  - `GET /experiments/{id}` — Detailed run info
  - `GET /scores` — Score timeline data for charts
  - `GET /agents` — List all agents and their configs
  - `POST /agents/{agent}/model` — Swap LLM model per agent
  - `GET /models` — List available OpenRouter models
  - `WS /ws` — WebSocket for real-time updates

### Frontend (Next.js)
- **Port**: 3000
- **Pages**:
  - `/` — Overview dashboard with live score chart
  - `/experiments` — Experiment list with filtering
  - `/experiments/[id]` — Detailed run view with agent activity log
  - `/agents` — Agent activity timeline
  - `/config` — Model configuration (swap LLMs per agent)
  - `/leaderboard` — Ranked comparison of runs

### Starting the Dashboard
```bash
# Backend
cd dashboard/backend
source ../../.venv/bin/activate
uvicorn main:app --reload --port 8000

# Frontend
cd dashboard/frontend
npm install
npm run dev
```

---

## Adding New Agents

1. Add agent config to `configs/agents.yaml`:
```yaml
agents:
  my_new_agent:
    name: "My New Agent"
    description: "What this agent does"
    model: "meta-llama/llama-3.3-70b-instruct:free"
    fast_model: "openai/gpt-oss-20b:free"
    upgrade_model: "openai/gpt-4o"
    skills:
      - "skill1"
      - "skill2"
    system_prompt_addon: |
      You are a specialist in...
```

2. Add the prompt addon to `AGENT_PROMPTS` in `MLAgentBench/agents/agent_specialized.py`

3. Add routing keywords to `ROUTING_KEYWORDS` in `MLAgentBench/agents/orchestrator.py`

4. Run with `--agent-role my_new_agent`

---

## Hardware

- 2× NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- Python 3.10.12
- PyTorch 2.6.0+cu124
- venv at `.venv/`

---

## File Structure

```
MLSS26_HACKATHON/
├── AGENTS.md                          # This file
├── .env.example                       # API key template
├── .env                               # Actual API keys (gitignored)
├── .gitignore
├── Project_definition.md              # Research project definition
├── configs/
│   ├── models.yaml                    # OpenRouter model configurations
│   └── agents.yaml                    # Agent → model mappings
├── MLAgentBench/                      # Forked codebase
│   ├── LLM.py                         # LLM API router (OpenRouter support)
│   ├── runner.py                      # Entry point with --agent-role flag
│   ├── agents/
│   │   ├── agent.py                   # Base Agent class
│   │   ├── agent_research.py          # ResearchAgent (original)
│   │   ├── agent_specialized.py       # 8 specialized agents
│   │   ├── continual_learning.py      # EWC + replay + versioning
│   │   └── orchestrator.py            # Agent orchestrator
│   ├── benchmarks/
│   │   └── identify-contrails/        # Primary task (satellite imagery)
│   └── benchmarks_base/               # MLRC-Bench research tasks
├── data/
│   └── era5/                          # ERA5 NetCDF data
├── dashboard/
│   ├── backend/                       # FastAPI
│   └── frontend/                      # Next.js
├── scripts/
│   ├── setup.sh                       # Full installation
│   └── run_hackathon.sh               # Main launch script
└── .venv/                             # Python virtual environment
```
