# MLSS26 Hackathon — Multi-Agent AutoResearch for Contrail Detection

An autonomous multi-agent research system for satellite image segmentation, forked from [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) with improvements from [MLRC-Bench](https://github.com/yunx-z/MLRC-Bench). Uses **free OpenRouter models** to power 8 specialized agents that collaboratively improve contrail detection models.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — agents modify code, train for a fixed time budget, evaluate, keep/discard, and repeat indefinitely.

---

## Architecture

```
User / Dashboard
       |
       v
  Orchestrator  ─── Continual Learning Manager (EWC + Replay)
       |
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │    │    │    │    │    │    │    │    │
  R    A    CV   DL  LLM  SAT  CL   PHY
```

| Component | File | Description |
|-----------|------|-------------|
| **Orchestrator** | `MLAgentBench/agents/orchestrator.py` | Routes subproblems to agents, manages iterations |
| **8 Specialized Agents** | `MLAgentBench/agents/agent_specialized.py` | Domain experts extending ResearchAgent |
| **Continual Learning** | `MLAgentBench/agents/continual_learning.py` | EWC penalty + replay buffer + checkpoint versioning |
| **LLM Router** | `MLAgentBench/LLM.py` | OpenRouter API (22 free models) |
| **Model Config** | `configs/models.yaml` | Free OpenRouter models with metadata |
| **Agent Config** | `configs/agents.yaml` | Agent → model mappings + domain skill prompts |

---

## Project Structure

```
MLSS26_HACKATHON/
├── AGENTS.md                     # Full agent documentation
├── README.md                     # This file
├── program.md                    # Karpathy-style autoresearch instructions
├── .env.example                  # API key template (no real keys)
├── configs/
│   ├── agents.yaml               # 8 agents with model assignments
│   └── models.yaml               # 22 free OpenRouter models
├── MLAgentBench/                 # Forked codebase
│   ├── LLM.py                    # OpenRouter API router
│   ├── runner.py                 # Entry point with --agent-role flag
│   ├── agents/
│   │   ├── agent.py              # Base Agent class
│   │   ├── agent_research.py     # ResearchAgent (original)
│   │   ├── agent_specialized.py  # 8 specialized agents + domain skills
│   │   ├── continual_learning.py # EWC + replay + versioning
│   │   └── orchestrator.py       # Agent orchestrator
│   └── benchmarks/
│       ├── identify-contrails/   # Primary task (GOES-16 satellite)
│       └── cifar10/              # Secondary task
├── dashboard/
│   ├── backend/                  # FastAPI (port 8000)
│   │   └── main.py               # API: experiments, agents, scores, models
│   └── frontend/                 # Next.js (port 3000)
│       └── app/
│           ├── page.tsx          # Overview with live score charts
│           ├── experiments/      # Experiment list + detail view
│           ├── agents/           # Agent status and models
│           ├── config/           # Model swap panel
│           └── leaderboard/      # Ranked experiment comparison
├── data/
│   └── era5/                     # ERA5 Amazon basin data (2023-2024)
├── scripts/
│   ├── run_hackathon.sh          # Launch an agent experiment
│   ├── start_dashboard.sh        # Start backend + frontend
│   └── setup.sh                  # Full environment setup
└── .venv/                        # Python virtual environment
```

---

## 8 Specialized Agents

| Agent | Role | Model | Upgrade Path |
|-------|------|-------|-------------|
| **Research Literature** | Paper search, citations, method summarization | `qwen/qwen3-coder:free` | `gpt-4o` |
| **AutoResearch** | Experiment planning, hypothesis generation | `nemotron-3-ultra-550b-a55b:free` | `claude-sonnet-4` |
| **CV Expert** | Image preprocessing, augmentation, architectures | `gemma-4-26b-a4b-it:free` | `gpt-4o` |
| **DL Expert** | Training loops, loss functions, optimization | `hermes-3-llama-3.1-405b:free` | `claude-sonnet-4` |
| **LLM Expert** | Prompt engineering, multimodal reasoning, coordination | `qwen3-next-80b-a3b-instruct:free` | `gpt-4o` |
| **Satellite Expert** | Remote sensing, spectral analysis, geospatial | `nemotron-nano-12b-v2-vl:free` | `gpt-4o` |
| **Continual Learning** | Anti-forgetting, EWC, checkpoint versioning | `nemotron-3-nano-omni-30b-a3b:free` | `claude-sonnet-4` |
| **Physics Expert** | Physics constraints, advection, CSI metrics | `nemotron-3-super-120b-a12b:free` | `gpt-4o` |

All agents use **free OpenRouter models** — zero API cost.

---

## Quick Start

### 1. Setup
```bash
source .venv/bin/activate
export OPENROUTER_API_KEY=sk-or-v1-...
```

### 2. Run an experiment
```bash
# Use the CV Expert for satellite imagery
python -m MLAgentBench.runner \
  --task identify-contrails \
  --device 0 \
  --log-dir logs/run1 \
  --work-dir workspace \
  --agent-role cv_expert \
  --llm-name "google/gemma-4-26b-a4b-it:free" \
  --max-time 18000
```

### 3. Start the dashboard
```bash
bash scripts/start_dashboard.sh
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

### 4. Run the autonomy loop
```bash
# Karpathy-style: modify, train, evaluate, repeat
bash scripts/run_hackathon.sh autoresearch identify-contrails
```

---

## Task: Identify Contrails

- **Data**: GOES-16 ABI satellite imagery (bands 8-16), 256×256 patches  
- **Label**: Binary segmentation masks (contrail vs. no-contrail)  
- **Metric**: Dice Score  
- **Baseline**: `nn.Conv2d(3, 2, 1)` — single convolutional layer  
- **Kaggle**: https://kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming

---

## Dashboard

| Page | Route | Feature |
|------|-------|---------|
| Overview | `/` | Live score chart, experiment stats |
| Experiments | `/experiments` | List all runs with filtering |
| Experiment Detail | `/experiments/[id]` | Score progression + agent activity log |
| Agents | `/agents` | Agent status and model assignments |
| Config | `/config` | Swap LLM models per agent at runtime |
| Leaderboard | `/leaderboard` | Ranked experiment comparison |

---

## ERA5 Data (Amazon Basin)

Downloaded atmospheric data for the Amazon region:

| Detail | Value |
|--------|-------|
| Variables | Temperature, humidity, u/v wind |
| Levels | 500, 700, 850, 1000 hPa |
| Region | 5°N–20°S, 80°W–35°W |
| Years | 2023, 2024 |
| Frequency | 12-hourly (00:00, 12:00 UTC) |
| Size | ~416MB total (NetCDF4) |

---

## License

MIT — inherited from [MLAgentBench](https://github.com/snap-stanford/MLAgentBench).
