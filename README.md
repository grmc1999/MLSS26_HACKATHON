# MLSS26 Hackathon — Scientific AI AutoResearch

An autonomous **Scientific AI AutoResearch** system for satellite image segmentation, forked from [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) with improvements from [MLRC-Bench](https://github.com/yunz-x/MLRC-Bench). Uses **free OpenRouter models** to power specialized agents that collaborate on the `identify-contrails` satellite imagery task.

The system unifies **Karpathy's autoresearch** (modify → train → evaluate → keep/discard → repeat) with **8 specialized scientific AI agents** that provide domain expertise at each step.

---

## Unified Architecture

```
User / Dashboard
       |
       v
+------------------------------+
|    Scientific AutoResearch   |  ← Unified Loop: modify → run → eval → keep/discard → repeat
|    Orchestrator              |
+------------------------------+
       |         ↑
       |         | consultation
       v         |
+------------------------------+
|    8 Specialized Agents     |  ← Domain experts for advice
|  (CV, DL, Satellite, etc.)  |
+------------------------------+
       |
       v
+------------------------------+
|     Experiment Pipeline     |  ← train.py → run_exp.py → eval.py
+------------------------------+
       |
       v
+------------------------------+
|   Dashboard + Logging       |  ← TSV, JSONL, FastAPI, Next.js
+------------------------------+
```

### Core Loop (Karpathy-style)

```
LOOP FOREVER (up to max_iterations):
  1. Consult specialized agent for next hypothesis
  2. Modify train.py (architecture, loss, aug, etc.)
  3. git commit
  4. Run experiment → extract Dice Score
  5. If improved → KEEP (advance branch)
  6. If worse/crash → DISCARD (git revert)
  7. Log results, repeat
```

| Component | File | Description |
|-----------|------|-------------|
| **AutoResearch Orchestrator** | `MLAgentBench/agents/orchestrator.py` | Unified loop + 14 subcommands + agent routing |
| **8 Specialized Agents** | `MLAgentBench/agents/agent_specialized.py` | Domain experts with skill-integrated prompts |
| **Autoresearch Skill** | `.opencode/skills/autoresearch/SKILL.md` | 14 subcommands for the loop |
| **🚀 AutoResearch Scientific** | `.opencode/skills/autoresearch_scientific/SKILL.md` | Autoresearch loop + 8 scientific agents + 14 subcommands |
| **Task Instructions** | `program.md` | Task-specific autoresearch protocol |
| **Continual Learning** | `MLAgentBench/agents/continual_learning.py` | EWC + replay + checkpoint versioning |
| **LLM Router** | `MLAgentBench/LLM.py` | OpenRouter API (22 free models) |
| **Model Config** | `configs/models.yaml` | Free OpenRouter models with metadata |
| **Agent Config** | `configs/agents.yaml` | Agent → model mappings + domain skill prompts |

---

## 🚀 Autoresearch Scientific Mode

The **`autoresearch_scientific`** skill merges the OpenCode autoresearch loop with the 8 specialized agents into a single unified command.

### Usage
```
# From OpenCode (chat)
/autoresearch_scientific Goal="Improve Dice Score" Metric="Test Dice" Iterations=25
/autoresearch_scientific_plan Agent=cv_expert
/autoresearch_scientific_ship

# From CLI
bash scripts/run_autoresearch_scientific.sh [agent_role] [iterations]
python -m MLAgentBench.agents.orchestrator --agent cv_expert --iterations 10 --subcommand ship
```

### How It Works
```
LOOP FOREVER (bounded):
  1. Route to best agent via keyword matching
  2. Agent proposes hypothesis + code change with scientific reasoning
  3. Modify train.py → git commit → run experiment
  4. Extract Test Dice → decide KEEP (if improved) or DISCARD (if worse/crash)
  5. Log to TSV + dashboard → repeat
```

### Available Agents for Routing
`research_literature`, `autoresearch`, `cv_expert`, `dl_expert`, `llm_expert`, `satellite_expert`, `continual_learning`, `physics_expert`

---

## 14 AutoResearch Subcommands

| Command | Purpose |
|---------|---------|
| `/plan` | Generate next experiment hypothesis from previous results |
| `/run` | Execute a single experiment iteration (modify → commit → run → eval → keep/discard) |
| `/fix` | Debug a crashed experiment — read stack trace, repair code |
| `/analyze` | Deep analysis of results: learning curves, overfitting, significance |
| `/ship` | Lock in best model: final eval, export checkpoint, generate submission |
| `/learn` | Extract lessons from past iterations |
| `/reason` | Chain-of-thought reasoning about experiment trajectory |
| `/probe` | Deep-dive into model internals (activations, gradients, attention) |
| `/improve` | Focused improvement on weakest cases |
| `/debug` | Interactive debugging session |
| `/evals` | Comprehensive evaluation suite (Dice, IoU, precision, recall) |
| `/regression` | Verify new changes don't break existing functionality |
| `/predict` | Predict outcome of proposed change before running |
| `/scenario` | Run what-if scenarios (different weather, time, geography) |

---

## 8 Specialized Scientific AI Agents

Each agent integrates domain skills (computer-vision, deep-learning, imaging-algorithms) and can be consulted during the autoresearch loop for expert advice.

| Agent | Role | Model | Expertise |
|-------|------|-------|-----------|
| **Research Literature** | Paper search & citations | `qwen/qwen3-coder:free` | SOTA methods, related work |
| **AutoResearch** | Experiment planning | `nemotron-3-ultra-550b-a55b:free` | Hypothesis, iteration strategy |
| **CV Expert** | Architecture design | `gemma-4-26b-a4b-it:free` | U-Net, DeepLab, SegFormer, augmentation |
| **DL Expert** | Training optimization | `hermes-3-llama-3.1-405b:free` | Loss functions, optimizers, schedulers |
| **LLM Expert** | Agent coordination | `qwen3-next-80b-a3b-instruct:free` | Prompt engineering, multi-agent |
| **Satellite Expert** | Remote sensing | `nemotron-nano-12b-v2-vl:free` | GOES-16 bands, false color, ERA5 |
| **Continual Learning** | Anti-forgetting | `nemotron-3-nano-omni-30b-a3b:free` | EWC, replay, checkpoint versioning |
| **Physics Expert** | Atmospheric physics | `nemotron-3-super-120b-a12b:free` | PINNs, advection, CSI metrics |

All agents use **free OpenRouter models** — zero API cost.

---

## Project Structure

```
MLSS26_HACKATHON/
├── AGENTS.md                     # Full agent documentation
├── README.md                     # This file
├── program.md                    # Task autoresearch protocol
├── .opencode/
│   └── skills/
│       ├── autoresearch/
│       │   └── SKILL.md          # Autoresearch skill (14 subcommands)
│       └── autoresearch_scientific/
│           └── SKILL.md          # 🚀 Scientific AI (agents + loop)
├── configs/
│   ├── agents.yaml               # 8 agents with model assignments
│   └── models.yaml               # 22 free OpenRouter models
├── MLAgentBench/                 # Forked codebase
│   ├── LLM.py                    # OpenRouter API router
│   ├── runner.py                 # Entry point with --agent-role flag
│   └── agents/
│       ├── agent.py              # Base Agent class
│       ├── agent_research.py     # ResearchAgent (original)
│       ├── agent_specialized.py  # 8 specialized agents + domain skills
│       ├── continual_learning.py # EWC + replay + versioning
│       └── orchestrator.py       # 🆕 Unified AutoResearch Orchestrator
├── experiments/                  # Experiment logs (TSV, JSONL, handoff)
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
│   ├── run_exp.py                # Standalone experiment CLI
│   ├── run_autoresearch_scientific.sh  # 🚀 Launch Scientific AI loop
│   ├── run_hackathon.sh          # Launch agent experiment
│   ├── start_dashboard.sh        # Start backend + frontend
│   └── setup.sh                  # Full environment setup
└── .venv/                        # Python virtual environment
```

---

## Task: Identify Contrails

- **Data**: GOES-16 ABI satellite imagery (bands 8-16), 256×256 patches  
- **Label**: Binary segmentation masks (contrail vs. no-contrail)  
- **Metric**: Dice Score  
- **Baseline**: `nn.Conv2d(3, 2, 1)` — single convolutional layer  
- **Current best**: **0.6000 Test Dice** (class weight tuning, [0.1, 15.0])
- **Kaggle**: https://kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming

---

## Quick Start

### 1. Setup
```bash
source .venv/bin/activate
export OPENROUTER_API_KEY=sk-or-v1-...
```

### 2. Run the AutoResearch loop
```bash
# Start the autonomous experiment loop
python -m MLAgentBench.runner \
  --task identify-contrails \
  --device 0 \
  --log-dir logs/autoresearch_run1 \
  --work-dir workspace \
  --agent-role autoresearch \
  --llm-name "nvidia/nemotron-3-ultra-550b-a55b:free" \
  --agent-max-steps 25
```

### 3. Or use a specialized agent
```bash
# Use the CV Expert for satellite imagery
bash scripts/run_hackathon.sh cv_expert identify-contrails 0
```

### 4. Start the dashboard
```bash
# Terminal 1 — Backend
source .venv/bin/activate
cd dashboard/backend && uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd dashboard/frontend && npm run dev

# Open http://localhost:3000
```

### 5. Run a standalone experiment
```bash
# Quick experiment with custom params
python scripts/run_exp.py --epochs 50 --base-ch 32 --lr 1e-4

# List past experiments
python scripts/run_exp.py --list
```

---

## Dashboard

| Page | Route | Feature |
|------|-------|---------|
| Overview | `/` | Live score chart, experiment stats by type |
| Experiments | `/experiments` | List all runs with source badges + filtering |
| Experiment Detail | `/experiments/[id]` | Score progression, run details, agent log |
| Agents | `/agents` | Agent status and model assignments |
| Config | `/config` | Swap LLM models per agent at runtime |
| Leaderboard | `/leaderboard` | Ranked experiment comparison with medals |

---

## AutoExperiment Loop (CLI)

Run the autonomous loop from the command line:

```bash
python scripts/run_exp.py --epochs 50   # Baseline
```

The orchestrator follows the Karpathy protocol:
1. **Precondition**: Verify git repo, clean tree, GPU
2. **Baseline**: Run once, record in TSV
3. **Loop**: Modify → Commit → Verify → Decide (keep/discard) → Log
4. **Handoff**: Write `experiments/loop-{date}/handoff.json`

Results are logged to `experiments/loop-{YYMMDD}-{HHMM}/results.tsv` and the dashboard.

---

## ERA5 Data (Amazon Basin)

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
