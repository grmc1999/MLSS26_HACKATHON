# MLSS26 Hackathon — Scientific AI AutoResearch (Chest X-ray OOD)

An autonomous **Scientific AI AutoResearch** system for medical chest X-ray classification with **Out-of-Distribution (OOD) detection**. Trains on PneumoniaMNIST (2 classes: normal, pneumonia), evaluates on ChestMNIST (3 classes: normal, pneumonia, **consolidation — an unseen OOD class**).

Uses **free OpenRouter models** to power 8 specialized agents that collaborate to improve the OOD detection model.

---

## Task: Chest X-ray OOD Detection

| | |
|---|---|
| **Train** | PneumoniaMNIST (4,708 samples, 28×28) |
| **Classes** | normal, pneumonia |
| **Test** | ChestMNIST 3-class subset (600 samples) |
| **Classes** | normal (300), pneumonia (59), **consolidation 🆕 OOD** (241) |
| **Goal** | Maximize OOD F1 + in-distribution accuracy |
| **Baseline** | Test acc: 22%, OOD F1: 0.15 |

The scientific challenge: a model trained only on PneumoniaMNIST must:
1. Correctly classify normal vs pneumonia from ChestMNIST (domain transfer)
2. Detect consolidation as **OOD** despite looking nearly identical to pneumonia on X-rays

---

## Architecture

```
User / Dashboard
       |
       v
+------------------------------+
|    Scientific AutoResearch   |
|    Orchestrator              |  ← modify → train → eval → keep/discard
+------------------------------+
       |         ↑
       |         | consultation (route_to_agent)
       v         |
+------------------------------+
|    8 Specialized Agents     |  ← domain experts
+------------------------------+
       |
       v
+------------------------------+
|   Experiment Pipeline       |  ← scripts/run_medmnist.py
+------------------------------+
```

---

## 8 Specialized Agents

| Agent | Role | Focus for this Task |
|-------|------|---------------------|
| **CV Expert** | Architecture design | CNN for 28x28, OOD detectors |
| **DL Expert** | Training optimization | Confidence calibration, thresholds |
| **LLM Expert** | Agent coordination | Multi-agent research synthesis |
| **Continual Learning** | Anti-forgetting | Domain adaptation across datasets |
| **AutoResearch** | Experiment planning | Loop strategy, hypothesis |
| **Research Literature** | Paper search | OOD detection, medical transfer learning |
| **Satellite Expert** → **Medical Expert** | Chest X-ray analysis | X-ray modality, pneumonia patterns |
| **Physics Expert** → **Robustness Expert** | Uncertainty quantification | OOD scoring, Mahalanobis distance |

---

## 15 AutoResearch Subcommands

| Command | Purpose |
|---------|---------|
| `/autoresearch` | Iterate against metric: modify → verify → keep/discard |
| `/autoresearch_plan` | Generate next hypothesis |
| `/autoresearch_run` | Execute single iteration |
| `/autoresearch_fix` | Debug crashes |
| `/autoresearch_analyze` | Deep analysis of results |
| `/autoresearch_ship` | Lock best model |
| `/autoresearch_learn` | Extract cross-iteration lessons |
| `/autoresearch_evals` | Comprehensive metrics |
| `/autoresearch_probe` | Model internals |
| `/autoresearch_improve` | Targeted improvements |
| `/autoresearch_debug` | Interactive debugging |
| `/autoresearch_reason` | Trajectory analysis |
| `/autoresearch_scenario` | What-if scenarios |
| `/autoresearch_regression` | Regression testing |
| `/autoresearch_scientific` | 🧪 Full loop + 8 agents |

---

## Quick Start

### 1. Setup
```bash
source .venv/bin/activate
export OPENROUTER_API_KEY=sk-or-v1-...
```

### 2. Run baseline experiment
```bash
python scripts/run_medmnist.py --epochs 20
```

### 3. Run the autonomy loop
```bash
python -m MLAgentBench.agents.orchestrator \
    --agent autoresearch \
    --iterations 25 \
    --verify "python scripts/run_medmnist.py --epochs 20"
```

### 4. Start dashboard
```bash
# Terminal 1
cd dashboard/backend && uvicorn main:app --port 8000

# Terminal 2
cd dashboard/frontend && npm run dev
# → http://localhost:3000
```

---

## Data

| Dataset | Source | Samples | Role |
|---------|--------|---------|------|
| **PneumoniaMNIST** | MedMNIST (auto-download) | 4,708 train + 524 val | Training |
| **ChestMNIST subset** | Extracted from MedMNIST | 600 test | Evaluation |
| Path: `data/medmnist_subset/chestmnist_3class.npz` (432 KB) | | | |

---

## Project Structure

```
MLSS26_HACKATHON/
├── AGENTS.md
├── README.md
├── program.md
├── .opencode/skills/autoresearch/SKILL.md     # 15 subcommands
├── configs/agents.yaml                         # Agents config
├── configs/models.yaml                         # OpenRouter models
├── MLAgentBench/agents/
│   ├── orchestrator.py                         # Unified loop
│   ├── agent_specialized.py                    # 8 agents
│   └── continual_learning.py
├── MLAgentBench/benchmarks/medmnist/           # 🆕 Current task
│   ├── env/train.py                            # Training script
│   └── env/loader.py                           # Data loader
├── data/medmnist_subset/                       # ChestMNIST 3-class subset
├── scripts/
│   ├── run_medmnist.py                         # 🆕 Experiment CLI
│   └── run_autoresearch_scientific.sh          # Scientific AI launcher
├── experiments/                                # Results
├── dashboard/
│   ├── backend/ (FastAPI)
│   └── frontend/ (Next.js)
└── .venv/
```

---

## Dashboard

| Page | Route | Feature |
|------|-------|---------|
| Overview | `/` | Score chart, experiment stats |
| Experiments | `/experiments` | List with source badges |
| Experiment Detail | `/experiments/[id]` | Score + agent log |
| Agents | `/agents` | Status and models |
| Config | `/config` | Model swap panel |
| Leaderboard | `/leaderboard` | Ranked by OOD F1 |

---

## License

MIT — inherited from [MLAgentBench](https://github.com/snap-stanford/MLAgentBench).
