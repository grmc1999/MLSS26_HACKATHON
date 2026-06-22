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
| **Baseline** | ID test acc: ~0.71, OOD F1: ~0.12 |

The scientific challenge: a model trained only on PneumoniaMNIST must:
1. Correctly classify normal vs pneumonia from ChestMNIST (domain transfer)
2. Detect consolidation as **OOD** despite looking nearly identical to pneumonia on X-rays

---

## Baseline: SimpleCNN

The starting model is a 2-layer CNN (`SimpleCNN`) with batch norm, max-pooling, dropout, and leaky ReLU activations — designed for 28×28 grayscale chest X-rays.

```
Conv2D(1→32) → BN → LeakyReLU → MaxPool(2×2)
Conv2D(32→64) → BN → LeakyReLU → MaxPool(2×2)
Flatten → Dropout → FC(3136→128) → LeakyReLU → Dropout → FC(128→3)
```

**Key design choices:**
- **3 output logits** (not 2) — the third logit is reserved for OOD detection via softmax thresholding
- **LeakyReLU(0.1)** instead of ReLU to prevent dead neurons on low-contrast X-rays
- **Dropout(0.25)** on both the flattened features and the hidden layer for regularization
- **Adam optimizer** (lr=1e-3), **CrossEntropyLoss**, 20 epochs, batch size 64

**OOD detection**: after training, the model's softmax probabilities are thresholded at 0.7 — any sample with max probability < 0.7 is flagged as OOD (consolidation). This requires the model to be well-calibrated for reliable confidence-based detection.

**Baseline performance** (on ChestMNIST 3-class test set):
| Metric | Value |
|--------|-------|
| ID Test Acc (normal + pneumonia) | ~0.71 |
| OOD F1 | ~0.12 |
| Val Accuracy (PneumoniaMNIST) | ~0.975 |
| Parameters | 420,931 |

Improvements are measured against this baseline. The model's main limitation is the 3 logit architecture — the third logit receives no direct training signal, making it unstable for OOD detection. SimpleCNN has limited capacity for the domain shift between PneumoniaMNIST (training) and ChestMNIST (test).

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

## 15 AutoResearch Slash Commands

Available as opencode commands (type `/` in the TUI). Defined in `.opencode/commands/autoresearch_*.md`.

| Command | Purpose |
|---------|---------|
| `/autoresearch` | Iterate against metric: modify → verify → keep/discard |
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
| `/autoresearch_scientific` | 🧪 Full loop + 8 specialized agents |

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
├── models/
│   ├── Qwen3-VL-Embedding-2B/                  # Visual RAG embedding model (2.1B)
│   └── Qwen3-VL-Embedding-LoRA/                # LoRA adapters (lora_vit, dora_ls005, hyper3)
└── .venv/
```

---

## Visual RAG — Medical Literature Retrieval

PixelRAG provides visual retrieval over medical literature screenshots. The embedding model (`Qwen3-VL-Embedding-2B`) and LoRA adapters are at `models/`.

### Pipeline (agentic workflow)
1. **Render** medical PDFs as screenshot tiles (`pixelshot`)
2. **Embed** tiles with Qwen3-VL-Embedding-2B
3. **Index** embeddings in FAISS
4. **Retrieve** relevant papers before each train.py modification

### Fine-tuning
```bash
cd /tmp/PixelRAG/train && uv sync
# Modify train.py to point to your dataset
# LoRA on Qwen3-VL-Embedding-2B with torch.compile
uv run python train.py
```

## Dashboard

| Page | Route | Feature |
|------|-------|---------|
| Overview | `/` | Score chart, experiment stats |
| Experiments | `/experiments` | List with source badges |
| Experiment Detail | `/experiments/[id]` | Score + agent log, PCA embeddings, per-class accuracy |
| Agents | `/agents` | Status and models |
| Config | `/config` | Model swap panel |
| Leaderboard | `/leaderboard` | Ranked by OOD F1 |

---

## License

MIT — inherited from [MLAgentBench](https://github.com/snap-stanford/MLAgentBench).
