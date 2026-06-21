# MLSS26_HACKATHON Implementation Plan

## Status: Ready to Execute

All research complete. API keys collected. Dependencies installed. Ready for file creation once plan mode is exited.

---

## What's Already Done (via bash commands)

1. **MLAgentBench forked** — copied from `snap-stanford/MLAgentBench` into workspace
2. **MLRC-Bench improvements merged** — `constants.py`, `utils.py`, `LLM_as_a_Judge.py`, `llm_test_cases.py`, `capability_level.py`, `capability_levels.json`, `benchmarks_base/` directory, improved `LLM.py` and `runner.py`
3. **Python venv created** — `.venv/` with Python 3.10
4. **Dependencies installed** — PyTorch 2.6.0+cu124, torchvision, scikit-learn, pandas, matplotlib, seaborn, tqdm, kaggle, openai, anthropic, transformers, pydantic, dacite, pyyaml, requests, python-dotenv, h5py, netcdf4, cdsapi, fastapi, uvicorn, websockets, sqlalchemy
5. **Project directories created** — `configs/`, `data/era5/`, `dashboard/backend/`, `dashboard/frontend/`
6. **ERA5 data downloaded** — 2 NetCDF files (~209MB each, 2023 & 2024 pressure levels)



**First step when implementation begins**: Verify Kaggle credentials work by running `kaggle competitions list -s contrails`. If the `KGAT_` prefix causes issues, may need to use the key without prefix or try alternative auth method.

---

## Files to Create

### 1. `configs/models.yaml` — OpenRouter Model Configuration

All 27 free OpenRouter models with metadata (context length, modalities, best_for tags). Three tiers:
- **free**: 27 models (default tier)
- **premium**: 3 upgrade path models (gpt-4o, claude-sonnet-4, claude-3.5-sonnet)
- Default model: `meta-llama/llama-3.3-70b-instruct:free`
- Default fast model: `openai/gpt-oss-20b:free`
- Default multimodal: `google/gemma-4-26b-a4b-it:free`

### 2. `configs/agents.yaml` — Agent Role → Model Mappings

8 specialized agents:

| Agent | Role | Default Free Model | Why |
|-------|------|-------------------|-----|
| research_literature | Paper search, citations, literature review | `qwen/qwen3-coder:free` (1M ctx, strong text) | Long context for paper summaries |
| autoresearch | Experiment planning, hypothesis generation | `nvidia/nemotron-3-ultra-550b-a55b:free` (1M ctx) | Largest free model, complex reasoning |
| cv_expert | Image preprocessing, augmentation, CNN architectures | `google/gemma-4-26b-a4b-it:free` (multimodal) | Can process images directly |
| dl_expert | Training loops, loss functions, optimizers | `nousresearch/hermes-3-llama-3.1-405b:free` (405B) | Largest params, deep learning reasoning |
| llm_expert | Prompt engineering, multimodal reasoning, coordination | `qwen/qwen3-next-80b-a3b-instruct:free` | Strong instruction following |
| satellite_expert | Remote sensing, spectral analysis, geospatial transforms | `nvidia/nemotron-nano-12b-v2-vl:free` (multimodal) | Vision model for satellite imagery |
| continual_learning | Anti-forgetting, EWC/replay, model versioning | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` (multimodal + reasoning) | Reasoning for when to commit/rollback |
| physics_expert | Physics-informed constraints, advection, CSI metrics | `nvidia/nemotron-3-super-120b-a12b:free` (1M ctx) | Strong math/physics reasoning |

### 3. `.env.example` — Environment Variable Template

```
OPENROUTER_API_KEY=sk-or-v1-...
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=KGAT_f65dea602f3ba5c17df27cf56aa6f75e
KAGGLE_CONFIG_DIR=.kaggle
FAST_MODEL=openai/gpt-oss-20b:free
LOG_DIR=logs
```

### 4. `.gitignore` — Update Existing

Add: `.env`, `.venv/`, `.kaggle/`, `*.zip`, `data/`, `workspace/`, `logs/`, `__pycache__/`, `node_modules/`, `.next/`

### 5. `MLAgentBench/LLM.py` — Add OpenRouter API Support

Add `complete_text_openrouter()` function that:
- Uses `openai` library with `base_url="https://openrouter.ai/api/v1"`
- Reads API key from env var `OPENROUTER_API_KEY`
- Routes any model ID containing `:free` or starting with known provider prefixes to OpenRouter
- Supports cost tracking (free models = $0)
- Handles rate limits with retries

Also modify `complete_text()` routing logic to check for OpenRouter models first.

### 6. `MLAgentBench/agents/agent_specialized.py` — Specialized Agent Base

Base class for specialized agents that:
- Extends `ResearchAgent` with role-specific system prompts
- Loads model config from `configs/agents.yaml`
- Has specialized action sets per role
- Integrates with continual learning loop

### 7. `MLAgentBench/agents/orchestrator.py` — Agent Orchestrator

Coordinates multiple specialized agents:
- Routes subproblems to the right agent based on content
- Manages agent state across iterations
- Integrates continual learning checkpoint management
- Logs all agent activities for dashboard

### 8. `MLAgentBench/agents/continual_learning.py` — Continual Learning Manager

Manages model updates across iterations:
- **Before each iteration**: Load previous best checkpoint
- **During training**: Log parameter changes (Fisher information for EWC)
- **After training**: Evaluate improvement vs forgetting
- **Decision**: Commit if improvement > threshold AND forgetting < threshold; rollback otherwise
- **Memory**: Maintain model registry with version scores
- **Replay buffer**: Store exemplar samples for replay during fine-tuning

### 9. `AGENTS.md` — Agent Documentation

Covers:
- All 8 agent roles, skills, and model mappings
- OpenRouter configuration (free models, upgrade paths)
- Continual learning loop explanation
- ERA5 data pipeline instructions
- Dashboard usage guide
- How to add new agents
- How to swap models at runtime via dashboard

### 10. `dashboard/backend/main.py` — FastAPI Backend

Endpoints:
- `GET /experiments` — list all experiment runs
- `GET /experiments/{id}` — detailed run info
- `GET /scores` — score timeline data for charts
- `POST /agents/{agent}/model` — swap LLM model per agent
- `WS /ws` — WebSocket for real-time updates
- `GET /agents` — list all agents and their configs
- `GET /models` — list available OpenRouter models

SQLite database for experiment storage. Reads MLAgentBench log directories.

### 11. `dashboard/frontend/` — Next.js Dashboard

Pages:
- `/` — Overview with live score chart
- `/experiments` — Experiment list with filtering
- `/experiments/[id]` — Detailed run view with agent activity log
- `/agents` — Agent activity timeline
- `/config` — Model configuration (swap LLMs per agent)
- `/leaderboard` — Ranked comparison of runs

Components: ScoreChart (recharts), AgentTimeline, ModelSwapPanel, LeaderboardTable

### 12. `scripts/download_era5.py` — ERA5 Data Pipeline

Uses `cdsapi` to download ERA5 single-levels (precipitation) data for Amazon region. Already have pressure-levels data downloaded.

### 13. `scripts/run_hackathon.sh` — Main Launch Script

Activates venv, sets environment variables, launches MLAgentBench runner with identify-contrails task, starts dashboard backend and frontend.

### 14. `scripts/setup.sh` — Full Installation Script

For new team members: install venv, dependencies, download data, configure API keys.

---

## Implementation Order

1. **Configs** → `models.yaml`, `agents.yaml`, `.env.example`, `.gitignore`
2. **OpenRouter** → Modify `LLM.py` to add OpenRouter API support
3. **Agents** → Create 8 specialized agents + orchestrator + continual learning
4. **AGENTS.md** → Comprehensive documentation
5. **Kaggle** → Download contrails dataset (pending username)
6. **Dashboard backend** → FastAPI with SQLite
7. **Dashboard frontend** → Next.js with real-time updates
8. **ERA5 pipeline** → Data processing scripts (low priority)

---

## Hardware

- 2× NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- Python 3.10.12
- PyTorch 2.6.0+cu124 installed in venv