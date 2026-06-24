"""FastAPI backend for the MLSS26_HACKATHON dashboard.

Provides endpoints for:
- Experiment listing and details
- Score timeline data
- Visualization data for experiments
- Agent configuration and model swapping
- OpenRouter model listing
- WebSocket for real-time updates
"""
import os
import sys
import json
import glob
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"

app = FastAPI(title="MLSS26_HACKATHON Dashboard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelSwapRequest(BaseModel):
    model_id: str


def load_yaml(path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_models_config():
    return load_yaml(CONFIGS_DIR / "models.yaml")


def get_agents_config():
    return load_yaml(CONFIGS_DIR / "agents.yaml")


EXPERIMENTS_RUNS = PROJECT_ROOT / "experiments" / "runs.jsonl"

TASK_METRICS = {
    "flu": {
        "primary": "Test MAE", "secondary": "Val MAE",
        "tertiary": "Params", "color": "#f59e0b",
    },
}


def parse_runs_jsonl() -> list[dict]:
    experiments = []
    if not EXPERIMENTS_RUNS.exists():
        return experiments
    with open(EXPERIMENTS_RUNS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                model = data.get("model", "")
                exp = {
                    "id": str(hash(line))[-8:],
                    "path": str(EXPERIMENTS_RUNS),
                    "timestamp": data.get("timestamp", ""),
                    "steps": [],
                    "scores": [],
                    "final_score": data.get("test_acc") or data.get("test_mae"),
                    "status": "completed",
                    "source": "run_exp",
                    "task": "flu",
                    "details": {
                        "model": model,
                        "epochs": data.get("epochs", 20),
                        "lr": data.get("lr", 0.001),
                        "batch": data.get("batch", 64),
                        "params": data.get("params", 0),
                        "elapsed_s": data.get("elapsed_s", 0),
                        "test_acc": data.get("test_acc"),
                        "test_acc_id": data.get("test_acc_id"),
                        "val_acc": data.get("val_acc"),
                        "ood_f1": data.get("ood_f1"),
                        "ood_precision": data.get("ood_precision"),
                        "ood_recall": data.get("ood_recall"),
                        "test_mae": data.get("test_mae"),
                        "val_mae": data.get("val_mae"),
                    },
                }
                experiments.append(exp)
            except json.JSONDecodeError:
                continue
    return experiments


def split_tsv(line: str) -> list[str]:
    """Split a TSV line handling both real tabs and literal \\t strings."""
    if "\t" in line:
        return line.split("\t")
    if "\\t" in line:
        return line.split("\\t")
    return [line]


def load_reasoning(loop_dir: Path, iteration: int) -> dict:
    """Load hypothesis/mechanism/risk from iteration JSONs."""
    iter_file = loop_dir / "iterations" / f"iter-{iteration}.json"
    if not iter_file.exists():
        return {}
    try:
        data = json.loads(iter_file.read_text())
        jr = data.get("jury_reasoning", {})
        ch = data.get("change", {})
        return {
            "hypothesis": jr.get("hypothesis", ""),
            "mechanism": jr.get("mechanism", ""),
            "expected_delta": jr.get("expected_delta", ""),
            "risk": jr.get("risk", ""),
            "change_type": ch.get("type", ""),
            "diff_summary": ch.get("diff_summary", ""),
        }
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def parse_loop_results() -> list[dict]:
    experiments = []
    loop_dirs = sorted(PROJECT_ROOT.glob("experiments/loop-flu-*/results.tsv"), reverse=True)
    for tsv_path in loop_dirs:
        lines = tsv_path.read_text().strip().split("\n")
        if len(lines) < 2:
            continue
        loop_dir = tsv_path.parent
        loop_id = loop_dir.name

        headers = None
        data_start = 0
        for i, line in enumerate(lines):
            if not line or line.startswith("#"):
                data_start = i + 1
                continue
            low = line.lower().replace("\\t", "\t")
            if any(kw in low for kw in ["iteration", "commit", "timestamp", "test"]):
                headers = [h.strip().lower() for h in split_tsv(line)]
                data_start = i + 1
                break

        if headers:
            h = {name: idx for idx, name in enumerate(headers)}
            idx_iter = h.get("iteration", 0)
            idx_commit = h.get("commit", 1)
            idx_mae = h.get("test_mae", h.get("test_acc", 2))
            idx_val_mae = h.get("val_mae", 3)
            idx_status = h.get("status", 6)
            idx_desc = h.get("description", 7)
        else:
            idx_iter, idx_commit = 0, 1
            idx_mae, idx_val_mae = 2, 3
            idx_status, idx_desc = 6, 7

        iterations_data = []
        for line in lines[data_start:]:
            if not line or line.startswith("#"):
                continue
            parts = split_tsv(line)
            if len(parts) < 7:
                continue
            try:
                def pval(idx):
                    if idx is None or idx >= len(parts):
                        return None
                    v = parts[idx].strip()
                    if v and v not in (".", "-", "?"):
                        try:
                            return float(v)
                        except ValueError:
                            return None
                    return None

                iteration = int(parts[idx_iter].strip()) if idx_iter is not None and idx_iter < len(parts) else 0
                commit = parts[idx_commit].strip() if idx_commit is not None and idx_commit < len(parts) else ""
                test_mae = pval(idx_mae)
                val_mae = pval(idx_val_mae)
                status = parts[idx_status].strip() if idx_status < len(parts) else ""
                description = parts[idx_desc].strip() if idx_desc < len(parts) else ""

                reason = load_reasoning(loop_dir, iteration)

                iterations_data.append({
                    "iteration": iteration,
                    "commit": commit,
                    "test_mae": test_mae,
                    "val_mae": val_mae,
                    "status": status,
                    "description": description,
                    "hypothesis": reason.get("hypothesis", ""),
                    "mechanism": reason.get("mechanism", ""),
                    "expected_delta": reason.get("expected_delta", ""),
                    "risk": reason.get("risk", ""),
                    "change_type": reason.get("change_type", ""),
                    "diff_summary": reason.get("diff_summary", ""),
                })
            except (ValueError, IndexError):
                continue

        best_mae = min((i["test_mae"] for i in iterations_data if i["status"] == "keep" and i["test_mae"] is not None), default=None)
        first_mae = iterations_data[0]["test_mae"] if iterations_data else None

        experiments.append({
            "id": loop_id,
            "path": str(loop_dir),
            "timestamp": loop_id.replace("loop-", ""),
            "final_score": iterations_data[-1]["test_mae"] if iterations_data else None,
            "status": "completed",
            "source": "auto_loop",
            "task": "flu",
            "details": {
                "total_iterations": len(iterations_data),
                "kept": sum(1 for d in iterations_data if d["status"] == "keep"),
                "discarded": sum(1 for d in iterations_data if d["status"] == "discard"),
                "best_mae": best_mae,
                "first_mae": first_mae,
            },
            "iterations": iterations_data,
        })
    return experiments


def parse_experiment(log_dir: Path) -> dict:
    experiment = {
        "id": log_dir.name,
        "path": str(log_dir),
        "timestamp": datetime.fromtimestamp(log_dir.stat().st_mtime).isoformat()
                        if log_dir.exists() else None,
        "steps": [],
        "scores": [],
        "final_score": None,
        "status": "unknown",
        "agent_log_file": str(log_dir / "agent_log" / "main_log") if (log_dir / "agent_log" / "main_log").exists() else None,
    }
    agent_log = log_dir / "agent_log"
    env_log = log_dir / "env_log"

    main_log = agent_log / "main_log"
    if main_log.exists():
        content = main_log.read_text(errors="replace")
        step_count = content.count("Step ")
        experiment["total_steps"] = step_count
        if "Final message" in content:
            experiment["status"] = "completed"
        else:
            experiment["status"] = "running" if step_count > 0 else "pending"

    trace_file = env_log / "trace.json"
    if trace_file.exists():
        try:
            trace = json.loads(trace_file.read_text())
            steps = trace.get("steps", [])
            for i, step in enumerate(steps):
                step_info = {
                    "step": i,
                    "action": step.get("action", {}),
                    "observation": str(step.get("observation", ""))[:500],
                }
                experiment["steps"].append(step_info)
                score = step.get("score")
                if score is not None:
                    experiment["scores"].append({"step": i, "score": score})
            if experiment["scores"]:
                experiment["final_score"] = experiment["scores"][-1]["score"]
        except (json.JSONDecodeError, KeyError):
            pass

    overall_time = env_log / "overall_time.txt"
    if overall_time.exists():
        experiment["runtime"] = overall_time.read_text().strip()

    return experiment


def get_all_experiments(task: str = "flu"):
    experiments = []
    for exp in parse_loop_results():
        if task is None or exp.get("task") == task:
            experiments.append(exp)
    for exp in parse_runs_jsonl():
        if task is None or exp.get("task") == task:
            experiments.append(exp)
    if LOGS_DIR.exists():
        for log_dir in sorted(LOGS_DIR.iterdir(), reverse=True):
            if log_dir.is_dir():
                exp = parse_experiment(log_dir)
                if task is None or exp.get("task") == task:
                    experiments.append(exp)
    experiments.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return experiments


def find_experiment_by_id(experiment_id: str) -> Optional[dict]:
    # Check LOGS_DIR
    if LOGS_DIR.exists():
        log_dir = LOGS_DIR / experiment_id
        if log_dir.exists() and log_dir.is_dir():
            return parse_experiment(log_dir)
    # Check runs.jsonl
    for exp in parse_runs_jsonl():
        if exp["id"] == experiment_id:
            return exp
    # Check loop results
    for exp in parse_loop_results():
        if exp["id"] == experiment_id:
            return exp
    return None


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.get("/")
async def root():
    return {"status": "ok", "message": "MLSS26_HACKATHON Dashboard API v2"}


@app.get("/experiments")
async def list_experiments(task: Optional[str] = None):
    return {"experiments": get_all_experiments(task)}


@app.get("/tasks")
async def list_tasks():
    return {"tasks": TASK_METRICS, "default": "flu"}


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    exp = find_experiment_by_id(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@app.get("/experiments/{experiment_id}/iterations")
async def get_experiment_iterations(experiment_id: str):
    exp = find_experiment_by_id(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    iterations = exp.get("iterations", [])
    return {"experiment_id": experiment_id, "iterations": iterations}


@app.get("/experiments/{experiment_id}/reasoning")
async def get_experiment_reasoning(experiment_id: str):
    exp = find_experiment_by_id(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    iterations = exp.get("iterations", [])
    reasoning = []
    for it in iterations:
        it_num = it.get("iteration", 0)
        reasoning.append({
            "iteration": it_num,
            "status": it.get("status", ""),
            "test_mae": it.get("test_mae"),
            "description": it.get("description", ""),
            "hypothesis": it.get("hypothesis", ""),
            "mechanism": it.get("mechanism", ""),
            "expected_delta": it.get("expected_delta", ""),
            "risk": it.get("risk", ""),
            "change_type": it.get("change_type", ""),
            "diff_summary": it.get("diff_summary", ""),
        })
    return {"experiment_id": experiment_id, "reasoning": reasoning}


@app.get("/scores")
async def get_scores(experiment_id: Optional[str] = None):
    if experiment_id:
        exp = find_experiment_by_id(experiment_id)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return {"experiment_id": experiment_id, "scores": exp.get("scores", [])}
    experiments = get_all_experiments()
    all_scores = {}
    for exp in experiments:
        if exp.get("scores"):
            all_scores[exp["id"]] = exp["scores"]
    return {"all_scores": all_scores}


@app.get("/experiments/{experiment_id}/viz")
async def get_experiment_viz(experiment_id: str):
    exp = find_experiment_by_id(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp.get("source") == "auto_loop":
        loop_dir = Path(exp["path"])
        viz_file = loop_dir / "viz" / "data.json"
        if viz_file.exists():
            return json.loads(viz_file.read_text())
    return {}


@app.get("/agents")
async def list_agents():
    config = get_agents_config()
    agents = config.get("agents", {})
    result = []
    for name, cfg in agents.items():
        result.append({
            "name": name,
            "display_name": cfg.get("name", name),
            "description": cfg.get("description", ""),
            "model": cfg.get("model", ""),
            "fast_model": cfg.get("fast_model", ""),
            "upgrade_model": cfg.get("upgrade_model", ""),
            "skills": cfg.get("skills", []),
        })
    return {"agents": result}


@app.post("/agents/{agent_name}/model")
async def swap_agent_model(agent_name: str, request: ModelSwapRequest):
    config = get_agents_config()
    agents = config.get("agents", {})
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    agents[agent_name]["model"] = request.model_id
    config["agents"] = agents
    with open(CONFIGS_DIR / "agents.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    await manager.broadcast({"type": "model_swap", "agent": agent_name, "model": request.model_id})
    return {"status": "ok", "agent": agent_name, "model": request.model_id}


@app.get("/models")
async def list_models():
    config = get_models_config()
    return {
        "free_models": config.get("tiers", {}).get("free", []),
        "premium_models": config.get("tiers", {}).get("premium", []),
        "default_model": config.get("default_model", ""),
        "default_fast_model": config.get("default_fast_model", ""),
    }


@app.get("/status")
async def get_status():
    experiments = get_all_experiments()
    config = get_agents_config()
    return {
        "total_experiments": len(experiments),
        "total_agents": len(config.get("agents", {})),
        "logs_dir": str(LOGS_DIR),
        "logs_exist": LOGS_DIR.exists(),
        "timestamp": datetime.now().isoformat(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
