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

def infer_task(exp: dict) -> str:
    details = exp.get("details", {})
    model = str(details.get("model", "")).lower()
    if any(k in model for k in ["lstm", "gru", "tcn", "transformer", "diffusion"]):
        return "flu"
    if "val_mae" in details or "test_mae" in details:
        return "flu"
    return "medmnist"


TASK_METRICS = {
    "medmnist": {
        "primary": "OOD F1", "secondary": "ID Test Acc",
        "tertiary": "Val Acc", "color": "#8b5cf6",
    },
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
                model = data.get("model", "SimpleCNN")
                is_flu = any(k in model.lower() for k in ["lstm", "gru", "tcn", "transformer", "diffusion"])
                exp = {
                    "id": str(hash(line))[-8:],
                    "path": str(EXPERIMENTS_RUNS),
                    "timestamp": data.get("timestamp", ""),
                    "steps": [],
                    "scores": [],
                    "final_score": data.get("test_acc") or data.get("test_mae"),
                    "status": "completed",
                    "source": "run_exp",
                    "task": "flu" if is_flu else "medmnist",
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


def parse_loop_results() -> list[dict]:
    experiments = []
    loop_dirs = sorted(PROJECT_ROOT.glob("experiments/loop-*/results.tsv"), reverse=True)
    for tsv_path in loop_dirs:
        lines = tsv_path.read_text().strip().split("\n")
        if len(lines) < 2:
            continue
        loop_id = tsv_path.parent.name
        is_flu = "flu" in loop_id.lower()

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
            idx_commit = h.get("commit", h.get("timestamp", 1))
            idx_metric = h.get("test_mae", h.get("test_acc", h.get("metric", 2)))
            idx_metric2 = h.get("val_mae", h.get("ood_f1", h.get("secondary", 3)))
            idx_metric3 = h.get("val_acc", 4)
            idx_metric4 = h.get("test_acc_id", h.get("test_id_acc", 5))
            idx_status = h.get("status", 7)
            idx_desc = h.get("description", 8)
            # If idx_metric4 points to the params column, null it out
            if idx_metric4 is not None and idx_metric4 < len(headers):
                col_name = headers[idx_metric4]
                if col_name in ("params", "memory_gb", "timestamp", "status", "commit"):
                    idx_metric4 = None
            # Ensure val_acc doesn't overlap with params
            if idx_metric3 is not None and idx_metric3 < len(headers):
                col_name3 = headers[idx_metric3]
                if col_name3 in ("params", "memory_gb", "timestamp", "status", "commit"):
                    idx_metric3 = None
        else:
            idx_iter, idx_commit = 0, 1
            idx_metric, idx_metric2, idx_metric3, idx_metric4 = 2, 3, 4, 5
            idx_status, idx_desc = 7, 8

        scores = []
        iterations_data = []
        final_score = None
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

                iteration = parts[idx_iter].strip() if idx_iter is not None and idx_iter < len(parts) else ""
                commit = parts[idx_commit].strip() if idx_commit is not None and idx_commit < len(parts) else ""
                m1 = pval(idx_metric)
                m2 = pval(idx_metric2)
                m3 = pval(idx_metric3)
                m4 = pval(idx_metric4)
                memory_gb = parts[6].strip() if len(parts) > 6 else ""
                status = parts[idx_status].strip() if idx_status < len(parts) else ""
                description = parts[idx_desc].strip() if idx_desc < len(parts) else ""

                metric = m1 if m1 is not None else (m2 if m2 is not None else (m4 if m4 is not None else 0.0))
                if metric is not None and not is_flu:
                    scores.append({"step": int(iteration) if iteration else 0, "score": metric})
                    final_score = metric

                iterations_data.append({
                    "iteration": int(iteration) if iteration else 0,
                    "commit": commit,
                    "test_acc": m1,
                    "ood_f1": m2,
                    "val_acc": m3,
                    "test_acc_id": m4,
                    "memory_gb": memory_gb,
                    "status": status,
                    "description": description,
                })
            except (ValueError, IndexError):
                continue

        experiments.append({
            "id": loop_id,
            "path": str(tsv_path.parent),
            "timestamp": loop_id.replace("loop-", ""),
            "steps": [{"step": i, "action": {}, "observation": ""} for i in range(len(scores))],
            "scores": scores,
            "final_score": final_score,
            "status": "completed" if scores else "unknown",
            "source": "auto_loop",
            "task": "flu" if is_flu else "medmnist",
            "details": {
                "iterations": iterations_data,
                "total_iterations": len(iterations_data),
                "kept": sum(1 for d in iterations_data if d["status"] == "keep"),
                "discarded": sum(1 for d in iterations_data if d["status"] == "discard"),
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


def get_all_experiments(task: str = None):
    experiments = []
    if LOGS_DIR.exists():
        for log_dir in sorted(LOGS_DIR.iterdir(), reverse=True):
            if log_dir.is_dir():
                exp = parse_experiment(log_dir)
                if task is None or exp.get("task") == task:
                    experiments.append(exp)
    for exp in parse_runs_jsonl():
        if task is None or exp.get("task") == task:
            experiments.append(exp)
    for exp in parse_loop_results():
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
    return {"tasks": TASK_METRICS}


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
