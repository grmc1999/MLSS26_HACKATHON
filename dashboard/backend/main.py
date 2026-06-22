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
                exp = {
                    "id": str(hash(line))[-8:],
                    "path": str(EXPERIMENTS_RUNS),
                    "timestamp": data.get("timestamp", ""),
                    "steps": [],
                    "scores": [],
                    "final_score": data.get("test_acc"),
                    "status": "completed",
                    "source": "run_exp",
                    "details": {
                        "model": data.get("model", "SimpleCNN"),
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
                    },
                }
                experiments.append(exp)
            except json.JSONDecodeError:
                continue
    return experiments


def parse_loop_results() -> list[dict]:
    experiments = []
    loop_dirs = sorted(PROJECT_ROOT.glob("experiments/loop-*/results.tsv"), reverse=True)
    for tsv_path in loop_dirs:
        lines = tsv_path.read_text().strip().split("\n")
        if len(lines) < 2:
            continue
        loop_id = tsv_path.parent.name
        scores = []
        iterations_data = []
        final_score = None
        for i, line in enumerate(lines[1:]):
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                try:
                    iteration = parts[0].strip()
                    commit = parts[1].strip()
                    test_acc = float(parts[2]) if parts[2] else None
                    ood_f1 = float(parts[3]) if parts[3] else None
                    memory_gb = parts[4].strip() if len(parts) > 4 else ""
                    status = parts[5].strip() if len(parts) > 5 else ""
                    description = parts[6].strip() if len(parts) > 6 else ""
                    metric = test_acc if test_acc is not None else 0.0
                    scores.append({"step": int(iteration), "score": metric})
                    final_score = metric
                    iterations_data.append({
                        "iteration": int(iteration),
                        "commit": commit,
                        "test_acc": test_acc,
                        "ood_f1": ood_f1,
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


def get_all_experiments():
    experiments = []
    if LOGS_DIR.exists():
        for log_dir in sorted(LOGS_DIR.iterdir(), reverse=True):
            if log_dir.is_dir():
                experiments.append(parse_experiment(log_dir))
    experiments.extend(parse_runs_jsonl())
    experiments.extend(parse_loop_results())
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
async def list_experiments():
    return {"experiments": get_all_experiments()}


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
