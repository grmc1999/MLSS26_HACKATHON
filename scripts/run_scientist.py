#!/usr/bin/env python3
"""AI Scientist — LangGraph-based orchestrator with Ollama.
Inspired by AI-Scientist-v2 (tree search) and EvoScientist (multi-agent memory).

Phases in a LangGraph state machine:
  Research → Ideate → Plan → Implement → Jury → Review → Run → Decide → Log
+ Parallel tree search (try 2 proposals, keep best)
+ Persistent lessons memory (lessons.json)
+ Full context (run.log, results.tsv, active model)

Usage:
    python scripts/run_scientist.py --task flu
    python scripts/run_scientist.py --task medmnist --model ollama
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Optional, Annotated
import operator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from MLAgentBench.agents.agent_specialized import search_medical_literature, load_expert

# ── Config ──

TASKS = {
    "medmnist": {
        "runner": "python scripts/run_medmnist.py --epochs 25",
        "metric_pattern": r"OOD F1:\s*([\d.]+)",
        "secondary_pattern": r"Test ID Acc:\s*([\d.]+)",
        "direction": "lower",
        "lit_md_dir": "literature_md",
        "index_dir": "index_output",
    },
    "flu": {
        "runner": "python scripts/run_exp.py --epochs 30 --lr 0.001 --hidden-dim 64 --model gru --num-layers 3",
        "metric_pattern": r"Test MAE:\s*([\d.]+)",
        "secondary_pattern": r"Val MAE:\s*([\d.]+)",
        "direction": "lower",
        "lit_md_dir": "literature_flu_md",
        "index_dir": "index_output_flu",
    },
}

# ── State ──

class ScientistState(TypedDict):
    task: str
    iteration: int
    best_metric: float
    best_desc: str
    current_model: str
    history: List[dict]
    lessons: List[str]
    run_log_tail: str
    active_model: str
    loop_dir: str
    max_iter: float
    proposal_a: Optional[str]
    proposal_b: Optional[str]
    result_a: Optional[float]
    result_b: Optional[float]
    final_metric: float
    status: str

# ── LLM helpers ──

def ask_ollama(prompt, model="qwen2.5:7b", max_tokens=512):
    import ollama
    resp = ollama.generate(model=model, prompt=prompt, options={"num_predict": max_tokens, "temperature": 0.7})
    return resp["response"].strip()

def ask_transformers(prompt, orch_model, orch_tok, max_tokens=512):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    messages = [{"role": "user", "content": prompt}]
    text = orch_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = orch_tok([text], return_tensors="pt").to(orch_model.device)
    with torch.no_grad():
        out = orch_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return orch_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# ── Graph nodes ──

def load_lesson_memory(state: ScientistState) -> ScientistState:
    loop_dir = Path(state["loop_dir"])
    lessons_file = loop_dir / "lessons.json"
    if lessons_file.exists():
        state["lessons"] = json.loads(lessons_file.read_text())
    else:
        state["lessons"] = []
    return state

def research(state: ScientistState) -> ScientistState:
    task = state["task"]
    print(f"\n{'='*50}\n  Iter {state['iteration']}: Research\n{'='*50}")
    try:
        rag = search_medical_literature(f"improve {task} forecasting", k=3, task=task)
        rag_ctx = "\n".join(f"  [{r['score']:.3f}] {r.get('title','?')}" for r in rag)
    except Exception as e:
        rag_ctx = f"  (RAG: {e})"
    print(rag_ctx)

    # Read run.log tail
    log_path = ROOT / "run.log"
    if log_path.exists():
        state["run_log_tail"] = subprocess.run(["tail", "-30", str(log_path)], capture_output=True, text=True).stdout
    return state

def ideate(state: ScientistState) -> ScientistState:
    print(f"\n  Ideate")
    task = state["task"]
    cfg = TASKS[task]

    prompt = f"""Task: {task}. Best metric: {state['best_metric']:.4f}.
Current model: {state['current_model']}
Recent run.log tail: {state.get('run_log_tail', 'N/A')[:500]}

Lessons learned: {chr(10).join(state.get('lessons', [])[-3:])}

Generate TWO distinct proposals to improve the metric.
Each must be DIFFERENT in approach (one architecture, one hyperparameter).
Format exactly:
=== PROPOSAL A ===
CHANGE: <description>
DELTA: <expected>
=== PROPOSAL B ===
CHANGE: <description>
DELTA: <expected>"""

    if state.get("active_model") == "ollama":
        response = ask_ollama(prompt)
    else:
        response = ask_transformers(prompt, orch_model, orch_tok)

    # Parse proposals
    parts = response.split("=== PROPOSAL")
    for part in parts:
        if part.startswith(" A"):
            state["proposal_a"] = part[3:].strip()
        elif part.startswith(" B"):
            state["proposal_b"] = part[3:].strip()
    print(f"  A: {state.get('proposal_a','')[:80]}...")
    print(f"  B: {state.get('proposal_b','')[:80]}...")
    return state

def run_proposal_a(state: ScientistState) -> ScientistState:
    print(f"\n  Run A")
    cfg = TASKS[state["task"]]
    result = subprocess.run(f"cd {ROOT} && {cfg['runner']} > run_a.log 2>&1", shell=True, timeout=600)
    log = Path("run_a.log").read_text() if Path("run_a.log").exists() else ""
    m = re.search(cfg["metric_pattern"], log)
    state["result_a"] = float(m.group(1)) if m else None
    return state

def run_proposal_b(state: ScientistState) -> ScientistState:
    print(f"\n  Run B")
    cfg = TASKS[state["task"]]
    result = subprocess.run(f"cd {ROOT} && {cfg['runner']} > run_b.log 2>&1", shell=True, timeout=600)
    log = Path("run_b.log").read_text() if Path("run_b.log").exists() else ""
    m = re.search(cfg["metric_pattern"], log)
    state["result_b"] = float(m.group(1)) if m else None
    return state

def decide(state: ScientistState) -> ScientistState:
    print(f"\n  Decide")
    cfg = TASKS[state["task"]]
    best_proposal = None
    best_result = float("inf")

    for name, result, proposal in [("A", state["result_a"], state["proposal_a"]), ("B", state["result_b"], state["proposal_b"])]:
        if result is not None and result < state["best_metric"] and result < best_result:
            best_result = result
            best_proposal = proposal

    if best_result < state["best_metric"]:
        state["best_metric"] = best_result
        state["status"] = "keep"
        state["lessons"].append(f"Iter {state['iteration']}: KEPT ({best_result:.4f}) — {best_proposal[:60] if best_proposal else ''}")
        print(f"  ✅ KEPT: {best_result:.4f}")
    else:
        state["status"] = "discard"
        state["lessons"].append(f"Iter {state['iteration']}: DISCARDED (best={state['best_metric']:.4f})")
        print(f"  ❌ DISCARDED (best={state['best_metric']:.4f})")
        if state["result_a"] is not None:
            state["lessons"].append(f"  FAILED: A={state['result_a']:.4f} — {state.get('proposal_a','')[:60]}")
        if state["result_b"] is not None:
            state["lessons"].append(f"  FAILED: B={state['result_b']:.4f} — {state.get('proposal_b','')[:60]}")

    return state

def log_result(state: ScientistState) -> ScientistState:
    print(f"\n  Log")
    loop_dir = Path(state["loop_dir"])
    results_file = loop_dir / "results.tsv"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    desc = (state.get("proposal_a") or "")[:60]
    results_file.write_text(results_file.read_text() + f"{state['iteration']}\t{commit}\t{state['best_metric']}\t-\t-\t-\t0.5\t{state['status']}\t{desc}\n")

    # Save lessons
    with open(loop_dir / "lessons.json", "w") as f:
        json.dump(state["lessons"][-50:], f, indent=2)

    return state

def should_continue(state: ScientistState) -> str:
    if state["iteration"] >= state["max_iter"]:
        return "end"
    return "continue"

# ── Build graph ──

def build_scientist_graph():
    builder = StateGraph(ScientistState)

    builder.add_node("research", research)
    builder.add_node("ideate", ideate)
    builder.add_node("run_a", run_proposal_a)
    builder.add_node("run_b", run_proposal_b)
    builder.add_node("decide", decide)
    builder.add_node("log_result", log_result)

    builder.set_entry_point("research")
    builder.add_edge("research", "ideate")
    builder.add_conditional_edges("ideate", lambda s: "run_a" if s.get("proposal_a") else "run_b")
    builder.add_edge("ideate", "run_a")
    builder.add_edge("ideate", "run_b")
    builder.add_edge("run_a", "decide")
    builder.add_edge("run_b", "decide")
    builder.add_edge("decide", "log_result")
    builder.add_conditional_edges("log_result", should_continue, {"continue": "research", "end": END})

    return builder.compile(checkpointer=MemorySaver())

# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="flu", choices=list(TASKS.keys()))
    parser.add_argument("--iterations", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--model", default="transformers", choices=["transformers", "ollama"])
    args = parser.parse_args()

    max_iter = args.iterations if args.iterations > 0 else float("inf")
    loop_dir = f"experiments/scientist-{args.task}-{datetime.now().strftime('%y%m%d-%H%M')}"
    Path(loop_dir).mkdir(parents=True, exist_ok=True)

    global orch_model, orch_tok
    if args.model == "transformers":
        print("[BOOT] Loading Qwen2.5-7B-Instruct...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        orch_model = AutoModelForCausalLM.from_pretrained(
            str(ROOT / "models/Qwen2.5-7B-Instruct"), torch_dtype=torch.bfloat16, device_map=device)
        orch_tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen2.5-7B-Instruct"))
        orch_model.eval()
    else:
        print("[BOOT] Using Ollama (qwen2.5:7b)...")
        try:
            import ollama
            ollama.pull("qwen2.5:7b")
            print("  Model pulled")
        except:
            print("  Using existing model")

    print(f"\n{'='*60}")
    print(f"  AI Scientist — {args.task}")
    print(f"  Engine: LangGraph + {args.model}")
    print(f"  Log: {loop_dir}")
    print(f"{'='*60}\n")

    # Init results
    results_file = Path(loop_dir) / "results.tsv"
    results_file.write_text("iteration\tcommit\tmetric\tsecondary\tval\tid_acc\tmem\tstatus\tdescription\n")

    graph = build_scientist_graph()
    config = {"configurable": {"thread_id": loop_dir}}

    state = ScientistState(
        task=args.task,
        iteration=0,
        best_metric=float("inf"),
        best_desc="baseline",
        current_model="gru",
        history=[],
        lessons=[],
        run_log_tail="",
        active_model=args.model,
        loop_dir=loop_dir,
        max_iter=max_iter,
        proposal_a=None,
        proposal_b=None,
        result_a=None,
        result_b=None,
        final_metric=0.0,
        status="running",
    )

    for s in graph.stream(state, config):
        for node, val in s.items():
            if val and val.get("iteration"):
                state = val

    print(f"\n{'='*60}")
    print(f"  Done: {state.get('iteration',0)} iterations")
    print(f"  Best: {state.get('best_metric',0):.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
