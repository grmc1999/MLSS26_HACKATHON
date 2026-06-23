#!/usr/bin/env python3
"""Autonomous orchestrator — runs the full pipeline loop forever using local LLMs.

Usage:
    python scripts/run_orchestrator.py --task flu
    python scripts/run_orchestrator.py --task medmnist --iterations 50
    CUDA_VISIBLE_DEVICES=1 python scripts/run_orchestrator.py --task flu --headless &
"""
import argparse, os, re, subprocess, sys, time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from MLAgentBench.agents.agent_specialized import search_medical_literature, search_flu_context_rag, load_expert

TASKS = {
    "medmnist": {
        "runner": "python scripts/run_medmnist.py --epochs 25",
        "train_py": "MLAgentBench/benchmarks/medmnist/env/train.py",
        "metric_pattern": r"OOD F1:\s*([\d.]+)",
        "secondary_pattern": r"Test ID Acc:\s*([\d.]+)",
        "direction": "higher",
    },
    "flu": {
        "runner": "python scripts/run_exp.py --epochs 30 --lr 0.001 --hidden-dim 64 --model gru --num-layers 3",
        "train_py": "env/train.py",
        "metric_pattern": r"Test MAE:\s*([\d.]+)",
        "secondary_pattern": r"Val MAE:\s*([\d.]+)",
        "direction": "lower",
    },
}

def ask_llm(model, tokenizer, prompt, max_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="flu", choices=list(TASKS.keys()))
    parser.add_argument("--iterations", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    cfg = TASKS[args.task]
    max_iter = args.iterations if args.iterations > 0 else float("inf")
    loop_id = f"loop-auto-{args.task}-{datetime.now().strftime('%y%m%d-%H%M')}"
    log_dir = ROOT / "experiments" / loop_id
    log_dir.mkdir(parents=True, exist_ok=True)
    results_file = log_dir / "results.tsv"

    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    print(f"[BOOT] Loading Qwen2.5-7B-Instruct orchestrator on {device}...")
    orch_model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / "models/Qwen2.5-7B-Instruct"), torch_dtype=torch.bfloat16, device_map=device)
    orch_tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen2.5-7B-Instruct"))
    orch_model.eval()
    print(f"[BOOT] Orchestrator ready. Log: {log_dir}")

    results_file.write_text("iteration\tcommit\tmetric\tsecondary\tstatus\tdescription\n")
    iteration = 0
    best_metric = float("inf") if cfg["direction"] == "lower" else 0.0
    best_desc = ""

    while iteration < max_iter:
        iteration += 1
        print(f"\n{'='*50}\n  Iteration {iteration}\n{'='*50}")

        if iteration % 5 == 1:
            query = f"improve {cfg['direction']} {args.task} forecasting"
            try:
                if args.task == "flu":
                    # Hybrid: vector hits + FalkorDB relational context (model x
                    # country x method x metric). Degrades to vector-only if
                    # FalkorDB is unreachable -- never raises.
                    rag_ctx = search_flu_context_rag(query, k=3)["combined_context"]
                else:
                    rag = search_medical_literature(query, k=3, task=args.task)
                    rag_ctx = "\n".join(f"- {r.get('title','?')}" for r in rag)
            except: rag_ctx = ""
        else:
            rag_ctx = ""

        ctx = f"""Task: {args.task}. Current best: {best_metric:.4f} ({best_desc or 'baseline'}).
RAG: {rag_ctx}
Propose ONE specific change to {cfg['train_py']} to improve the metric.
Respond with a concise plan: what change, expected delta, risk."""

        print("[ORCH] Proposing...")
        proposal = ask_llm(orch_model, orch_tokenizer, ctx)
        print(f"  Proposal: {proposal[:200]}...")

        runner = cfg["runner"]
        print(f"[RUN] {runner}")
        result = subprocess.run(f"cd {ROOT} && {runner} > run.log 2>&1", shell=True, timeout=600)
        log_text = Path("run.log").read_text() if Path("run.log").exists() else ""

        primary = re.search(cfg["metric_pattern"], log_text)
        secondary = re.search(cfg["secondary_pattern"], log_text)
        p_val = float(primary.group(1)) if primary else None
        s_val = float(secondary.group(1)) if secondary else None

        if p_val is None:
            print(f"  CRASH — no metric found")
            results_file.write_text(results_file.read_text() + f"{iteration}\t-\t-\t-\tcrash\n")
            continue

        if cfg["direction"] == "lower":
            improved = p_val < best_metric
        else:
            improved = p_val > best_metric

        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

        if improved:
            best_metric = p_val
            best_desc = proposal[:80]
            status = "keep"
            print(f"  ✅ KEPT: {p_val:.4f} (best={best_metric:.4f})")
        else:
            status = "discard"
            print(f"  ❌ DISCARDED: {p_val:.4f} (best={best_metric:.4f})")

        results_file.write_text(results_file.read_text() + f"{iteration}\t{commit}\t{p_val}\t{s_val or ''}\t{status}\t{proposal[:80]}\n")

        if iteration >= max_iter:
            break

    print(f"\n{'='*50}\n  Done — {iteration} iterations, best={best_metric:.4f}\n{'='*50}")

if __name__ == "__main__":
    main()
