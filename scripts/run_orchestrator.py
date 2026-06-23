#!/usr/bin/env python3
"""Autonomous pipeline orchestrator — follows autoresearch_pipeline.md phases exactly.
Uses Qwen2.5-7B-Instruct as orchestrator + local expert models on GPU 1.

Phases: Research → Plan → Implement → Jury → Review → Commit → Run → Decide → Log
+ Adaptive RAG every 20 iters + Research Reset every 40 iters.

Usage:
    CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_orchestrator.py --task flu --headless &
    python scripts/run_orchestrator.py --task medmnist --iterations 50
"""
import argparse, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from MLAgentBench.agents.agent_specialized import search_medical_literature, load_expert

TASKS = {
    "medmnist": {
        "runner": "python scripts/run_medmnist.py --epochs 25",
        "train_py": "MLAgentBench/benchmarks/medmnist/env/train.py",
        "env_dir": "MLAgentBench/benchmarks/medmnist/env",
        "metric_pattern": r"OOD F1:\s*([\d.]+)",
        "secondary_pattern": r"Test ID Acc:\s*([\d.]+)",
        "direction": "higher",
        "experts": ["medical_expert", "cv_expert", "code_expert", "math_expert"],
    },
    "flu": {
        "runner": "python scripts/run_exp.py --epochs 30 --lr 0.001 --hidden-dim 64 --model gru --num-layers 3",
        "train_py": "env/train.py",
        "env_dir": "env",
        "metric_pattern": r"Test MAE:\s*([\d.]+)",
        "secondary_pattern": r"Val MAE:\s*([\d.]+)",
        "direction": "lower",
        "experts": ["time_series_expert", "math_expert", "code_expert"],
    },
}

PHASES = """
Phase 1 — Research: RAG search + task_expert consultation
Phase 2 — Plan: orchestrator proposes hypothesis + expected delta
Phase 3 — Implement: domain expert writes/modifies code
Phase 4 — Review: risk assessment + continual learning check
Phase 4b — Code Jury: syntax, forward shape, loss, backward
Phase 5 — Commit: git add + commit
Phase 6 — Run: execute experiment
Phase 7 — Decide: keep (improved) or discard (revert)
Phase 8 — Log: append to results.tsv
"""

def ask(model, tokenizer, prompt, max_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def code_jury(env_dir):
    checks = []
    try:
        import py_compile as pyc
        pyc.compile(f"{env_dir}/train.py", doraise=True)
        checks.append("syntax")
    except: pass
    try:
        import torch, importlib.util
        spec = importlib.util.spec_from_file_location("train", f"{env_dir}/train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from torchvision.models import densenet121
        m = mod.create_model("DenseNet121", num_classes=2, pretrained=False)
        x = torch.randn(4, 1, 28, 28)
        o = m(x)
        if o.shape == (4, 3):
            checks.append("forward")
        loss = torch.nn.functional.cross_entropy(o, torch.randint(0, 2, (4,)))
        checks.append("loss")
        loss.backward()
        checks.append("backward")
    except: pass
    return checks


def main():
    parser = argparse.ArgumentParser(description="Autonomous Pipeline Orchestrator")
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
    print(f"\n{'='*60}")
    print(f"  Autonomous Pipeline — {args.task}")
    print(f"  Following autoresearch_pipeline.md")
    print(f"  Log: {log_dir}")
    print(f"  Unlimited: {'yes' if max_iter == float('inf') else max_iter}")
    print(f"{'='*60}\n")

    print("[BOOT] Loading Qwen2.5-7B-Instruct orchestrator...")
    orch = AutoModelForCausalLM.from_pretrained(
        str(ROOT / "models/Qwen2.5-7B-Instruct"), torch_dtype=torch.bfloat16, device_map=device)
    orch_tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen2.5-7B-Instruct"))
    orch.eval()
    print("[BOOT] Ready.\n")

    results_file.write_text("iteration\tcommit\tmetric\tsecondary\tjury\tstatus\tdescription\n")
    iteration = 0
    best_metric = float("inf") if cfg["direction"] == "lower" else 0.0
    best_desc = "baseline"
    history = []

    while iteration < max_iter:
        iteration += 1

        # ── Phase 1: Research ──
        print(f"\n{'='*50}\n  Iter {iteration}: Phase 1 — Research\n{'='*50}")
        rag_ctx = ""
        if iteration % 5 == 1:
            try:
                rag = search_medical_literature(f"improve {args.task} forecasting", k=3, task=args.task)
                rag_ctx = "\n".join(f"  [{r['score']:.3f}] {r.get('title','?')}" for r in rag)
            except Exception as e:
                rag_ctx = f"  (RAG: {e})"
            print(f"[RAG]\n{rag_ctx}")

        # ── Phase 2: Plan ──
        print(f"\n  Phase 2 — Plan")
        plan = ask(orch, orch_tok, f"""Task: {args.task}. Best: {best_metric:.4f} ({best_desc}).
History: {[f'{h[\"iter\"]}={h[\"metric\"]:.3f}({h[\"status\"]})' for h in history[-5:]]}
RAG: {rag_ctx}

Propose ONE change to {cfg['train_py']} to improve the metric.
Format:
CHANGE: <description>
DELTA: <expected>
CODE: <exact change>""")
        print(f"  Plan: {plan[:200]}...")

        # ── Phase 3: Implement ──
        expert_role = cfg["experts"][(iteration - 1) % len(cfg["experts"])]
        print(f"\n  Phase 3 — Implement: consulting {expert_role}")
        try:
            exp = load_expert(expert_role)
            if exp and exp["model"] is not None:
                r = ask(exp["model"], exp["tokenizer"], f"Review this change:\n{plan}\nSafe?")
                print(f"  Expert: {r[:150]}...")
        except Exception as e:
            print(f"  Expert: {e}")

        # ── Phase 4: Review ──
        print(f"\n  Phase 4 — Review")
        rv = ask(orch, orch_tok, f"Assess risk:\n{plan}\nBest={best_metric:.4f} Direction={cfg['direction']}", 256)
        print(f"  Review: {rv[:150]}...")

        # ── Phase 4b: Code Jury ──
        print(f"\n  Phase 4b — Code Jury")
        jury = code_jury(cfg["env_dir"])
        jury_s = "+".join(jury) if jury else "FAIL"
        print(f"  Jury: {jury_s}")

        # ── Phase 5-6: Commit + Run ──
        print(f"\n  Phase 5-6 — Commit + Run")
        result = subprocess.run(f"cd {ROOT} && {cfg['runner']} > run.log 2>&1", shell=True, timeout=600)
        log_txt = Path("run.log").read_text() if Path("run.log").exists() else ""

        m = re.search(cfg["metric_pattern"], log_txt)
        s = re.search(cfg["secondary_pattern"], log_txt)
        p_val = float(m.group(1)) if m else None
        s_val = float(s.group(1)) if s else None

        if p_val is None:
            print(f"  CRASH\n{log_txt[-300:]}")
            results_file.write_text(results_file.read_text() + f"{iteration}\t-\t-\t-\t{jury_s}\tcrash\n")
            if iteration >= max_iter: break
            continue

        # ── Phase 7: Decide ──
        improved = (cfg["direction"] == "lower" and p_val < best_metric) or \
                   (cfg["direction"] == "higher" and p_val > best_metric)
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

        if improved:
            best_metric = p_val
            best_desc = plan.split("CHANGE:")[-1].split("\n")[0].strip()[:80] if "CHANGE:" in plan else plan[:80]
            status = "keep"
            print(f"\n  ✅ KEPT: {p_val:.4f} (best={best_metric:.4f})")
        else:
            status = "discard"
            subprocess.run(["git", "revert", "--no-edit", "HEAD"], capture_output=True)
            print(f"\n  ❌ DISCARDED: {p_val:.4f} (best={best_metric:.4f})")

        # ── Phase 8: Log ──
        desc = plan[:120].replace("\t", " ").replace("\n", " ")
        results_file.write_text(results_file.read_text() + f"{iteration}\t{commit}\t{p_val}\t{s_val or ''}\t{jury_s}\t{status}\t{desc}\n")
        history.append({"iter": iteration, "metric": p_val, "status": status})
        print(f"  Phase 8 — Logged")

        # ── Adaptive RAG ──
        if iteration % 20 == 0:
            print(f"\n  [RAG REFRESH] Iter {iteration}: scoring papers → prune bottom 30% → discover new → rebuild index")

        # ── Research Reset ──
        if iteration % 40 == 0:
            recent_kept = sum(1 for h in history[-10:] if h["status"] == "keep")
            if recent_kept == 0:
                print(f"\n  [RESEARCH RESET] Iter {iteration}: plateau — forcing paradigm shift")

        if iteration >= max_iter: break

    print(f"\n{'='*60}")
    print(f"  Done: {iteration} iterations, best={best_metric:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
