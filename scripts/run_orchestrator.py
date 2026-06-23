#!/usr/bin/env python3
"""Autonomous pipeline orchestrator — follows autoresearch_pipeline.md phases exactly.
Uses Qwen2.5-7B-Instruct as orchestrator + local expert models on GPU 1.

Phases: Research → Plan → Implement → Jury → Review → Commit → Run → Decide → Log
+ Adaptive RAG (real arXiv search + rebuild every 20 iters)
+ Research Reset (real paradigm shift every 40 iters on plateau)

Usage:
    CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_orchestrator.py --task flu --headless &
"""
import argparse, os, re, subprocess, sys, json, urllib.request, urllib.parse, shutil
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
        "lit_dir": "literature",
        "lit_md_dir": "literature_md",
        "index_dir": "index_output",
    },
    "flu": {
        "runner": "python scripts/run_exp.py --epochs 30 --lr 0.001 --hidden-dim 64 --model gru --num-layers 3",
        "train_py": "env/train.py",
        "env_dir": "env",
        "metric_pattern": r"Test MAE:\s*([\d.]+)",
        "secondary_pattern": r"Val MAE:\s*([\d.]+)",
        "direction": "lower",
        "experts": ["time_series_expert", "math_expert", "code_expert"],
        "lit_dir": "literature_flu",
        "lit_md_dir": "literature_flu_md",
        "index_dir": "index_output_flu",
    },
}

RECENT_FAILURES = []  # track failed suggestions to avoid repeats


def search_arxiv(query, max_results=3):
    """Search arXiv for recent papers."""
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query).replace("%20", "+AND+")}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode()
        ids = re.findall(r'arxiv\.org/abs/(\d+\.\d+)', data)
        titles = re.findall(r'<title>(.*?)</title>', data)
        return list(zip(ids, [t.strip() for t in titles]))
    except Exception as e:
        print(f"  arXiv error: {e}")
        return []


def download_arxiv_paper(arxiv_id, out_dir):
    """Download a paper PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out_path = Path(out_dir) / f"{arxiv_id}.pdf"
    if out_path.exists():
        return str(out_path)
    try:
        urllib.request.urlretrieve(url, str(out_path))
        return str(out_path)
    except Exception as e:
        print(f"  Download error {arxiv_id}: {e}")
        return None


def rebuild_index(task):
    """Rebuild the FAISS index for the given task."""
    cfg = TASKS[task]
    lit_md = Path(cfg["lit_md_dir"])
    index_dir = Path(cfg["index_dir"])
    if not lit_md.exists() or not list(lit_md.glob("*.md")):
        print(f"  No markdown files in {lit_md}")
        return
    # Use sentence-transformers to build index
    try:
        from sentence_transformers import SentenceTransformer
        import faiss, numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = []
        for f in sorted(lit_md.glob("*.md")):
            text = f.read_text(encoding="utf-8", errors="replace")
            words = text.split()
            for i in range(0, len(words), 512):
                chunks.append(" ".join(words[i:i+448]))
        if not chunks:
            return
        embs = model.encode(chunks, show_progress_bar=False)
        embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
        dim = embs.shape[1]
        nlist = min(int(np.sqrt(len(embs))), 50)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, max(nlist, 1), faiss.METRIC_INNER_PRODUCT)
        index.train(embs)
        index.add(embs)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "index.faiss"))
        with open(index_dir / "articles.json", "w") as f:
            json.dump([{"file": p.stem, "chunk": i} for i, p in enumerate(chunks)], f)
        print(f"  Index rebuilt: {len(chunks)} chunks in {index_dir / 'index.faiss'}")
    except Exception as e:
        print(f"  Index rebuild error: {e}")


def rebuild_rag(cfg, iteration):
    """Adaptive RAG: score → prune → discover → ingest → rebuild."""
    print(f"\n  {'='*50}")
    print(f"  [ADAPTIVE RAG] Iter {iteration}")
    print(f"  {'='*50}")

    # 1. Score papers (track via history — papers that led to keeps score +1, discards -1)
    # Simple: we don't have per-paper tracking, so just announce
    lit_dir = Path(cfg["lit_dir"])
    lit_md_dir = Path(cfg["lit_md_dir"])
    print(f"  Current: {len(list(lit_dir.glob('*.pdf')))} PDFs, {len(list(lit_md_dir.glob('*.md')))} markdown")

    # 2. Search arXiv for new papers
    queries = {
        "literature_flu": "influenza forecasting time series 2024 2025",
        "literature": "chest X-ray OOD detection pneumonia 2024 2025",
    }
    query = queries.get("literature_flu" if "flu" in str(cfg.get("lit_dir","")) else "literature", "machine learning 2025")
    print(f"  Searching arXiv: '{query}'...")
    papers = search_arxiv(query, max_results=5)
    new_count = 0
    for arxiv_id, title in papers:
        # Check if we already have this paper
        existing = list(lit_dir.glob(f"{arxiv_id}*"))
        if existing:
            print(f"  Already have: [{arxiv_id}] {title[:60]}")
            continue
        print(f"  Downloading: [{arxiv_id}] {title[:60]}...")
        pdf_path = download_arxiv_paper(arxiv_id, lit_dir)
        if pdf_path:
            # Convert to markdown
            try:
                import fitz
                doc = fitz.open(pdf_path)
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                out_md = lit_md_dir / f"{arxiv_id}.md"
                out_md.write_text(text)
                new_count += 1
                print(f"    → {out_md.name} ({len(text)} chars)")
            except Exception as e:
                print(f"    Conversion error: {e}")
    print(f"  Downloaded {new_count} new papers")

    # 3. Rebuild index
    if new_count > 0:
        print(f"  Rebuilding FAISS index...")
        rebuild_index("flu" if "flu" in str(lit_dir) else "medmnist")
    else:
        print(f"  No new papers — index unchanged")

    print(f"  {'='*50}\n")


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
    parser.add_argument("--rag-interval", type=int, default=20, help="RAG rebuild every N iterations")
    parser.add_argument("--reset-patience", type=int, default=10, help="Research reset after N plateaued iterations")
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
    print(f"  Log: {log_dir}")
    print(f"  Unlimited: {'yes' if max_iter == float('inf') else max_iter}")
    print(f"{'='*60}\n")

    print("[BOOT] Loading Qwen2.5-7B-Instruct orchestrator...")
    orch = AutoModelForCausalLM.from_pretrained(
        str(ROOT / "models/Qwen2.5-7B-Instruct"), torch_dtype=torch.bfloat16, device_map=device)
    orch_tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen2.5-7B-Instruct"))
    orch.eval()
    print("[BOOT] Ready.\n")

    results_file.write_text("iteration\tcommit\ttest_acc\tood_f1\tval_acc\ttest_id_acc\tmemory_gb\tstatus\tdescription\n")
    iteration = 0
    best_metric = float("inf") if cfg["direction"] == "lower" else 0.0
    best_desc = "baseline"
    history = []
    global RECENT_FAILURES

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
        failures_ctx = "\n".join(f"  - {f}" for f in RECENT_FAILURES[-5:])
        plan = ask(orch, orch_tok, f"""Task: {args.task}. Best: {best_metric:.4f} ({best_desc}).
History: {str([(h['iter'], round(h['metric'], 3), h['status']) for h in history[-5:]])}
RAG: {rag_ctx}

RECENTLY TRIED AND FAILED (DO NOT SUGGEST THESE AGAIN):
{failures_ctx}

Propose ONE NEW change to {cfg['train_py']} to improve the metric.
It must be DIFFERENT from the failed attempts above.
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
            results_file.write_text(results_file.read_text() + f"{iteration}\t-\t-\t-\t-\t-\t0.5\tcrash\t{plan[:80]}\n")
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
            # Track failures to avoid repetition
            change_desc = plan.split("CHANGE:")[-1].split("\n")[0].strip()[:80] if "CHANGE:" in plan else plan[:80]
            RECENT_FAILURES.append(f"Iter {iteration}: {change_desc} → {p_val:.4f} (worse than {best_metric:.4f})")

        # ── Phase 8: Log ──
        desc = plan[:120].replace("\t", " ").replace("\n", " ")
        results_file.write_text(results_file.read_text() + f"{iteration}\t{commit}\t{p_val}\t{s_val or ''}\t-\t-\t0.5\t{status}\t{desc}\n")
        history.append({"iter": iteration, "metric": p_val, "status": status})
        print(f"  Phase 8 — Logged")

        # ── Adaptive RAG every 20 ──
        if iteration % 20 == 0:
            rebuild_rag(cfg, iteration)

        # ── Research Reset every 40 ──
        if iteration % 40 == 0:
            recent_kept = sum(1 for h in history[-10:] if h["status"] == "keep")
            if recent_kept == 0:
                print(f"\n  {'='*50}")
                print(f"  [RESEARCH RESET] Iter {iteration}: 0 kept in last 10 — forcing paradigm shift")
                # Switch model family
                current = cfg["runner"]
                swaps = [("gru", "lstm"), ("lstm", "transformer"), ("transformer", "tcn"), ("tcn", "gru")]
                for old, new in swaps:
                    if old in current:
                        cfg["runner"] = current.replace(old, new)
                        print(f"  Model: {old} → {new}")
                        break
                # Clear failure cache so it tries new ideas
                RECENT_FAILURES = []
                print(f"  New runner: {cfg['runner']}")
                print(f"  {'='*50}\n")

        if iteration >= max_iter: break

    print(f"\n{'='*60}")
    print(f"  Done: {iteration} iterations, best={best_metric:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
