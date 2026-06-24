"""Run mandatory literature/RAG search for a given iteration and save evidence to JSON.

Usage:
    python scripts/run_rag_search.py --task flu --iteration 4 \\
        --query "cross-country ILI forecasting domain adaptation" \\
        --k 5 --out experiments/loop-flu-YYMMDD-HHMM/iterations/iter-4-rag.json
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_utils import ensure_parent_dir, write_json, utc_now_iso


def import_rag_functions(rag_module: str, task: str):
    if rag_module:
        import importlib
        mod = importlib.import_module(rag_module)
        fn = getattr(mod, "search_flu_context_rag", None)
        if fn is None:
            raise ImportError(f"{rag_module} has no 'search_flu_context_rag'")
        return fn

    try:
        from MLAgentBench.agents.agent_specialized import search_flu_context_rag
    except ImportError as e:
        print(f"ERROR: Could not import RAG functions: {e}", file=sys.stderr)
        print("Pass --rag-module to specify an alternative module path.", file=sys.stderr)
        sys.exit(2)

    return search_flu_context_rag


def normalize_flu_hits(rag_output: dict) -> list[dict]:
    results = []
    for h in rag_output.get("vector_hits", []):
        if "error" in h:
            continue
        results.append({
            "title": h.get("title", h.get("file", "?")),
            "source": "local_rag_vector",
            "score": float(h.get("score", 0)),
            "snippet": "",
            "metadata": {},
        })
    graph = rag_output.get("graph_context", "")
    if graph.strip():
        results.append({
            "title": "knowledge_graph_context",
            "source": "falkordb",
            "score": 1.0,
            "snippet": graph.strip()[:500],
            "metadata": {},
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RAG literature search and save results to JSON")
    parser.add_argument("--task", required=True, choices=["flu"])
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rag-module", default=None, help="Alternative import path for RAG functions")
    parser.add_argument("--allow-empty", action="store_true", help="Exit 0 even if results are empty")
    args = parser.parse_args()

    rag_fn = import_rag_functions(args.rag_module, args.task)

    print(f"Running RAG search for iteration {args.iteration} (task={args.task}, k={args.k})")
    print(f"  Query: {args.query}")

    raw = rag_fn(args.query, k=args.k)

    results = normalize_flu_hits(raw)

    record = {
        "iteration": args.iteration,
        "task": args.task,
        "query": args.query,
        "k": args.k,
        "timestamp": utc_now_iso(),
        "results": results,
    }

    if not results:
        record["warning"] = "rag returned zero results"
        if not args.allow_empty:
            write_json(args.out, record)
            print(f"WARNING: RAG returned zero results. Write to {args.out}")
            sys.exit(1)

    ensure_parent_dir(args.out)
    write_json(args.out, record)
    print(f"  Found {len(results)} result(s)")
    for r in results:
        print(f"    - {r['title']} (score: {r['score']:.3f})")
    print(f"  Wrote: {args.out}")


if __name__ == "__main__":
    main()
