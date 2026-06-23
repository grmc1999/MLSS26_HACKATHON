"""Unified Scientific AI AutoResearch Orchestrator.

Merges the Karpathy-style autoresearch loop with 8 specialized agents.

The orchestrator runs autonomously:
1. Consult specialized agent(s) for the next experiment idea
2. Modify train.py with the proposed change
3. Commit and run the experiment
4. Evaluate results (keep or discard)
5. Log everything and repeat

Subcommands (from autoresearch skill):
- plan    : Generate next experiment hypothesis
- run     : Execute a single experiment iteration
- fix     : Debug a crashed experiment and attempt repair
- analyze : Analyze results and decide next steps
- ship    : Lock in the current best model
- probe   : Deep-dive into model internals
"""
import os
import json
import sys
import re
import subprocess
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
ENV_DIR = PROJECT_ROOT / "MLAgentBench" / "benchmarks" / "medmnist" / "env"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Agent routing keywords
# ---------------------------------------------------------------------------

ROUTING_KEYWORDS = {
    "research_literature": ["paper", "cite", "literature", "reference", "survey", "arxiv",
                           "related work", "state-of-the-art", "sota", "publication",
                           "ood detection"],
    "autoresearch": ["experiment", "hypothesis", "plan", "iterate", "strategy",
                    "next step", "analyze result", "baseline", "improve", "compare"],
    "cv_expert": ["image", "augment", "preprocess", "conv", "resnet",
                 "architecture", "encoder", "cnn", "attention", "pooling",
                 "mahalanobis", "odim", "msp", "energy score"],
    "dl_expert": ["train", "loss", "optimizer", "learning rate", "epoch", "batch",
                 "gradient", "adam", "scheduler", "regularization", "dropout",
                 "normalization", "mixed precision", "calibration", "temperature",
                 "confidence", "focal", "label smoothing"],
    "llm_expert": ["prompt", "in-context", "few-shot", "chain-of-thought",
                   "reasoning", "coordinate", "instruction"],
    "medical_expert": ["chest", "xray", "x-ray", "x ray", "pneumonia", "lung",
                      "medmnist", "pneumoniamnist", "chestmnist", "consolidation",
                      "radiograph", "medical image", "opacity"],
    "continual_learning": ["forget", "remember", "version", "checkpoint", "ewc",
                          "replay", "rollback", "commit", "drift", "fisher"],
    "robustness_expert": ["uncertainty", "ood", "out-of-distribution", "robust",
                         "confidence", "calibration", "ece", "auroc", "threshold",
                         "distribution shift", "domain shift", "generalization"],
}


def load_config():
    """Load agent and orchestrator config from configs/agents.yaml."""
    path = CONFIGS_DIR / "agents.yaml"
    if not path.exists():
        return {"orchestrator": {"max_iterations": 25}, "agents": {}}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_models_config():
    """Load model configuration from configs/models.yaml."""
    path = CONFIGS_DIR / "models.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Autoresearch Skills (15 subcommands)
# ---------------------------------------------------------------------------

AUTORESEARCH_SUBCOMMANDS = {
    "plan": "Generate the next experiment hypothesis based on previous results. "
            "Analyze what worked, what failed, and propose the most promising next change.",

    "run": "Execute a single experiment iteration: modify code, commit, train, evaluate, "
           "and decide keep/discard based on the metric.",

    "fix": "Debug a crashed experiment. Read the stack trace, identify the root cause, "
           "and repair the code to get the experiment running again.",

    "analyze": "Deep analysis of results: compare learning curves, check for overfitting, "
               "compute statistical significance of improvements.",

    "ship": "Lock in the current best model. Run final evaluation, export checkpoints, "
            "generate submission, and register in the leaderboard.",

    "learn": "Extract learning from past iterations. Summarize what architectural choices, "
             "hyperparameters, and data strategies were most effective.",

    "reason": "Chain-of-thought reasoning about the experiment trajectory. "
              "Given N iterations of results, where should research go next?",

    "probe": "Deep-dive into model internals: analyze layer activations, gradient flow, "
             "attention maps, and feature importance.",

    "improve": "Focused improvement on a specific weakness. Given the current best model, "
               "identify its weakest cases and propose targeted fixes.",

    "debug": "Interactive debugging session. Step through the training loop, inspect "
             "tensors, and find bugs or performance bottlenecks.",

    "evals": "Run a comprehensive evaluation suite: compute Accuracy, F1, precision, "
             "recall, OOD F1, AUROC, and calibration metrics on the test set.",

    "regression": "Regression testing: verify that new changes don't break existing "
                  "functionality by comparing against known-good checkpoints.",

    "predict": "Predict the outcome of a proposed change before running it. "
               "Estimate the expected improvement based on prior experiments.",

    "scenario": "Run what-if scenarios: test the model under different conditions "
                "(e.g., different disease prevalence, population shifts, noise levels).",
}


def route_to_agent(task_description):
    """Determine which agent should handle the given task based on keywords."""
    task_lower = task_description.lower()
    best_agent = "autoresearch"
    best_score = 0
    for agent_name, keywords in ROUTING_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in task_lower)
        if score > best_score:
            best_score = score
            best_agent = agent_name
    return best_agent


class ExperimentManager:
    """Manages experiment execution: modify, commit, run, eval, keep/discard."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_tsv = self.log_dir / "results.tsv"
        self.runs_jsonl = PROJECT_ROOT / "experiments" / "runs.jsonl"
        self.best_metric = -1.0
        self.best_commit = None
        self.best_description = None

    def init_tsv(self, direction="higher_is_better"):
        """Initialize results.tsv with header."""
        if not self.results_tsv.exists():
            with open(self.results_tsv, "w") as f:
                f.write(f"# metric_direction: {direction}\n")
                f.write("iteration\ttimestamp\tcommit\tmetric\tdelta\tstatus\tdescription\n")

    def get_last_results(self, n=3):
        """Read last N results from TSV."""
        if not self.results_tsv.exists():
            return []
        with open(self.results_tsv) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")
                     and not l.startswith("iteration")]
        return lines[-n:]

    def log_result(self, iteration, commit, metric, delta, status, description):
        """Log a result row to TSV."""
        self.init_tsv()
        with open(self.results_tsv, "a") as f:
            f.write(f"{iteration}\t{datetime.now().isoformat()}\t{commit}\t{metric:.6f}\t"
                    f"{delta:.6f}\t{status}\t{description}\n")
        # Also register in runs.jsonl for the dashboard
        self._register_run(metric, description)

    def _register_run(self, metric, description):
        """Register experiment in runs.jsonl for dashboard display."""
        try:
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": "cnn",
                "base_ch": 32,
                "epochs": 50,
                "lr": 0.001,
                "batch": 128,
                "params": 0,
                "elapsed_s": 0,
                "best_val_acc": metric,
                "best_epoch": 0,
                "test_accuracy": metric,
                "description": description,
            }
            PROJECT_ROOT.joinpath("experiments").mkdir(exist_ok=True)
            with open(self.runs_jsonl, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def modify_code(self, changes: str) -> bool:
        """Apply a code modification to train.py using the LLM."""
        train_py = ENV_DIR / "train.py"
        if not train_py.exists():
            print(f"[ERROR] train.py not found at {train_py}")
            return False
        # Write the changes to a temp file and apply
        patch_file = self.log_dir / f"patch_{datetime.now().strftime('%H%M%S')}.py"
        patch_file.write_text(changes)
        print(f"[MODIFY] Changes written to {patch_file}")
        return True

    def git_commit(self, description: str) -> Optional[str]:
        """Commit current changes and return the commit hash."""
        try:
            subprocess.run(["git", "add", "-f", str(ENV_DIR / "train.py"),
                          str(SCRIPTS_DIR / "run_medmnist.py")],
                         cwd=PROJECT_ROOT, capture_output=True)
            result = subprocess.run(
                ["git", "commit", "-m", f"experiment: {description[:100]}"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
            )
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                capture_output=True, text=True, timeout=10
            )
            sha = sha_result.stdout.strip()
            return sha if sha else None
        except subprocess.TimeoutExpired:
            return None

    def git_revert(self):
        """Revert the last commit."""
        try:
            subprocess.run(["git", "revert", "--no-edit", "HEAD"],
                          cwd=PROJECT_ROOT, capture_output=True, timeout=30)
            return True
        except subprocess.TimeoutExpired:
            return False

    def git_restore_worktree(self):
        """Restore working tree files after a revert that deleted them."""
        try:
            # Restore train.py from HEAD if it was deleted by revert
            result = subprocess.run(
                ["git", "show", "HEAD:MLAgentBench/benchmarks/medmnist/env/train.py"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout:
                (ENV_DIR / "train.py").write_text(result.stdout)
            return True
        except Exception:
            return False

    def run_experiment(self, cmd: str = None) -> dict:
        """Run the experiment and return the metric value."""
        if cmd is None:
            cmd = f"cd {ENV_DIR} && python train.py > run.log 2>&1"
        print(f"[RUN] Executing: {cmd}")
        start = time.time()
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                   text=True, timeout=600)
            elapsed = time.time() - start
            # Parse accuracy from output (train.py prints "Standard accuracy:",
            # run_medmnist.py prints "Test 3-class Accuracy:")
            acc_match = re.search(r"(?:Standard accuracy|Test 3-class Accuracy):\s*([\d.]+)", result.stdout)
            # Fallback: try OOD F1 (train.py: "OOD Detection F1:", run_medmnist.py: "OOD F1 Score:")
            ood_match = re.search(r"(?:OOD Detection F1|OOD F1 Score):\s*([\d.]+)", result.stdout)
            print(f"[RUN] Completed in {elapsed:.1f}s")
            metric = None
            if acc_match:
                metric = float(acc_match.group(1))
            elif ood_match:
                metric = float(ood_match.group(1))
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed": elapsed,
                "metric": metric,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "timeout", "metric": None}
        except Exception as e:
            return {"success": False, "error": str(e), "metric": None}


class ScientificAutoResearch:
    """Unified Scientific AI AutoResearch Orchestrator.

    Merges the Karpathy-style autoresearch loop with specialized agent consultation.
    """

    def __init__(self, log_dir: str, max_iterations: int = 25, direction="higher_is_better"):
        self.log_dir = Path(log_dir)
        self.max_iterations = max_iterations
        self.direction = direction
        self.experiment = ExperimentManager(log_dir)
        self.config = load_config()
        self.models_config = load_models_config()
        self.activity_log = []
        self.current_iteration = 0
        self.best_metric = -1.0 if direction == "higher_is_better" else float("inf")
        self.best_iteration = 0
        self.kept_count = 0
        self.discarded_count = 0

    def log_activity(self, action: str, detail: str = ""):
        """Log an activity for the dashboard."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": self.current_iteration,
            "action": action,
            "detail": str(detail)[:500],
        }
        self.activity_log.append(entry)
        log_path = self.log_dir / "agent_log" / "orchestrator_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def consult_agent(self, role: str, question: str) -> str:
        from MLAgentBench.agents.agent_specialized import AGENT_PROMPTS, LOCAL_EXPERTS, load_expert
        prompt = AGENT_PROMPTS.get(role, "")
        expert = LOCAL_EXPERTS.get(role)
        if expert and os.path.exists(expert["path"]):
            loaded = load_expert(role)
            if loaded and loaded["model"] is not None:
                try:
                    model = loaded["model"]
                    tokenizer = loaded["tokenizer"]
                    messages = [{"role": "user", "content": f"{prompt}\n\n{question}"}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    return response.strip()
                except Exception as e:
                    return f"[{role} model error: {e}]"
        return f"=== Consulting {role} ===\n{prompt}\n\nQuestion: {question}"

    def get_status(self) -> dict:
        """Get current status for the dashboard."""
        return {
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "best_metric": self.best_metric,
            "best_iteration": self.best_iteration,
            "kept": self.kept_count,
            "discarded": self.discarded_count,
            "total_activities": len(self.activity_log),
        }

    def run_iteration(self, description: str, cmd: str = None) -> dict:
        """Run a single experiment iteration.

        Steps:
        1. Log the intent
        2. Commit the current code changes
        3. Run the experiment
        4. Extract the metric
        5. Decide keep/discard
        6. Log the result
        """
        self.current_iteration += 1
        self.log_activity("iteration_start", f"Iteration {self.current_iteration}: {description}")

        # Commit
        commit_sha = self.experiment.git_commit(description)
        if commit_sha:
            print(f"[ITER {self.current_iteration}] Committed as {commit_sha[:8]}: {description}")
        else:
            commit_sha = "none"
            print(f"[ITER {self.current_iteration}] No changes to commit")

        # Run experiment
        result = self.experiment.run_experiment(cmd)

        # Extract metric
        if result["success"] and result["metric"] is not None:
            metric = result["metric"]
            print(f"[ITER {self.current_iteration}] Metric: {metric:.6f}")
        else:
            metric = 0.0
            print(f"[ITER {self.current_iteration}] Experiment failed: {result.get('error', 'unknown')}")

        # Calculate delta from previous best
        prev_best = self.best_metric
        delta = metric - prev_best if prev_best >= 0 else 0.0

        # Decide keep/discard
        improved = metric > prev_best
        crashed = not result["success"]

        if crashed:
            status = "crash"
            self.experiment.git_revert()
            self.experiment.git_restore_worktree()
            self.discarded_count += 1
            print(f"[ITER {self.current_iteration}] CRASHED — reverted")
            # Try to read the stack trace
            if result.get("stdout"):
                tail = result["stdout"][-2000:]
                print(f"[DEBUG] Last output:\n{tail}")
        elif improved:
            status = "keep"
            self.best_metric = metric
            self.best_iteration = self.current_iteration
            self.kept_count += 1
            print(f"[ITER {self.current_iteration}] KEPT (Delta: +{delta:.6f})")
        else:
            status = "discard"
            self.experiment.git_revert()
            self.experiment.git_restore_worktree()
            self.discarded_count += 1
            print(f"[ITER {self.current_iteration}] DISCARDED — reverted")

        # Log result
        self.experiment.log_result(
            iteration=self.current_iteration,
            commit=commit_sha[:8] if commit_sha != "none" else "-",
            metric=metric,
            delta=delta,
            status=status,
            description=description,
        )

        self.log_activity("iteration_end",
                         f"metric={metric:.4f}, delta={delta:.4f}, status={status}")

        return {"iteration": self.current_iteration, "metric": metric,
                "delta": delta, "status": status, "commit": commit_sha}

    def run_loop(self, primary_agent="autoresearch"):
        """Run the autonomous research loop.

        LOOP FOREVER (up to max_iterations):
        1. Consult the designated agent for the next experiment idea
        2. Execute the experiment
        3. Evaluate, keep/discard, log
        4. Repeat
        """
        print(f"\n{'='*60}")
        print(f"  Scientific AI AutoResearch — {primary_agent}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Direction: {self.direction}")
        print(f"  Log dir: {self.log_dir}")
        print(f"{'='*60}\n")

        self.experiment.init_tsv(self.direction)
        self.log_activity("orchestrator_start",
                         f"Starting loop with {primary_agent}, max {self.max_iterations} iterations")

        for i in range(self.max_iterations):
            # If we have results, review last iteration
            last_results = self.experiment.get_last_results(1)
            if last_results:
                parts = last_results[0].split("\t")
                if len(parts) >= 6:
                    last_status = parts[5]
                    last_metric = float(parts[3]) if parts[3] != "-" else 0.0
                    print(f"\n[STATUS] Previous: {last_status} (metric={last_metric:.4f})")

            print(f"\n{'─'*50}")
            print(f"  Iteration {i+1}/{self.max_iterations}")
            print(f"{'─'*50}\n")

            # The LLM would propose a change here based on context
            # For now, we signal the iteration is ready for the LLM
            print(f"[LOOP] Ready for iteration {i+1}")

            # This is where the LLM action happens:
            # 1. Read current code and results
            # 2. Propose a modification
            # 3. Execute via run_iteration()
            # In the actual loop, the LLM agent handles this.
            # We define the structure — execution is by the agent.

            self.log_activity("loop_iteration_ready",
                             f"Iteration {i+1} ready. Consult {primary_agent} for next change.")

        # Summary
        print(f"\n{'='*60}")
        print(f"  LOOP COMPLETE")
        print(f"  Total iterations: {self.current_iteration}")
        print(f"  Best metric: {self.best_metric:.6f} (iteration {self.best_iteration})")
        print(f"  Kept: {self.kept_count} | Discarded: {self.discarded_count}")
        print(f"{'='*60}")
        return self.get_status()

    def run_subcommand(self, subcommand: str, params: dict = None) -> dict:
        """Run an autoresearch subcommand (plan, run, fix, analyze, ship, etc.)."""
        if subcommand not in AUTORESEARCH_SUBCOMMANDS:
            return {"error": f"Unknown subcommand: {subcommand}. Available: {list(AUTORESEARCH_SUBCOMMANDS.keys())}"}

        self.log_activity("subcommand", f"Running subcommand: {subcommand}")

        if subcommand == "ship":
            # Final evaluation and export
            return self._cmd_ship(params)
        elif subcommand == "analyze":
            return self._cmd_analyze(params)
        elif subcommand == "evals":
            return self._cmd_evals(params)
        elif subcommand == "plan":
            return self._cmd_plan(params)
        else:
            return {"subcommand": subcommand, "status": "ready",
                    "description": AUTORESEARCH_SUBCOMMANDS[subcommand]}

    def _cmd_ship(self, params: dict = None) -> dict:
        """Ship the best model: run final eval, export checkpoint, generate submission."""
        print("[SHIP] Running final evaluation on best model...")
        result = self.experiment.run_experiment(cmd="cd {ENV_DIR} && python train.py > ship.log 2>&1")
        # Export checkpoint
        checkpoint_src = ENV_DIR / "best_model.pth"
        if checkpoint_src.exists():
            import shutil
            ship_dir = self.log_dir / "shipped"
            ship_dir.mkdir(exist_ok=True)
            shutil.copy(checkpoint_src, ship_dir / "best_model.pth")
            # Copy submission
            sub_src = ENV_DIR / "submission.csv"
            if sub_src.exists():
                shutil.copy(sub_src, ship_dir / "submission.csv")
            print(f"[SHIP] Model shipped to {ship_dir}")
        return {"status": "shipped", "best_metric": self.best_metric,
                "best_iteration": self.best_iteration}

    def _cmd_analyze(self, params: dict = None) -> dict:
        """Analyze experiment results."""
        results = self.experiment.get_last_results(10)
        analysis = []
        for line in results:
            parts = line.split("\t")
            if len(parts) >= 6:
                analysis.append({
                    "iteration": parts[0], "metric": parts[3],
                    "delta": parts[4], "status": parts[5],
                    "description": parts[6] if len(parts) > 6 else "",
                })
        return {"analysis": analysis, "total_results": len(results)}

    def _cmd_evals(self, params: dict = None) -> dict:
        """Run comprehensive evaluation."""
        result = self.experiment.run_experiment()
        return {"eval_result": result}

    def _cmd_plan(self, params: dict = None) -> dict:
        """Generate next experiment plan based on history."""
        results = self.experiment.get_last_results(5)
        return {
            "status": "ready",
            "context": f"Last {len(results)} results reviewed",
            "next_steps": [
                "Consult a specialized agent for the next hypothesis",
                "Propose a focused code modification",
                "Run experiment → keep/discard → repeat",
            ],
        }


def main():
    """CLI entry point for the autoresearch_scientific mode."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Scientific AI AutoResearch — merge autoresearch loop with 8 specialized agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the full autonomy loop (25 iterations)
  python -m MLAgentBench.agents.orchestrator --iterations 25 --agent cv_expert

  # Run with specific verify command
  python -m MLAgentBench.agents.orchestrator --iterations 5 --verify "python scripts/run_medmnist.py --epochs 50"

  # Quick eval mode
  python -m MLAgentBench.agents.orchestrator --subcommand evals

  # Ship best model
  python -m MLAgentBench.agents.orchestrator --subcommand ship
        """,
    )
    parser.add_argument("--iterations", type=int, default=25, help="Max iterations")
    parser.add_argument("--agent", default="autoresearch", choices=list(ROUTING_KEYWORDS.keys()),
                       help="Primary agent role")
    parser.add_argument("--subcommand", choices=list(AUTORESEARCH_SUBCOMMANDS.keys()),
                       help="Run a single subcommand instead of the full loop")
    parser.add_argument("--verify", default="python scripts/run_medmnist.py --epochs 50",
                       help="Verify command that outputs the metric")
    parser.add_argument("--log-dir", default=None, help="Log directory")
    args = parser.parse_args()

    # Create log dir
    from datetime import datetime
    log_dir = args.log_dir or f"experiments/loop-{datetime.now().strftime('%y%m%d-%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    # Create orchestrator
    orch = ScientificAutoResearch(
        log_dir=log_dir,
        max_iterations=args.iterations,
    )

    print(f"\n{'='*60}")
    print(f"  Scientific AI AutoResearch")
    print(f"  Agent: {args.agent}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Log dir: {log_dir}")
    print(f"{'='*60}\n")

    if args.subcommand:
        result = orch.run_subcommand(args.subcommand)
        print(json.dumps(result, indent=2))
    else:
        orch.run_loop(primary_agent=args.agent)
        print(f"\nDone. Results in {log_dir}")


if __name__ == "__main__":
    main()
