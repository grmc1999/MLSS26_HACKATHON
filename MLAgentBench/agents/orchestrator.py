""" Agent Orchestrator for MLSS26_HACKATHON.

Routes subproblems to specialized agents and coordinates iterations
with continual learning support.

Inspired by Karpathy's autoresearch autonomous experiment loop:
1. Modify code → train → evaluate → keep/discard → repeat
2. Never stop — run autonomously until interrupted
3. Log all results for review
"""
import os
import json
import sys
import time
import subprocess
import re
from datetime import datetime
from MLAgentBench.agents.agent_specialized import create_agent, load_orchestrator_config, AGENT_PROMPTS
from MLAgentBench.agents.continual_learning import ContinualLearningManager


# Keywords that trigger routing to specific agents
ROUTING_KEYWORDS = {
    "research_literature": ["paper", "cite", "literature", "reference", "survey", "arxiv", "related work", "state-of-the-art", "sota"],
    "autoresearch": ["experiment", "hypothesis", "plan", "iterate", "strategy", "next step", "analyze result", "baseline", "improve"],
    "cv_expert": ["image", "augment", "preprocess", "segment", "conv", "resnet", "unet", "u-net", "architecture", "encoder", "decoder", "deeplab", "segformer"],
    "dl_expert": ["train", "loss", "optimizer", "learning rate", "epoch", "batch", "gradient", "diffusion", "fine-tune", "lora", "adam", "scheduler"],
    "llm_expert": ["prompt", "in-context", "few-shot", "chain-of-thought", "reasoning", "coordinate"],
    "satellite_expert": ["satellite", "remote sensing", "spectral", "band", "goes", "abi", "infrared", "geospatial", "era5", "atmospheric", "contrail"],
    "continual_learning": ["forget", "remember", "version", "checkpoint", "ewc", "replay", "rollback", "commit", "drift", "fisher"],
    "physics_expert": ["physics", "advection", "continuity", "csi", "critical success", "physical", "residual", "constraint", "pde", "wind"],
}


class AgentOrchestrator:
    """Coordinates multiple specialized agents for the hackathon.

    The orchestrator:
    1. Receives a high-level task or subproblem
    2. Routes it to the most appropriate specialized agent
    3. Manages the continual learning loop across iterations
    4. Logs all agent activities for the dashboard
    5. Runs the Karpathy-style autoresearch experiment loop
    """

    def __init__(self, args, env, config=None):
        self.args = args
        self.env = env
        self.config = config or load_orchestrator_config()
        self.max_iterations = self.config.get("max_iterations", 50)
        self.cl_config = self.config.get("continual_learning", {})
        self.continual_learning = None
        if self.cl_config.get("enabled", False):
            self.continual_learning = ContinualLearningManager(
                checkpoint_dir=self.cl_config.get("checkpoint_dir", "checkpoints"),
                improvement_threshold=self.cl_config.get("improvement_threshold", 0.01),
                forgetting_threshold=self.cl_config.get("forgetting_threshold", 0.05),
                ewc_lambda=self.cl_config.get("ewc_lambda", 100.0),
                replay_buffer_size=self.cl_config.get("replay_buffer_size", 1000),
            )
        self.activity_log = []
        self.current_iteration = 0
        self.agent_scores = {}
        self.results_tsv = None

    def route_to_agent(self, task_description):
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

    def log_activity(self, agent_name, action, result, iteration=None):
        """Log an agent activity for the dashboard."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration or self.current_iteration,
            "agent": agent_name,
            "action": action,
            "result": str(result)[:500],
        }
        self.activity_log.append(entry)
        log_path = os.path.join(self.args.log_dir, "agent_log", "orchestrator_log.json")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def init_results_tsv(self):
        """Initialize the results.tsv file (Karpathy autoresearch format)."""
        self.results_tsv = os.path.join(self.args.log_dir, "results.tsv")
        os.makedirs(self.args.log_dir, exist_ok=True)
        if not os.path.exists(self.results_tsv):
            with open(self.results_tsv, "w") as f:
                f.write("iteration\tagent\tdice_score\tmemory_gb\tstatus\tdescription\n")

    def log_result(self, iteration, agent, dice_score, memory_gb, status, description):
        """Log a result to the TSV file."""
        if self.results_tsv is None:
            self.init_results_tsv()
        with open(self.results_tsv, "a") as f:
            f.write(f"{iteration}\t{agent}\t{dice_score:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")

    def run_iteration(self, agent_name="autoresearch"):
        """Run a single iteration with the specified agent."""
        self.current_iteration += 1
        self.log_activity(agent_name, "iteration_start", f"Autoresearch iteration {self.current_iteration}")

        agent = create_agent(agent_name, self.args, self.env)
        result = agent.run(self.env)

        self.log_activity(agent_name, "iteration_end", result)
        return result

    def run(self, primary_agent="autoresearch"):
        """Run the full Karpathy-style autoresearch orchestration loop.

        LOOP FOREVER (up to max_iterations):
        1. Run the agent for one iteration
        2. If continual learning enabled, manage checkpoints
        3. Log results
        4. Never stop until max_iterations or interrupted
        """
        self.init_results_tsv()
        self.log_activity("orchestrator", "start", f"Starting autoresearch loop with {primary_agent}")

        best_score = -float("inf")

        for i in range(self.max_iterations):
            result = self.run_iteration(primary_agent)

            if self.continual_learning:
                status = self.continual_learning.get_status()
                self.log_activity("orchestrator", "cl_status", status, iteration=self.current_iteration)
                if status.get("best_score", -float("inf")) > best_score:
                    best_score = status["best_score"]

            if "Finished" in str(result):
                self.log_activity("orchestrator", "complete", result, iteration=self.current_iteration)
                break

        self.log_activity("orchestrator", "end", f"Autoresearch complete. Best score: {best_score}")
        return self.activity_log

    def get_status(self):
        """Get current orchestrator status for the dashboard."""
        status = {
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "total_activities": len(self.activity_log),
            "agent_scores": self.agent_scores,
        }
        if self.continual_learning:
            status["continual_learning"] = self.continual_learning.get_status()
        return status
