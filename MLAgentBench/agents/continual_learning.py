""" Continual Learning Manager for MLSS26_HACKATHON.

Manages model updates across training iterations to prevent catastrophic forgetting.
Uses Elastic Weight Consolidation (EWC) and experience replay.
"""
import os
import json
import copy
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime


class ContinualLearningManager:
    """Manages model checkpoints across iterations with anti-forgetting mechanisms.

    Features:
    - Checkpoint versioning with score tracking
    - Elastic Weight Consolidation (EWC) penalty
    - Experience replay buffer
    - Automatic commit/rollback based on improvement vs forgetting thresholds
    """

    def __init__(
        self,
        checkpoint_dir="checkpoints",
        improvement_threshold=0.01,
        forgetting_threshold=0.05,
        ewc_lambda=100.0,
        replay_buffer_size=1000,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.improvement_threshold = improvement_threshold
        self.forgetting_threshold = forgetting_threshold
        self.ewc_lambda = ewc_lambda
        self.replay_buffer_size = replay_buffer_size

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.registry_path = os.path.join(checkpoint_dir, "model_registry.json")
        self.registry = self._load_registry()
        self.fisher_information = None
        self.optimal_params = None
        self.replay_buffer = []

    def _load_registry(self):
        """Load the model registry from disk."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"versions": [], "best_version": None, "best_score": -float("inf")}

    def _save_registry(self):
        """Save the model registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def load_best_model(self, model, device="cuda"):
        """Load the best previous model checkpoint into the given model."""
        best_version = self.registry.get("best_version")
        if best_version is None:
            return False
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_v{best_version}.pth")
        if not os.path.exists(checkpoint_path):
            return False
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        return True

    def compute_fisher_information(self, model, dataloader, device="cuda", max_batches=50):
        """Compute Fisher Information Matrix for EWC penalty.

        The Fisher matrix estimates which parameters are important for previous tasks.
        Parameters with high Fisher values should change less to avoid forgetting.
        """
        model.eval()
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)

        batch_count = 0
        for inputs, targets in dataloader:
            if batch_count >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            if outputs.dim() > 1:
                labels = outputs.argmax(dim=1)
            else:
                labels = targets
            loss = nn.functional.cross_entropy(outputs, labels) if outputs.dim() > 1 else nn.functional.mse_loss(outputs, targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            batch_count += 1

        for name in fisher:
            fisher[name] /= max(batch_count, 1)

        self.fisher_information = fisher
        self.optimal_params = {name: param.data.clone() for name, param in model.named_parameters()}
        model.train()
        return fisher

    def ewc_penalty(self, model):
        """Compute the EWC penalty loss.

        L_ewc = sum_i (lambda/2) * F_i * (theta_i - theta*_i)^2
        """
        if self.fisher_information is None or self.optimal_params is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        penalty = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_information:
                penalty += (self.ewc_lambda / 2) * (self.fisher_information[name] * (param - self.optimal_params[name]) ** 2).sum()
        return penalty

    def add_to_replay_buffer(self, samples, scores=None):
        """Add samples to the experience replay buffer."""
        for sample in samples:
            if len(self.replay_buffer) >= self.replay_buffer_size:
                self.replay_buffer.pop(0)
            self.replay_buffer.append(sample)

    def get_replay_samples(self, n=None):
        """Get n random samples from the replay buffer."""
        if n is None:
            n = min(len(self.replay_buffer), self.replay_buffer_size // 4)
        if len(self.replay_buffer) == 0:
            return []
        indices = np.random.choice(len(self.replay_buffer), min(n, len(self.replay_buffer)), replace=False)
        return [self.replay_buffer[i] for i in indices]

    def evaluate_model(self, model, dataloader, metric_fn, device="cuda"):
        """Evaluate model performance on a dataloader."""
        model.eval()
        total_score = 0.0
        count = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                score = metric_fn(outputs, targets)
                total_score += score.item() if hasattr(score, "item") else score
                count += 1
        model.train()
        return total_score / max(count, 1)

    def should_commit(self, new_score, old_score, forgetting_measure=0.0):
        """Decide whether to commit the new model version.

        Commit if:
        - Improvement >= improvement_threshold
        - Forgetting < forgetting_threshold
        """
        improvement = new_score - old_score
        return improvement >= self.improvement_threshold and forgetting_measure < self.forgetting_threshold

    def commit_version(self, model, score, metadata=None):
        """Save a new model version to the registry."""
        version = len(self.registry["versions"]) + 1
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_v{version}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        version_info = {
            "version": version,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "checkpoint": checkpoint_path,
            "metadata": metadata or {},
        }
        self.registry["versions"].append(version_info)

        if score > self.registry["best_score"]:
            self.registry["best_score"] = score
            self.registry["best_version"] = version

        self._save_registry()
        return version

    def rollback(self, model, device="cuda"):
        """Rollback to the best previous version."""
        return self.load_best_model(model, device)

    def get_status(self):
        """Get the current status of the continual learning manager."""
        return {
            "total_versions": len(self.registry["versions"]),
            "best_version": self.registry["best_version"],
            "best_score": self.registry["best_score"],
            "replay_buffer_size": len(self.replay_buffer),
            "ewc_enabled": self.fisher_information is not None,
        }
