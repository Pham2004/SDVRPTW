"""A dependency-light masked PPO implementation for SDVRPTW."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from .env import FEATURE_NAMES, PolicySimulation, TrajectoryStep


@dataclass
class PPOConfig:
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    minibatch_size: int = 256
    evaluations_per_update: int = 32


class CandidatePolicy(nn.Module):
    """Scores a variable-size candidate set and estimates state value."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, 1)
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        logits = self.actor(encoded).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)

        weights = mask.unsqueeze(-1).to(encoded.dtype)
        mean_pool = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        max_pool = encoded.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        value = self.critic(torch.cat((mean_pool, max_pool), dim=-1)).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(self, config: PPOConfig, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = CandidatePolicy(len(FEATURE_NAMES), config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

    @staticmethod
    def _pad(feature_sets: Sequence[Sequence[Sequence[float]]], device):
        batch_size = len(feature_sets)
        max_candidates = max(len(rows) for rows in feature_sets)
        feature_dim = len(FEATURE_NAMES)
        features = torch.zeros(
            batch_size, max_candidates, feature_dim,
            dtype=torch.float32, device=device,
        )
        mask = torch.zeros(
            batch_size, max_candidates, dtype=torch.bool, device=device
        )
        for i, rows in enumerate(feature_sets):
            if not rows:
                raise ValueError("candidate set must not be empty")
            row_tensor = torch.as_tensor(rows, dtype=torch.float32, device=device)
            if row_tensor.ndim != 2 or row_tensor.shape[1] != feature_dim:
                raise ValueError(
                    f"expected candidate features (*, {feature_dim}), "
                    f"got {tuple(row_tensor.shape)}"
                )
            features[i, :len(rows)] = row_tensor
            mask[i, :len(rows)] = True
        return features, mask

    def select_action(self, features: Sequence[Sequence[float]],
                      deterministic: bool = False):
        x, mask = self._pad([features], self.device)
        with torch.no_grad():
            logits, value = self.model(x, mask)
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(self, feature_sets, actions):
        x, mask = self._pad(feature_sets, self.device)
        logits, values = self.model(x, mask)
        dist = Categorical(logits=logits)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        return dist.log_prob(actions_t), dist.entropy(), values


class PPOTrainer:
    """Trains with a strict objective-evaluation budget.

    One objective evaluation means evaluating one sampled policy on every
    problem in ``training_problems`` and averaging the fitness, matching GP's
    evaluation of one individual on its training ``ProblemSet``.  The raw
    number of simulator calls is tracked separately as ``simulator_rollouts``.
    """

    def __init__(
        self,
        training_problems: Sequence,
        training_time_slots: Sequence[float],
        fitness_fn: Callable,
        max_evaluations: int,
        config: Optional[PPOConfig] = None,
        weight: float = 0.5,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        if not training_problems:
            raise ValueError("training_problems must not be empty")
        if len(training_problems) != len(training_time_slots):
            raise ValueError("one time slot is required per training problem")
        if max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        self.training_problems = list(training_problems)
        self.training_time_slots = list(training_time_slots)
        self.fitness_fn = fitness_fn
        self.max_evaluations = int(max_evaluations)
        self.config = config or PPOConfig()
        self.weight = float(weight)
        self.seed = int(seed)

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.agent = PPOAgent(self.config, device=device)
        self.evaluation_count = 0
        self.simulator_rollouts = 0
        self.best_fitness = math.inf
        self.best_state_dict = copy.deepcopy(self.agent.model.state_dict())
        self.history: List[Dict[str, float]] = []

    def _rollout(self, problem, time_slot: float, deterministic: bool = False):
        sim = PolicySimulation(
            problem, self.agent, weight=self.weight,
            deterministic=deterministic,
        )
        distance, profit = sim.simulate_until(time_slot, float("inf"))
        fit = float(self.fitness_fn(problem, (distance, profit)))
        return sim.steps, distance, profit, fit, sim.get_routes(), sim.dropped_requests

    def _objective_evaluation(self):
        trajectories = []
        fits = []
        for problem, time_slot in zip(
                self.training_problems, self.training_time_slots):
            steps, _distance, _profit, fit, _routes, _dropped = self._rollout(
                problem, time_slot, deterministic=False
            )
            trajectories.append(steps)
            fits.append(fit)
            self.simulator_rollouts += 1
        self.evaluation_count += 1
        return trajectories, sum(fits) / len(fits)

    def _trajectory_records(self, trajectories):
        records = []
        gamma = self.config.gamma
        lam = self.config.gae_lambda
        for steps in trajectories:
            if not steps:
                continue
            advantage = 0.0
            returns = [0.0] * len(steps)
            advantages = [0.0] * len(steps)
            next_value = 0.0
            for i in range(len(steps) - 1, -1, -1):
                delta = steps[i].reward + gamma * next_value - steps[i].value
                advantage = delta + gamma * lam * advantage
                advantages[i] = advantage
                returns[i] = advantage + steps[i].value
                next_value = steps[i].value
            for step, adv, ret in zip(steps, advantages, returns):
                records.append((step, adv, ret))
        return records

    def _update(self, trajectories):
        records = self._trajectory_records(trajectories)
        if not records:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        adv_all = torch.tensor([r[1] for r in records], dtype=torch.float32)
        if len(adv_all) > 1:
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        metrics = []
        indices = list(range(len(records)))
        for _epoch in range(self.config.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                batch_indices = indices[start:start + self.config.minibatch_size]
                batch = [records[i] for i in batch_indices]
                steps = [item[0] for item in batch]
                old_logp = torch.tensor(
                    [step.log_prob for step in steps],
                    dtype=torch.float32, device=self.agent.device,
                )
                advantages = adv_all[batch_indices].to(self.agent.device)
                returns = torch.tensor(
                    [item[2] for item in batch],
                    dtype=torch.float32, device=self.agent.device,
                )
                new_logp, entropy, values = self.agent.evaluate_actions(
                    [step.features for step in steps],
                    [step.action for step in steps],
                )
                ratio = (new_logp - old_logp).exp()
                unclipped = ratio * advantages
                clipped = ratio.clamp(
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns)
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                )

                self.agent.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.model.parameters(), self.config.max_grad_norm
                )
                self.agent.optimizer.step()
                metrics.append((
                    float(policy_loss.detach().cpu()),
                    float(value_loss.detach().cpu()),
                    float(entropy_mean.detach().cpu()),
                ))

        count = max(len(metrics), 1)
        return {
            "policy_loss": sum(m[0] for m in metrics) / count,
            "value_loss": sum(m[1] for m in metrics) / count,
            "entropy": sum(m[2] for m in metrics) / count,
        }

    def train(self, progress: bool = True):
        update_idx = 0
        while self.evaluation_count < self.max_evaluations:
            update_idx += 1
            n_eval = min(
                self.config.evaluations_per_update,
                self.max_evaluations - self.evaluation_count,
            )
            update_trajectories = []
            update_fits = []
            for _ in range(n_eval):
                trajectories, avg_fit = self._objective_evaluation()
                update_trajectories.extend(trajectories)
                update_fits.append(avg_fit)
                if avg_fit < self.best_fitness:
                    self.best_fitness = avg_fit
                    self.best_state_dict = copy.deepcopy(
                        self.agent.model.state_dict()
                    )

            losses = self._update(update_trajectories)
            row = {
                "update": update_idx,
                "evaluations": self.evaluation_count,
                "simulator_rollouts": self.simulator_rollouts,
                "mean_training_fitness": sum(update_fits) / len(update_fits),
                "best_training_fitness": self.best_fitness,
                **losses,
            }
            self.history.append(row)
            if progress:
                print(
                    f"  update={update_idx:4d} "
                    f"evals={self.evaluation_count:6d}/{self.max_evaluations} "
                    f"fit={row['mean_training_fitness']:.5f} "
                    f"best={self.best_fitness:.5f}"
                )

        self.agent.model.load_state_dict(self.best_state_dict)
        return self.history

    def evaluate(self, problems: Sequence, time_slots: Sequence[float]):
        if len(problems) != len(time_slots):
            raise ValueError("one time slot is required per evaluation problem")
        results = []
        for problem, time_slot in zip(problems, time_slots):
            steps, distance, profit, fit, routes, dropped = self._rollout(
                problem, time_slot, deterministic=True
            )
            results.append({
                "distance": distance,
                "profit": profit,
                "fitness": fit,
                "served": sum(1 for route in routes for node in route if node != 0),
                "dropped": len(dropped),
                "routes": routes,
                "decisions": len(steps),
            })
        return results

    def checkpoint(self):
        return {
            "algorithm": "masked_ppo",
            "feature_names": list(FEATURE_NAMES),
            "config": asdict(self.config),
            "seed": self.seed,
            "max_evaluations": self.max_evaluations,
            "evaluation_count": self.evaluation_count,
            "simulator_rollouts": self.simulator_rollouts,
            "best_training_fitness": self.best_fitness,
            "model_state_dict": self.agent.model.state_dict(),
        }


__all__ = ["PPOAgent", "PPOConfig", "PPOTrainer"]
