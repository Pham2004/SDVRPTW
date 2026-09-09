import os
import sys
import unittest

PYTHON_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PYTHON_SRC)
for path in (PYTHON_SRC, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from sim.problem import Problem, Request


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RLBudgetTests(unittest.TestCase):
    def test_shaped_return_has_same_ordering_as_repository_fitness(self):
        from rl.env import PolicySimulation

        class FirstCandidatePolicy:
            @staticmethod
            def select_action(features, deterministic=False):
                return 0, 0.0, 0.0

        depot = Request(0, 0, 0, 0, 0, 20, 0, 0, 0, 0)
        requests = [Request(1, 3, 4, 1, 0, 10, 0, 0, 4, 0)]
        p = Problem(depot, requests, 1.0, 10.0, 1)
        sim = PolicySimulation(p, FirstCandidatePolicy(), weight=0.5)
        distance, profit = sim.simulate_until(1.0, float("inf"))
        fit = 0.5 * distance / 20.0 + 0.5 * (4.0 - profit) / 4.0
        self.assertAlmostEqual(sum(step.reward for step in sim.steps), 0.5 - fit)

    def test_budget_is_strict_and_rollouts_are_accounted_separately(self):
        from rl.ppo import PPOConfig, PPOTrainer

        depot = Request(0, 0, 0, 0, 0, 20, 0, 0, 0, 0)
        requests = [
            Request(1, 1, 0, 1, 0, 10, 0, 0, 2, 0),
            Request(2, 0, 1, 1, 0, 10, 0, 0, 2, 0),
        ]
        p = Problem(depot, requests, 1.0, 10.0, 1)

        def fitness(prob, result):
            distance, profit = result
            return 0.5 * distance / 20.0 + 0.5 * (4.0 - profit) / 4.0

        config = PPOConfig(
            hidden_dim=16,
            evaluations_per_update=2,
            ppo_epochs=1,
            minibatch_size=32,
        )
        trainer = PPOTrainer(
            [p], [1.0], fitness, max_evaluations=3,
            config=config, seed=7, device="cpu",
        )
        trainer.train(progress=False)
        self.assertEqual(trainer.evaluation_count, 3)
        self.assertEqual(trainer.simulator_rollouts, 3)
        self.assertEqual([row["evaluations"] for row in trainer.history], [2, 3])


if __name__ == "__main__":
    unittest.main()
