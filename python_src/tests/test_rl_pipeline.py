import csv
import os
import sys
import tempfile
import unittest

PYTHON_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PYTHON_SRC)
for path in (PYTHON_SRC, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RLPipelineTests(unittest.TestCase):
    def test_vehicle_mapping_supports_all_benchmark_sizes(self):
        from rl.data import infer_vehicle_config

        self.assertEqual(infer_vehicle_config(100), (10, 200.0))
        self.assertEqual(infer_vehicle_config(200), (50, 400.0))
        self.assertEqual(infer_vehicle_config(400), (100, 800.0))

    def test_generated_dataset_roundtrip_keeps_sdvrptw_fields(self):
        from rl.data import (
            NODE_FEATURES,
            generate_from_reference,
            load_dataset,
            problems_from_dataset,
            save_dataset,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "sample_1.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "x", "y", "demand", "open", "close", "servicetime",
                    "drone_serve", "time", "profit", "type",
                ])
                writer.writerow([0, 0, 0, 0, 100, 0, 1, 0, 0, 0])
                writer.writerow([1, 2, 3, 10, 30, 4, 0, 5, 7, 1])
                writer.writerow([4, 5, 6, 20, 50, 4, 1, 0, 11, 0])

            dataset = generate_from_reference(
                temp_dir, sample_count=3, seed=9, jitter=0.0,
                num_trucks=1, truck_capacity=20,
            )
            output = os.path.join(temp_dir, "train.pt")
            save_dataset(dataset, output)
            loaded = load_dataset(output)
            problems = problems_from_dataset(loaded)

            self.assertEqual(tuple(loaded["nodes"].shape), (3, 3, 10))
            self.assertEqual(loaded["node_features"], list(NODE_FEATURES))
            self.assertEqual(len(problems), 3)
            self.assertEqual(len(problems[0].requests), 2)
            self.assertTrue(all(req.profit in (7.0, 11.0)
                                for req in problems[0].requests))
            self.assertTrue(all(req.type in (0, 1)
                                for req in problems[0].requests))

    def test_exact_single_scenario_dataset_is_not_bootstrapped(self):
        from rl.data import dataset_from_scenario, problems_from_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "sample_1.csv")
            rows = [
                ["x", "y", "demand", "open", "close", "servicetime",
                 "drone_serve", "time", "profit", "type"],
                [0, 0, 0, 0, 100, 0, 1, 0, 0, 0],
                [1, 2, 3, 10, 30, 4, 0, 5, 7, 1],
                [4, 5, 6, 20, 50, 4, 1, 0, 11, 0],
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)

            dataset = dataset_from_scenario(
                csv_path, num_trucks=1, truck_capacity=20,
            )
            problems = problems_from_dataset(dataset)

            self.assertEqual(tuple(dataset["nodes"].shape), (1, 3, 10))
            self.assertEqual(
                dataset["generation"]["method"], "exact_single_scenario"
            )
            self.assertEqual(dataset["generation"]["reference_scenarios"], 1)
            self.assertEqual(problems[0].requests[0].profit, 7.0)
            self.assertEqual(problems[0].requests[1].profit, 11.0)

    def test_global_sampling_budget_counts_physical_rollouts(self):
        from rl.ppo import PPOConfig, PPOTrainer
        from sim.problem import Problem, Request

        depot = Request(0, 0, 0, 0, 0, 20, 0, 0, 0, 0)
        problems = [
            Problem(
                depot,
                [Request(1, float(i + 1), 0, 1, 0, 10, 0, 0, 2, 0)],
                1.0,
                10.0,
                1,
            )
            for i in range(3)
        ]

        def objective(problem, result):
            distance, profit = result
            return 0.5 * distance / 20.0 + 0.5 * (2.0 - profit) / 2.0

        trainer = PPOTrainer(
            problems,
            [1.0, 1.0, 1.0],
            objective,
            max_evaluations=5,
            config=PPOConfig(
                hidden_dim=8,
                evaluations_per_update=3,
                ppo_epochs=1,
                minibatch_size=32,
            ),
            seed=3,
            device="cpu",
            problems_per_evaluation=1,
            restore_best=False,
        )
        trainer.train(progress=False)
        self.assertEqual(trainer.evaluation_count, 5)
        self.assertEqual(trainer.simulator_rollouts, 5)


if __name__ == "__main__":
    unittest.main()
