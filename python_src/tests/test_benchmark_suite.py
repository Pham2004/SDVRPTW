import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PYTHON_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PYTHON_SRC)
for path in (PYTHON_SRC, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_benchmark_suite import aggregate, build_parser, _command


class BenchmarkSuiteTests(unittest.TestCase):
    def test_default_commands_use_paired_fair_protocol(self):
        args = build_parser().parse_args([])
        dataset = Path(REPO_ROOT) / "datasets" / "toy5"
        gp = _command(args, "gp", dataset, 42, Path("gp-out"))
        rl = _command(args, "rl", dataset, 42, Path("rl-out"))

        self.assertIn("--routing-depth", gp)
        self.assertEqual(gp[gp.index("--routing-depth") + 1], "8")
        self.assertEqual(gp[gp.index("--sequencing-depth") + 1], "6")
        self.assertIn("--exclude-training-scenarios", gp)
        self.assertIn("--exclude-training-scenario", rl)
        self.assertEqual(args.seeds, [42, 43, 44, 45, 46])

    def test_aggregate_writes_seed_statistics_and_paired_delta(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "h100_new"
            dataset.mkdir()
            for algorithm, values in (("gp", [0.2, 0.4]), ("rl", [0.5, 0.3])):
                for seed, fitness in zip((42, 43), values):
                    run_dir = root / "suite" / dataset.name / algorithm / f"seed_{seed}"
                    run_dir.mkdir(parents=True)
                    result = {
                        "target_path": str(dataset),
                        "overall": {
                            metric: fitness for metric in (
                                "avg_fitness", "macro_avg_fitness", "avg_distance",
                                "avg_profit", "avg_served", "avg_dropped",
                                "served_ratio", "dropped_ratio",
                            )
                        },
                        "instances": {
                            "c101": {
                                metric: fitness for metric in (
                                    "avg_fitness", "avg_distance", "avg_profit",
                                    "avg_served", "avg_dropped",
                                )
                            }
                        },
                    }
                    with open(run_dir / "results.json", "w", encoding="utf-8") as handle:
                        json.dump(result, handle)
                    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
                        json.dump({"status": "complete", "seed": seed}, handle)

            output = root / "suite"
            data = aggregate(output, [dataset], [42, 43], ["gp", "rl"])
            self.assertEqual(len(data["runs"]), 4)
            self.assertEqual(len(data["paired_comparison"]), 2)
            self.assertAlmostEqual(
                data["comparison_summary"][0]["rl_minus_gp_mean"], 0.1
            )
            with open(output / "summary_by_algorithm.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            gp = next(row for row in rows if row["algorithm"] == "gp")
            self.assertAlmostEqual(float(gp["avg_fitness_mean"]), 0.3)


if __name__ == "__main__":
    unittest.main()
