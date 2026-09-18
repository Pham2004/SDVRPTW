#!/usr/bin/env python3
"""Stage 3: evaluate a trained checkpoint on repository CSV data."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for path in (HERE, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from main import discover_instances
from rl.data import fitness_value, infer_vehicle_config
from rl.env import PolicySimulation
from rl.ppo import PPOAgent
from sim.problem import ProblemSet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one masked-PPO checkpoint on CSV scenarios"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("target_path")
    parser.add_argument("num_instances", nargs="?", type=int, default=None)
    parser.add_argument("num_scenarios", nargs="?", type=int, default=16)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument(
        "--skip-first-scenarios", type=int, default=0,
        help="exclude this many leading scenarios from reported evaluation",
    )
    parser.add_argument(
        "--instance-name", default=None,
        help="evaluate only this discovered instance name",
    )
    parser.add_argument("--output", default="datasets/rl_checkpoint_results.json")
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _customer_count(csv_path: str) -> int:
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = sum(
            1 for row in csv.reader(handle)
            if row and any(cell.strip() for cell in row)
        )
    return max(rows - 2, 0)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    max_instances = (
        args.max_instances
        if args.max_instances is not None
        else args.num_instances
    )
    max_scenarios = (
        args.max_scenarios
        if args.max_scenarios is not None
        else args.num_scenarios
    )
    if args.skip_first_scenarios < 0:
        raise SystemExit("--skip-first-scenarios must be non-negative")
    if args.skip_first_scenarios >= max_scenarios:
        raise SystemExit(
            "--skip-first-scenarios must be smaller than the scenario limit"
        )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    weight = float(checkpoint.get("weight", 0.5))
    num_time_slots = float(checkpoint.get("num_time_slots", 20.0))
    if num_time_slots <= 0:
        raise SystemExit("checkpoint contains invalid num_time_slots")
    agent = PPOAgent.from_checkpoint(args.checkpoint, device=args.device)
    instances = discover_instances(
        args.target_path, max_instances, max_scenarios
    )
    if args.instance_name is not None:
        if args.instance_name not in instances:
            raise SystemExit(
                f"instance {args.instance_name!r} not found in {args.target_path}"
            )
        instances = {args.instance_name: instances[args.instance_name]}
    if not instances:
        raise SystemExit(f"No CSV instances found in {args.target_path}")

    print("=" * 72)
    print("Masked PPO checkpoint evaluation")
    print("=" * 72)
    print(
        f"checkpoint={os.path.abspath(args.checkpoint)} device={agent.device} "
        f"weight={weight:g}"
    )
    print(
        f"dataset={os.path.abspath(args.target_path)} "
        f"instances={len(instances)} scenarios<={max_scenarios}"
    )

    started = time.time()
    all_results = {}
    total_rollouts = 0
    for instance_name, scenario_files in instances.items():
        customer_count = _customer_count(scenario_files[0])
        num_trucks, truck_capacity = infer_vehicle_config(customer_count)
        problem_set = ProblemSet.load_from_csvs(
            scenario_files,
            truck_speed=1.0,
            truck_capacity=truck_capacity,
            num_trucks=num_trucks,
        )
        if len(problem_set) <= args.skip_first_scenarios:
            raise SystemExit(
                f"{instance_name} has no scenarios left after skipping "
                f"{args.skip_first_scenarios}"
            )
        rows = []
        for scenario_idx, problem in enumerate(problem_set, 1):
            if scenario_idx <= args.skip_first_scenarios:
                continue
            simulation = PolicySimulation(
                problem, agent, weight=weight, deterministic=True
            )
            time_slot = problem.depot.close / num_time_slots
            distance, profit = simulation.simulate_until(
                time_slot, float("inf")
            )
            routes = simulation.get_routes()
            row = {
                "scenario": scenario_idx,
                "distance": round(float(distance), 6),
                "profit": round(float(profit), 6),
                "fitness": round(
                    fitness_value(problem, (distance, profit), weight), 8
                ),
                "served": sum(
                    1 for route in routes for node in route if node != 0
                ),
                "dropped": len(simulation.dropped_requests),
                "decisions": len(simulation.steps),
                "routes": routes,
            }
            rows.append(row)
            total_rollouts += 1

        summary = {
            "customers": customer_count,
            "num_trucks": num_trucks,
            "truck_capacity": truck_capacity,
            "scenario_count": len(rows),
            "avg_fitness": sum(row["fitness"] for row in rows) / len(rows),
            "avg_distance": sum(row["distance"] for row in rows) / len(rows),
            "avg_profit": sum(row["profit"] for row in rows) / len(rows),
            "avg_served": sum(row["served"] for row in rows) / len(rows),
            "avg_dropped": sum(row["dropped"] for row in rows) / len(rows),
            "scenario_results": rows,
        }
        all_results[instance_name] = summary
        if not args.quiet:
            print(
                f"{instance_name}: fitness={summary['avg_fitness']:.6f} "
                f"distance={summary['avg_distance']:.3f} "
                f"profit={summary['avg_profit']:.3f}"
            )

    elapsed = time.time() - started
    scenario_rows = [
        row
        for instance in all_results.values()
        for row in instance["scenario_results"]
    ]
    total_customers = sum(
        instance["customers"] * instance["scenario_count"]
        for instance in all_results.values()
    )
    overall = {
        "instance_count": len(all_results),
        "scenario_count": len(scenario_rows),
        "avg_fitness": sum(row["fitness"] for row in scenario_rows) / len(scenario_rows),
        "macro_avg_fitness": sum(
            instance["avg_fitness"] for instance in all_results.values()
        ) / len(all_results),
        "avg_distance": sum(row["distance"] for row in scenario_rows) / len(scenario_rows),
        "avg_profit": sum(row["profit"] for row in scenario_rows) / len(scenario_rows),
        "avg_served": sum(row["served"] for row in scenario_rows) / len(scenario_rows),
        "avg_dropped": sum(row["dropped"] for row in scenario_rows) / len(scenario_rows),
        "served_ratio": (
            sum(row["served"] for row in scenario_rows) / total_customers
            if total_customers else 0.0
        ),
        "dropped_ratio": (
            sum(row["dropped"] for row in scenario_rows) / total_customers
            if total_customers else 0.0
        ),
    }
    result = {
        "algorithm": checkpoint.get("algorithm", "masked_ppo"),
        "pipeline": checkpoint.get("pipeline"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "target_path": str(Path(args.target_path).resolve()),
        "weight": weight,
        "num_time_slots": num_time_slots,
        "training_evaluations": checkpoint.get("evaluation_count"),
        "training_simulator_rollouts": checkpoint.get("simulator_rollouts"),
        "report_simulator_rollouts": total_rollouts,
        "skipped_leading_scenarios": args.skip_first_scenarios,
        "elapsed_s": elapsed,
        "overall": overall,
        "instances": all_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "instance", "scenario_count", "avg_fitness", "avg_distance",
            "avg_profit", "avg_served", "avg_dropped",
        ]
        with open(summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for instance_name, instance in all_results.items():
                writer.writerow({
                    "instance": instance_name,
                    **{field: instance[field] for field in fields[1:]},
                })
            writer.writerow({
                "instance": "OVERALL",
                **{field: overall[field] for field in fields[1:]},
            })
        print(f"summary_csv={summary_path.resolve()}")
    print(
        f"results={output.resolve()} report_rollouts={total_rollouts} "
        f"elapsed={elapsed:.1f}s"
    )
    print(
        f"OVERALL fitness={overall['avg_fitness']:.6f} "
        f"distance={overall['avg_distance']:.3f} "
        f"profit={overall['avg_profit']:.3f} "
        f"served={overall['avg_served']:.2f} "
        f"dropped={overall['avg_dropped']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
