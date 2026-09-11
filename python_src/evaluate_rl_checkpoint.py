#!/usr/bin/env python3
"""Stage 3: evaluate a trained global checkpoint on repository CSV data."""
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
        description="Evaluate one global masked-PPO checkpoint on CSV scenarios"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("target_path")
    parser.add_argument("num_instances", nargs="?", type=int, default=None)
    parser.add_argument("num_scenarios", nargs="?", type=int, default=16)
    parser.add_argument("--output", default="datasets/rl_checkpoint_results.json")
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
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    weight = float(checkpoint.get("weight", 0.5))
    num_time_slots = float(checkpoint.get("num_time_slots", 20.0))
    if num_time_slots <= 0:
        raise SystemExit("checkpoint contains invalid num_time_slots")
    agent = PPOAgent.from_checkpoint(args.checkpoint, device=args.device)
    instances = discover_instances(
        args.target_path, args.num_instances, args.num_scenarios
    )
    if not instances:
        raise SystemExit(f"No CSV instances found in {args.target_path}")

    print("=" * 72)
    print("Global masked PPO checkpoint evaluation")
    print("=" * 72)
    print(
        f"checkpoint={os.path.abspath(args.checkpoint)} device={agent.device} "
        f"weight={weight:g}"
    )
    print(
        f"dataset={os.path.abspath(args.target_path)} "
        f"instances={len(instances)} scenarios<={args.num_scenarios}"
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
        rows = []
        for scenario_idx, problem in enumerate(problem_set, 1):
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
        "elapsed_s": elapsed,
        "instances": all_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(
        f"results={output.resolve()} report_rollouts={total_rollouts} "
        f"elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
