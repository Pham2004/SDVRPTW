#!/usr/bin/env python3
"""Train a masked-PPO baseline on the repository SDVRPTW simulator.

Examples
--------
Full fair-budget run (defaults to POP_SIZE * NUM_GEN evaluations)::

    python python_src/rl_dvrptw.py datasets/h100_new 1 16

Quick smoke test::

    python python_src/rl_dvrptw.py datasets/toy5 1 3 \
        --max-evaluations 8 --evaluations-per-update 4 --ppo-epochs 1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import find_dotenv, load_dotenv
    dotenv_path = find_dotenv() or os.path.join(REPO_ROOT, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except ImportError:
    pass

import numpy as np
import torch

from main import (
    NUM_GEN,
    NUM_TIME_SLOT,
    POP_SIZE,
    SCENARIO_TRAIN_RATIO,
    STRESS_FACTOR,
    TRAIN_FACTOR,
    WEIGHT,
    discover_instances,
    fitness,
)
from rl.ppo import PPOConfig, PPOTrainer
from sim.problem import ProblemSet


_DATASET_CFG = {
    100: (10, 200.0),
    200: (50, 400.0),
    400: (100, 800.0),
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else int(default)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else float(default)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Masked PPO baseline for stochastic dynamic VRPTW"
    )
    parser.add_argument("target_path")
    parser.add_argument("num_instances", nargs="?", type=int, default=None)
    parser.add_argument("num_scenarios", nargs="?", type=int, default=16)
    parser.add_argument(
        "--max-evaluations", type=int,
        default=_env_int("MAX_EVALUATIONS", POP_SIZE * NUM_GEN),
        help=("strict policy-evaluation budget; default MAX_EVALUATIONS or "
              "POP_SIZE*NUM_GEN"),
    )
    parser.add_argument(
        "--evaluations-per-update", type=int,
        default=_env_int("RL_EVALUATIONS_PER_UPDATE", 32),
    )
    parser.add_argument("--ppo-epochs", type=int,
                        default=_env_int("RL_PPO_EPOCHS", 4))
    parser.add_argument("--minibatch-size", type=int,
                        default=_env_int("RL_MINIBATCH_SIZE", 256))
    parser.add_argument("--hidden-dim", type=int,
                        default=_env_int("RL_HIDDEN_DIM", 128))
    parser.add_argument("--learning-rate", type=float,
                        default=_env_float("RL_LEARNING_RATE", 3e-4))
    parser.add_argument("--entropy-coef", type=float,
                        default=_env_float("RL_ENTROPY_COEF", 0.01))
    parser.add_argument("--seed", type=int,
                        default=_env_int("SEED", 0))
    parser.add_argument("--device", default=os.environ.get("RL_DEVICE"))
    parser.add_argument("--output", default=None,
                        help="JSON result path (default: beside target folder)")
    parser.add_argument("--checkpoint-dir", default=os.path.join(REPO_ROOT, "rl_checkpoints"))
    parser.add_argument("--quiet", action="store_true")
    return parser


def _dataset_vehicle_config(first_csv: str):
    with open(first_csv, newline="") as fh:
        n_rows = sum(1 for row in csv.reader(fh)
                     if row and any(cell.strip() for cell in row))
    n_customers = max(n_rows - 2, 0)
    dataset_size = min(_DATASET_CFG, key=lambda size: abs(size - n_customers))
    return n_customers, dataset_size, _DATASET_CFG[dataset_size]


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.target_path):
        raise SystemExit(f"Dataset folder not found: {args.target_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    instances = discover_instances(
        args.target_path, args.num_instances, args.num_scenarios
    )
    if not instances:
        raise SystemExit(f"No CSV instances found in {args.target_path}")

    first_csv = next(iter(instances.values()))[0]
    n_customers, dataset_size, vehicle_cfg = _dataset_vehicle_config(first_csv)
    num_trucks, truck_capacity = vehicle_cfg
    config = PPOConfig(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        evaluations_per_update=args.evaluations_per_update,
    )

    print("=" * 72)
    print("Masked PPO baseline for SDVRPTW")
    print("=" * 72)
    print(f"dataset={args.target_path} instances={len(instances)} scenarios<={args.num_scenarios}")
    print(f"detected_customers={n_customers} config={dataset_size} "
          f"trucks={num_trucks} capacity={truck_capacity:g}")
    print(f"search_budget={args.max_evaluations} objective evaluations "
          f"(GP default={POP_SIZE}*{NUM_GEN}={POP_SIZE * NUM_GEN})")
    print(f"training_scenario_ratio={SCENARIO_TRAIN_RATIO:g} seed={args.seed}")

    all_results: Dict[str, dict] = {}
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for name, scenario_files in instances.items():
        print(f"\n--- instance={name} scenarios={len(scenario_files)} ---")
        problem_set = ProblemSet.load_from_csvs(
            scenario_files,
            truck_speed=1.0,
            truck_capacity=truck_capacity,
            num_trucks=num_trucks,
        )
        time_slots = [p.depot.close / NUM_TIME_SLOT for p in problem_set]
        train_count = max(1, math.ceil(
            len(problem_set) * SCENARIO_TRAIN_RATIO
        ))
        training_problems = [
            p.clone_training(slot * TRAIN_FACTOR, STRESS_FACTOR)
            for p, slot in zip(problem_set[:train_count], time_slots[:train_count])
        ]
        training_time_slots = [
            slot / STRESS_FACTOR for slot in time_slots[:train_count]
        ]

        trainer = PPOTrainer(
            training_problems=training_problems,
            training_time_slots=training_time_slots,
            fitness_fn=fitness,
            max_evaluations=args.max_evaluations,
            config=config,
            weight=WEIGHT,
            seed=args.seed,
            device=args.device,
        )
        started = time.time()
        trainer.train(progress=not args.quiet)
        elapsed = time.time() - started

        # Deterministic reporting is deliberately outside the search budget,
        # just like GP's full_result pass at the end of a generation.
        scenario_results = trainer.evaluate(problem_set.problems, time_slots)
        for i, row in enumerate(scenario_results, 1):
            row["scenario"] = i
            row["distance"] = round(row["distance"], 6)
            row["profit"] = round(row["profit"], 6)
            row["fitness"] = round(row["fitness"], 8)
            if not args.quiet:
                print(
                    f"  scenario={i:2d} distance={row['distance']:10.3f} "
                    f"profit={row['profit']:9.3f} fitness={row['fitness']:.6f} "
                    f"served={row['served']} dropped={row['dropped']}"
                )
        avg_fitness = sum(row["fitness"] for row in scenario_results) / len(scenario_results)
        avg_distance = sum(row["distance"] for row in scenario_results) / len(scenario_results)
        avg_profit = sum(row["profit"] for row in scenario_results) / len(scenario_results)

        checkpoint_path = checkpoint_dir / f"{_safe_name(name)}.pt"
        torch.save(trainer.checkpoint(), checkpoint_path)
        all_results[name] = {
            "algorithm": "masked_ppo",
            "avg_final_fitness": avg_fitness,
            "avg_distance": avg_distance,
            "avg_profit": avg_profit,
            "best_training_fitness": trainer.best_fitness,
            "objective_evaluations": trainer.evaluation_count,
            "training_simulator_rollouts": trainer.simulator_rollouts,
            "report_simulator_rollouts": len(problem_set),
            "training_scenarios_per_evaluation": train_count,
            "elapsed_s": elapsed,
            "checkpoint": str(checkpoint_path.resolve()),
            "scenario_results": scenario_results,
            "history": trainer.history,
        }
        print(
            f"done {name}: avg_fitness={avg_fitness:.6f} "
            f"objective_evals={trainer.evaluation_count} "
            f"training_rollouts={trainer.simulator_rollouts} elapsed={elapsed:.1f}s"
        )

    output_path = Path(args.output) if args.output else (
        Path(args.target_path).resolve().parent / "rl_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False,
                  default=_json_default)
    print(f"\nResults saved to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
