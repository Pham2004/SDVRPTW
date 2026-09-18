#!/usr/bin/env python3
"""Stage 2: train a masked-PPO policy and save its checkpoint."""
from __future__ import annotations

import argparse
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

from main import NUM_GEN, NUM_TIME_SLOT, POP_SIZE, WEIGHT
from rl.data import TensorProblemDataset, fitness_value, load_dataset
from rl.ppo import PPOConfig, PPOTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one masked-PPO checkpoint on an SDVRPTW dataset"
    )
    parser.add_argument("dataset", help="training .pt from generate_rl_dataset.py")
    parser.add_argument("--output", default="rl_checkpoints/sdvrptw_global.pt")
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=POP_SIZE * NUM_GEN,
        help=(
            "strict number of training simulator rollouts; default equals "
            f"GP POP_SIZE*NUM_GEN ({POP_SIZE * NUM_GEN})"
        ),
    )
    parser.add_argument("--evaluations-per-update", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--weight", type=float, default=WEIGHT)
    parser.add_argument("--num-time-slots", type=float, default=NUM_TIME_SLOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_evaluations <= 0:
        raise SystemExit("--max-evaluations must be positive")
    if not 0.0 <= args.weight <= 1.0:
        raise SystemExit("--weight must be in [0, 1]")
    if args.num_time_slots <= 0:
        raise SystemExit("--num-time-slots must be positive")

    dataset = load_dataset(args.dataset)
    problems = TensorProblemDataset(dataset)
    time_slots = problems.time_slots(args.num_time_slots)
    config = PPOConfig(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        evaluations_per_update=args.evaluations_per_update,
    )

    def objective(problem, result):
        return fitness_value(problem, result, args.weight)

    # One problem is sampled per objective evaluation. For the fair pipeline,
    # the dataset itself contains exactly one fixed CSV scenario, so every
    # stochastic policy rollout sees that same scenario.
    trainer = PPOTrainer(
        training_problems=problems,
        training_time_slots=time_slots,
        fitness_fn=objective,
        max_evaluations=args.max_evaluations,
        config=config,
        weight=args.weight,
        seed=args.seed,
        device=args.device,
        problems_per_evaluation=1,
        restore_best=False,
    )
    print("=" * 72)
    print("Masked PPO training for repository SDVRPTW")
    print("=" * 72)
    print(
        f"dataset={os.path.abspath(args.dataset)} samples={len(problems)} "
        f"device={trainer.agent.device}"
    )
    print(
        f"budget={args.max_evaluations} simulator rollouts "
        f"(GP configured budget={POP_SIZE}*{NUM_GEN}={POP_SIZE * NUM_GEN})"
    )
    started = time.time()
    trainer.train(progress=not args.quiet)
    elapsed = time.time() - started

    checkpoint = trainer.checkpoint()
    checkpoint.update({
        "pipeline": "dataset_train_checkpoint_evaluate",
        "training_data_method": dataset.get("generation", {}).get("method"),
        "selection": "final_policy",
        "weight": float(args.weight),
        "num_time_slots": float(args.num_time_slots),
        "training_dataset": str(Path(args.dataset).resolve()),
        "dataset_generation": dataset.get("generation", {}),
        "vehicle_config": dataset.get("vehicle_config", {}),
        "training_budget_unit": "simulator_rollout",
        "elapsed_s": elapsed,
        "history": trainer.history,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(
        f"checkpoint={output.resolve()} evaluations={trainer.evaluation_count} "
        f"rollouts={trainer.simulator_rollouts} elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
