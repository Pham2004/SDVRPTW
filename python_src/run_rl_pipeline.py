#!/usr/bin/env python3
"""Run fair per-instance masked-PPO training on one exact scenario."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from main import discover_instances


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "For each benchmark instance, train one masked-PPO checkpoint on "
            "its first exact CSV scenario and evaluate the requested scenarios."
        )
    )
    parser.add_argument(
        "target", nargs="?", default="datasets/h100_new",
        help="CSV dataset folder, e.g. datasets/h100_new, h200_new, h400_new",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-evaluations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=16)
    parser.add_argument("--evaluations-per-update", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--exclude-training-scenario",
        action="store_true",
        help="exclude the first (training) scenario from final reporting",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"


def _run_stage(name: str, command: list[str], quiet: bool = False) -> None:
    if not quiet:
        print(f"\n{name}")
        print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _aggregate(all_results: dict) -> dict:
    rows = [
        row
        for instance in all_results.values()
        for row in instance["scenario_results"]
    ]
    total_customers = sum(
        instance["customers"] * instance["scenario_count"]
        for instance in all_results.values()
    )
    return {
        "instance_count": len(all_results),
        "scenario_count": len(rows),
        "avg_fitness": sum(row["fitness"] for row in rows) / len(rows),
        "macro_avg_fitness": sum(
            instance["avg_fitness"] for instance in all_results.values()
        ) / len(all_results),
        "avg_distance": sum(row["distance"] for row in rows) / len(rows),
        "avg_profit": sum(row["profit"] for row in rows) / len(rows),
        "avg_served": sum(row["served"] for row in rows) / len(rows),
        "avg_dropped": sum(row["dropped"] for row in rows) / len(rows),
        "served_ratio": sum(row["served"] for row in rows) / total_customers,
        "dropped_ratio": sum(row["dropped"] for row in rows) / total_customers,
    }


def _write_summary(path: Path, all_results: dict, overall: dict,
                   total_training_evaluations: int,
                   total_training_rollouts: int) -> None:
    fields = [
        "instance", "scenario_count", "avg_fitness", "avg_distance",
        "avg_profit", "avg_served", "avg_dropped",
        "training_evaluations", "training_simulator_rollouts",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for instance_name, instance in all_results.items():
            writer.writerow({
                "instance": instance_name,
                **{field: instance[field] for field in fields[1:]},
            })
        writer.writerow({
            "instance": "OVERALL",
            "scenario_count": overall["scenario_count"],
            "avg_fitness": overall["avg_fitness"],
            "avg_distance": overall["avg_distance"],
            "avg_profit": overall["avg_profit"],
            "avg_served": overall["avg_served"],
            "avg_dropped": overall["avg_dropped"],
            "training_evaluations": total_training_evaluations,
            "training_simulator_rollouts": total_training_rollouts,
        })


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    target = _absolute(args.target)
    if not target.is_dir():
        raise SystemExit(f"target dataset folder not found: {target}")
    if args.max_evaluations <= 0 or args.max_scenarios <= 0:
        raise SystemExit("evaluation and scenario limits must be positive")
    if args.exclude_training_scenario and args.max_scenarios <= 1:
        raise SystemExit(
            "held-out reporting requires --max-scenarios to be at least 2"
        )
    if args.max_instances is not None and args.max_instances <= 0:
        raise SystemExit("--max-instances must be positive")
    if not 0.0 <= args.weight <= 1.0:
        raise SystemExit("--weight must be in [0, 1]")

    instances = discover_instances(
        str(target), args.max_instances, args.max_scenarios
    )
    if not instances:
        raise SystemExit(f"no CSV instances found in {target}")

    if args.output_dir:
        output_dir = _absolute(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = (
            REPO_ROOT / "rl_runs"
            / f"{_safe_name(target.name)}_single_scenario_seed{args.seed}_{stamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    python = sys.executable
    started = time.time()
    all_results = {}
    training_sources = {}
    checkpoint_paths = {}
    total_training_evaluations = 0
    total_training_rollouts = 0
    total_report_rollouts = 0

    try:
        for instance_name, scenario_files in instances.items():
            # Fair protocol: one exact fixed training scenario per model.
            training_scenario = Path(scenario_files[0]).resolve()
            instance_dir = output_dir / _safe_name(instance_name)
            instance_dir.mkdir(parents=True, exist_ok=False)
            dataset_path = instance_dir / "training_dataset.pt"
            checkpoint_path = instance_dir / "checkpoint.pt"
            results_path = instance_dir / "results.json"
            summary_path = instance_dir / "summary.csv"

            generate_command = [
                python, str(HERE / "generate_rl_dataset.py"),
                str(training_scenario), str(dataset_path),
            ]
            train_command = [
                python, str(HERE / "train_rl_checkpoint.py"), str(dataset_path),
                "--output", str(checkpoint_path),
                "--max-evaluations", str(args.max_evaluations),
                "--evaluations-per-update", str(args.evaluations_per_update),
                "--ppo-epochs", str(args.ppo_epochs),
                "--minibatch-size", str(args.minibatch_size),
                "--hidden-dim", str(args.hidden_dim),
                "--learning-rate", str(args.learning_rate),
                "--entropy-coef", str(args.entropy_coef),
                "--weight", str(args.weight),
                "--seed", str(args.seed),
            ]
            evaluate_command = [
                python, str(HERE / "evaluate_rl_checkpoint.py"),
                str(checkpoint_path), str(target),
                "--instance-name", instance_name,
                "--max-scenarios", str(args.max_scenarios),
                "--output", str(results_path),
                "--summary-csv", str(summary_path),
            ]
            if args.exclude_training_scenario:
                evaluate_command.extend(["--skip-first-scenarios", "1"])
            if args.device:
                train_command.extend(["--device", args.device])
                evaluate_command.extend(["--device", args.device])
            if args.quiet:
                train_command.append("--quiet")
                evaluate_command.append("--quiet")

            _run_stage(
                f"{instance_name} - build exact one-scenario dataset",
                generate_command, args.quiet,
            )
            _run_stage(
                f"{instance_name} - train masked PPO",
                train_command, args.quiet,
            )
            _run_stage(
                f"{instance_name} - evaluate checkpoint",
                evaluate_command, args.quiet,
            )

            with open(results_path, encoding="utf-8") as handle:
                result = json.load(handle)
            if set(result["instances"]) != {instance_name}:
                raise RuntimeError(
                    f"evaluation for {instance_name} returned unexpected "
                    f"instances: {sorted(result['instances'])}"
                )
            instance_result = result["instances"][instance_name]
            instance_result["training_scenario"] = str(training_scenario)
            instance_result["training_evaluations"] = int(
                result["training_evaluations"]
            )
            instance_result["training_simulator_rollouts"] = int(
                result["training_simulator_rollouts"]
            )
            all_results[instance_name] = instance_result
            training_sources[instance_name] = str(training_scenario)
            checkpoint_paths[instance_name] = str(checkpoint_path.resolve())
            total_training_evaluations += instance_result["training_evaluations"]
            total_training_rollouts += instance_result["training_simulator_rollouts"]
            total_report_rollouts += int(result["report_simulator_rollouts"])
            print(
                f"{instance_name}: fitness={instance_result['avg_fitness']:.6f} "
                f"distance={instance_result['avg_distance']:.3f} "
                f"profit={instance_result['avg_profit']:.3f}"
            )
    except subprocess.CalledProcessError as error:
        print(
            f"\nPipeline failed with exit code {error.returncode}. "
            f"Partial artifacts remain in {output_dir}",
            file=sys.stderr,
        )
        return int(error.returncode or 1)

    elapsed = time.time() - started
    overall = _aggregate(all_results)
    result = {
        "algorithm": "masked_ppo",
        "protocol": "per_instance_exact_single_scenario",
        "target_path": str(target),
        "seed": args.seed,
        "training_scenarios_per_instance": 1,
        "exclude_training_scenario": args.exclude_training_scenario,
        "training_sources": training_sources,
        "training_evaluations": total_training_evaluations,
        "training_simulator_rollouts": total_training_rollouts,
        "report_simulator_rollouts": total_report_rollouts,
        "elapsed_s": elapsed,
        "overall": overall,
        "instances": all_results,
    }
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "run_manifest.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    _write_summary(
        summary_path, all_results, overall,
        total_training_evaluations, total_training_rollouts,
    )
    manifest = {
        "status": "complete",
        "algorithm": "masked_ppo",
        "protocol": result["protocol"],
        "dataset_name": target.name,
        "seed": args.seed,
        "target": str(target),
        "training_scenarios_per_instance": 1,
        "exclude_training_scenario": args.exclude_training_scenario,
        "max_evaluations_per_instance": args.max_evaluations,
        "instance_count": len(all_results),
        "training_evaluations": total_training_evaluations,
        "training_simulator_rollouts": total_training_rollouts,
        "report_simulator_rollouts": total_report_rollouts,
        "device": args.device or "auto",
        "elapsed_s": elapsed,
        "overall": overall,
        "checkpoints": checkpoint_paths,
        "artifacts": {
            "results_json": str(results_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"output_dir={output_dir.resolve()}")
    print(
        f"OVERALL fitness={overall['avg_fitness']:.6f} "
        f"distance={overall['avg_distance']:.3f} "
        f"profit={overall['avg_profit']:.3f} "
        f"train_rollouts={total_training_rollouts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
