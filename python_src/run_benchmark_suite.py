#!/usr/bin/env python3
"""Run and aggregate the fair 5-seed GP/RL benchmark suite."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_DATASETS = ["h100_new", "h200_new", "h400_new"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
METRICS = [
    "avg_fitness", "macro_avg_fitness", "avg_distance", "avg_profit",
    "avg_served", "avg_dropped", "served_ratio", "dropped_ratio",
]
BUDGET_METRICS = [
    "elapsed_s", "training_evaluations", "training_simulator_rollouts",
    "report_simulator_rollouts",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run GP and masked PPO on H100/H200/H400 for five paired seeds, "
            "resume completed runs, and aggregate all results."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--algorithms", nargs="+", choices=("gp", "rl"),
        default=["gp", "rl"],
    )
    parser.add_argument(
        "--output-dir", default="benchmark_runs/gp_rl_5seeds",
        help="stable suite directory; rerunning the command resumes it",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=16)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--gp-generations", type=int, default=100)
    parser.add_argument("--gp-population-size", type=int, default=100)
    parser.add_argument("--routing-depth", type=int, default=8)
    parser.add_argument("--sequencing-depth", type=int, default=6)
    parser.add_argument("--rl-max-evaluations", type=int, default=10_000)
    parser.add_argument("--rl-evaluations-per-update", type=int, default=32)
    parser.add_argument("--rl-ppo-epochs", type=int, default=4)
    parser.add_argument("--rl-minibatch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="RL device, e.g. cuda or cpu")
    parser.add_argument(
        "--include-training-scenario", action="store_true",
        help="also report scenario 1; default reports held-out scenarios only",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="continue other runs after one child run fails",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_dataset(value: str) -> Path:
    direct = _absolute(value)
    under_datasets = (REPO_ROOT / "datasets" / value).resolve()
    if direct.is_dir():
        return direct
    if under_datasets.is_dir():
        return under_datasets
    raise SystemExit(f"dataset folder not found: {value}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _run_complete(run_dir: Path, dataset: Path, seed: int) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "results.json"
    if not manifest_path.is_file() or not results_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        result = _read_json(results_path)
        return (
            manifest.get("status") == "complete"
            and int(manifest.get("seed")) == seed
            and Path(result["target_path"]).resolve() == dataset
            and bool(result.get("instances"))
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _archive_incomplete(run_dir: Path) -> Path:
    suffix = time.strftime("%Y%m%d-%H%M%S")
    archived = run_dir.with_name(f"{run_dir.name}.incomplete-{suffix}")
    counter = 1
    while archived.exists():
        archived = run_dir.with_name(
            f"{run_dir.name}.incomplete-{suffix}-{counter}"
        )
        counter += 1
    shutil.move(str(run_dir), str(archived))
    return archived


def _command(args, algorithm: str, dataset: Path, seed: int,
             run_dir: Path) -> list[str]:
    common = [
        "--output-dir", str(run_dir),
        "--seed", str(seed),
        "--max-scenarios", str(args.max_scenarios),
        "--weight", str(args.weight),
    ]
    if args.max_instances is not None:
        common.extend(["--max-instances", str(args.max_instances)])

    if algorithm == "gp":
        command = [
            sys.executable, "-u", str(HERE / "run_gp_pipeline.py"),
            str(dataset), *common,
            "--generations", str(args.gp_generations),
            "--population-size", str(args.gp_population_size),
            "--routing-depth", str(args.routing_depth),
            "--sequencing-depth", str(args.sequencing_depth),
            "--train-scenarios", "1",
        ]
        if not args.include_training_scenario:
            command.append("--exclude-training-scenarios")
        return command

    command = [
        sys.executable, "-u", str(HERE / "run_rl_pipeline.py"),
        str(dataset), *common,
        "--max-evaluations", str(args.rl_max_evaluations),
        "--evaluations-per-update", str(args.rl_evaluations_per_update),
        "--ppo-epochs", str(args.rl_ppo_epochs),
        "--minibatch-size", str(args.rl_minibatch_size),
        "--quiet",
    ]
    if args.device:
        command.extend(["--device", args.device])
    if not args.include_training_scenario:
        command.append("--exclude-training-scenario")
    return command


def _run_logged(command: list[str], log_path: Path, verbose: bool) -> int:
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            stripped = line.rstrip()
            if verbose or stripped.startswith("OVERALL") or "fitness=" in stripped:
                print(f"    {stripped}", flush=True)
        return process.wait()


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    return 1.96 * _std(values) / math.sqrt(len(values)) if values else 0.0


def _load_run_row(algorithm: str, dataset_name: str, seed: int,
                  run_dir: Path) -> tuple[dict, dict]:
    result = _read_json(run_dir / "results.json")
    overall = result["overall"]
    training_evaluations = result.get(
        "objective_evaluations", result.get("training_evaluations")
    )
    row = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "seed": seed,
        "status": "complete",
        "run_dir": str(run_dir.resolve()),
        "elapsed_s": result.get("elapsed_s"),
        "training_evaluations": training_evaluations,
        "training_simulator_rollouts": result.get("training_simulator_rollouts"),
        "report_simulator_rollouts": result.get("report_simulator_rollouts"),
        **{metric: overall.get(metric) for metric in METRICS},
    }
    return row, result


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output_dir: Path, datasets: list[Path], seeds: list[int],
              algorithms: list[str]) -> dict:
    run_rows = []
    instance_values: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for dataset in datasets:
        for algorithm in algorithms:
            for seed in seeds:
                run_dir = output_dir / dataset.name / algorithm / f"seed_{seed}"
                if not _run_complete(run_dir, dataset, seed):
                    continue
                row, result = _load_run_row(
                    algorithm, dataset.name, seed, run_dir
                )
                run_rows.append(row)
                for instance_name, instance in result["instances"].items():
                    key = (dataset.name, algorithm, instance_name)
                    bucket = instance_values.setdefault(
                        key, {metric: [] for metric in METRICS}
                    )
                    for metric in METRICS:
                        value = instance.get(metric)
                        if value is not None:
                            bucket[metric].append(float(value))

    run_fields = [
        "dataset", "algorithm", "seed", "status", "run_dir", "elapsed_s",
        "training_evaluations", "training_simulator_rollouts",
        "report_simulator_rollouts", *METRICS,
    ]
    _write_csv(output_dir / "runs.csv", run_rows, run_fields)

    summary_rows = []
    for dataset in datasets:
        for algorithm in algorithms:
            group = [
                row for row in run_rows
                if row["dataset"] == dataset.name
                and row["algorithm"] == algorithm
            ]
            if not group:
                continue
            summary = {
                "dataset": dataset.name,
                "algorithm": algorithm,
                "seeds_completed": len(group),
                "seeds_expected": len(seeds),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                summary[f"{metric}_mean"] = _mean(values)
                summary[f"{metric}_std"] = _std(values)
                summary[f"{metric}_ci95"] = _ci95(values)
            for metric in BUDGET_METRICS:
                values = [
                    float(row[metric]) for row in group
                    if row.get(metric) is not None
                ]
                if values:
                    summary[f"{metric}_mean"] = _mean(values)
                    summary[f"{metric}_std"] = _std(values)
            summary_rows.append(summary)
    summary_fields = ["dataset", "algorithm", "seeds_completed", "seeds_expected"]
    for metric in METRICS:
        summary_fields.extend([
            f"{metric}_mean", f"{metric}_std", f"{metric}_ci95"
        ])
    for metric in BUDGET_METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    _write_csv(output_dir / "summary_by_algorithm.csv", summary_rows, summary_fields)

    instance_rows = []
    for (dataset_name, algorithm, instance_name), buckets in sorted(
        instance_values.items()
    ):
        row = {
            "dataset": dataset_name,
            "algorithm": algorithm,
            "instance": instance_name,
            "seeds_completed": max((len(v) for v in buckets.values()), default=0),
        }
        for metric, values in buckets.items():
            if values:
                row[f"{metric}_mean"] = _mean(values)
                row[f"{metric}_std"] = _std(values)
        instance_rows.append(row)
    instance_fields = ["dataset", "algorithm", "instance", "seeds_completed"]
    for metric in METRICS:
        instance_fields.extend([f"{metric}_mean", f"{metric}_std"])
    _write_csv(output_dir / "summary_by_instance.csv", instance_rows, instance_fields)

    lookup = {
        (row["dataset"], row["algorithm"], int(row["seed"])): row
        for row in run_rows
    }
    paired_rows = []
    if "gp" in algorithms and "rl" in algorithms:
        for dataset in datasets:
            for seed in seeds:
                gp = lookup.get((dataset.name, "gp", seed))
                rl = lookup.get((dataset.name, "rl", seed))
                if not gp or not rl:
                    continue
                delta = float(rl["avg_fitness"]) - float(gp["avg_fitness"])
                paired_rows.append({
                    "dataset": dataset.name,
                    "seed": seed,
                    "gp_avg_fitness": gp["avg_fitness"],
                    "rl_avg_fitness": rl["avg_fitness"],
                    "rl_minus_gp_fitness": delta,
                    "winner": "gp" if delta > 0 else "rl" if delta < 0 else "tie",
                })
    paired_fields = [
        "dataset", "seed", "gp_avg_fitness", "rl_avg_fitness",
        "rl_minus_gp_fitness", "winner",
    ]
    _write_csv(output_dir / "paired_comparison.csv", paired_rows, paired_fields)

    comparison_rows = []
    for dataset in datasets:
        group = [row for row in paired_rows if row["dataset"] == dataset.name]
        if not group:
            continue
        gp_values = [float(row["gp_avg_fitness"]) for row in group]
        rl_values = [float(row["rl_avg_fitness"]) for row in group]
        deltas = [float(row["rl_minus_gp_fitness"]) for row in group]
        comparison_rows.append({
            "dataset": dataset.name,
            "paired_seeds": len(group),
            "gp_fitness_mean": _mean(gp_values),
            "gp_fitness_std": _std(gp_values),
            "rl_fitness_mean": _mean(rl_values),
            "rl_fitness_std": _std(rl_values),
            "rl_minus_gp_mean": _mean(deltas),
            "rl_minus_gp_std": _std(deltas),
            "gp_wins": sum(row["winner"] == "gp" for row in group),
            "rl_wins": sum(row["winner"] == "rl" for row in group),
            "ties": sum(row["winner"] == "tie" for row in group),
        })
    comparison_fields = [
        "dataset", "paired_seeds", "gp_fitness_mean", "gp_fitness_std",
        "rl_fitness_mean", "rl_fitness_std", "rl_minus_gp_mean",
        "rl_minus_gp_std", "gp_wins", "rl_wins", "ties",
    ]
    _write_csv(
        output_dir / "comparison_summary.csv", comparison_rows,
        comparison_fields,
    )

    aggregate_data = {
        "generated_at": _utc_now(),
        "lower_fitness_is_better": True,
        "runs": run_rows,
        "summary_by_algorithm": summary_rows,
        "summary_by_instance": instance_rows,
        "paired_comparison": paired_rows,
        "comparison_summary": comparison_rows,
    }
    _write_json(output_dir / "aggregate.json", aggregate_data)
    return aggregate_data


def _validate(args) -> None:
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must contain unique values")
    if args.max_scenarios <= 1 and not args.include_training_scenario:
        raise SystemExit("held-out reporting requires --max-scenarios >= 2")
    positive = {
        "--gp-generations": args.gp_generations,
        "--gp-population-size": args.gp_population_size,
        "--routing-depth": args.routing_depth,
        "--sequencing-depth": args.sequencing_depth,
        "--rl-max-evaluations": args.rl_max_evaluations,
        "--rl-evaluations-per-update": args.rl_evaluations_per_update,
        "--rl-ppo-epochs": args.rl_ppo_epochs,
        "--rl-minibatch-size": args.rl_minibatch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    if not 0.0 <= args.weight <= 1.0:
        raise SystemExit("--weight must be in [0, 1]")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _validate(args)
    datasets = [_resolve_dataset(value) for value in args.datasets]
    if len({dataset.name for dataset in datasets}) != len(datasets):
        raise SystemExit("dataset folder names must be unique")
    algorithms = list(dict.fromkeys(args.algorithms))
    output_dir = _absolute(args.output_dir)

    nominal_gp_budget = args.gp_generations * args.gp_population_size
    if "gp" in algorithms and "rl" in algorithms:
        print(
            f"Nominal per-instance budgets: GP={nominal_gp_budget}, "
            f"RL={args.rl_max_evaluations}"
        )
        if nominal_gp_budget != args.rl_max_evaluations:
            print("WARNING: nominal GP and RL evaluation budgets differ")

    planned = [
        (dataset, seed, algorithm)
        for dataset in datasets
        for seed in args.seeds
        for algorithm in algorithms
    ]
    print(
        f"Suite: {len(datasets)} datasets x {len(args.seeds)} seeds x "
        f"{len(algorithms)} algorithms = {len(planned)} runs"
    )
    print(f"Output: {output_dir}")

    if args.dry_run:
        for index, (dataset, seed, algorithm) in enumerate(planned, 1):
            run_dir = output_dir / dataset.name / algorithm / f"seed_{seed}"
            print(
                f"[{index}/{len(planned)}] ",
                subprocess.list2cmdline(
                    _command(args, algorithm, dataset, seed, run_dir)
                ),
            )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "datasets": [str(dataset) for dataset in datasets],
        "seeds": args.seeds,
        "algorithms": algorithms,
        "max_instances": args.max_instances,
        "max_scenarios": args.max_scenarios,
        "weight": args.weight,
        "include_training_scenario": args.include_training_scenario,
        "gp_generations": args.gp_generations,
        "gp_population_size": args.gp_population_size,
        "routing_depth": args.routing_depth,
        "sequencing_depth": args.sequencing_depth,
        "rl_max_evaluations": args.rl_max_evaluations,
        "rl_evaluations_per_update": args.rl_evaluations_per_update,
        "rl_ppo_epochs": args.rl_ppo_epochs,
        "rl_minibatch_size": args.rl_minibatch_size,
        "device": args.device or "auto",
    }
    suite_manifest_path = output_dir / "suite_manifest.json"
    if suite_manifest_path.is_file():
        existing = _read_json(suite_manifest_path)
        if existing.get("config") != config:
            raise SystemExit(
                f"{output_dir} contains a suite with different parameters; "
                "choose another --output-dir"
            )
        manifest = existing
        manifest["status"] = "running"
        manifest["resumed_at"] = _utc_now()
    else:
        manifest = {
            "status": "running",
            "created_at": _utc_now(),
            "config": config,
            "runs": {},
        }
    _write_json(suite_manifest_path, manifest)

    failures = []
    suite_started = time.time()
    for index, (dataset, seed, algorithm) in enumerate(planned, 1):
        key = f"{dataset.name}/{algorithm}/seed_{seed}"
        run_dir = output_dir / key
        if _run_complete(run_dir, dataset, seed):
            print(f"[{index}/{len(planned)}] SKIP complete {key}", flush=True)
            manifest["runs"][key] = {
                "status": "complete", "run_dir": str(run_dir.resolve())
            }
            _write_json(suite_manifest_path, manifest)
            continue
        if run_dir.exists():
            archived = _archive_incomplete(run_dir)
            print(f"[{index}/{len(planned)}] archived partial run: {archived}")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        command = _command(args, algorithm, dataset, seed, run_dir)
        log_path = run_dir.parent / f"seed_{seed}.console.log"
        print(f"[{index}/{len(planned)}] RUN {key}", flush=True)
        started = time.time()
        returncode = _run_logged(command, log_path, args.verbose)
        elapsed = time.time() - started
        if returncode == 0 and _run_complete(run_dir, dataset, seed):
            print(f"[{index}/{len(planned)}] DONE {key} ({elapsed:.1f}s)")
            status = "complete"
        else:
            status = "failed"
            failures.append(key)
            print(
                f"[{index}/{len(planned)}] FAILED {key}; see {log_path}",
                file=sys.stderr,
            )
        manifest["runs"][key] = {
            "status": status,
            "returncode": returncode,
            "elapsed_s": elapsed,
            "run_dir": str(run_dir.resolve()),
            "console_log": str(log_path.resolve()),
        }
        _write_json(suite_manifest_path, manifest)
        if failures and not args.continue_on_error:
            break

    aggregated = aggregate(output_dir, datasets, args.seeds, algorithms)
    expected = len(planned)
    completed = len(aggregated["runs"])
    manifest["status"] = (
        "complete" if completed == expected and not failures else "partial"
    )
    manifest["completed_at"] = _utc_now()
    manifest["elapsed_this_invocation_s"] = time.time() - suite_started
    manifest["completed_runs"] = completed
    manifest["expected_runs"] = expected
    manifest["failures"] = failures
    manifest["artifacts"] = {
        name: str((output_dir / name).resolve())
        for name in (
            "runs.csv", "summary_by_algorithm.csv", "summary_by_instance.csv",
            "paired_comparison.csv", "comparison_summary.csv", "aggregate.json",
        )
    }
    _write_json(suite_manifest_path, manifest)

    print(f"Completed {completed}/{expected} runs")
    print(f"Summary: {(output_dir / 'comparison_summary.csv').resolve()}")
    return 0 if completed == expected and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
