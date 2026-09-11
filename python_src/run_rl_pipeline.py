#!/usr/bin/env python3
"""Run the complete global masked-PPO pipeline for H100, H200, or H400."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate training data, train one global masked-PPO checkpoint, "
            "evaluate all scenarios, and aggregate results"
        )
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="datasets/h100_new",
        help="CSV dataset folder, e.g. datasets/h100_new, h200_new, h400_new",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="generation/reference CSV folder; defaults to target",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--max-evaluations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=16)
    parser.add_argument("--evaluations-per-update", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _run_stage(name: str, command: list[str]) -> None:
    print(f"\n{'=' * 72}")
    print(name)
    print("=" * 72)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "dataset"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    target = _absolute(args.target)
    reference = _absolute(args.reference_dir or str(target))
    if not target.is_dir():
        raise SystemExit(f"target dataset folder not found: {target}")
    if not reference.is_dir():
        raise SystemExit(f"reference folder not found: {reference}")
    if args.samples <= 0 or args.max_evaluations <= 0:
        raise SystemExit("--samples and --max-evaluations must be positive")
    if args.max_scenarios <= 0:
        raise SystemExit("--max-scenarios must be positive")
    if args.max_instances is not None and args.max_instances <= 0:
        raise SystemExit("--max-instances must be positive")

    dataset_name = _safe_name(target.name)
    if args.output_dir:
        output_dir = _absolute(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = (
            REPO_ROOT / "rl_runs"
            / f"{dataset_name}_seed{args.seed}_{stamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    dataset_path = output_dir / "training_dataset.pt"
    checkpoint_path = output_dir / "checkpoint.pt"
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "run_manifest.json"
    python = sys.executable
    started = time.time()

    generate_command = [
        python, str(HERE / "generate_rl_dataset.py"), str(reference),
        str(dataset_path), "--samples", str(args.samples),
        "--seed", str(args.seed), "--jitter", str(args.jitter),
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
        "--weight", str(args.weight), "--seed", str(args.seed),
    ]
    evaluate_command = [
        python, str(HERE / "evaluate_rl_checkpoint.py"),
        str(checkpoint_path), str(target),
        "--max-scenarios", str(args.max_scenarios),
        "--output", str(results_path), "--summary-csv", str(summary_path),
    ]
    if args.max_instances is not None:
        evaluate_command.extend(["--max-instances", str(args.max_instances)])
    if args.device:
        train_command.extend(["--device", args.device])
        evaluate_command.extend(["--device", args.device])
    if args.quiet:
        train_command.append("--quiet")
        evaluate_command.append("--quiet")

    try:
        _run_stage("STAGE 1/3 - Generate training dataset", generate_command)
        _run_stage("STAGE 2/3 - Train one global checkpoint", train_command)
        _run_stage("STAGE 3/3 - Evaluate and aggregate target", evaluate_command)
    except subprocess.CalledProcessError as error:
        print(
            f"\nPipeline failed with exit code {error.returncode}. "
            f"Partial artifacts remain in {output_dir}",
            file=sys.stderr,
        )
        return int(error.returncode or 1)

    with open(results_path, encoding="utf-8") as handle:
        results = json.load(handle)
    elapsed = time.time() - started
    manifest = {
        "status": "complete",
        "algorithm": "masked_ppo",
        "dataset_name": dataset_name,
        "seed": args.seed,
        "target": str(target),
        "reference_dir": str(reference),
        "samples": args.samples,
        "max_evaluations": args.max_evaluations,
        "device": args.device or "auto",
        "elapsed_s": elapsed,
        "artifacts": {
            "training_dataset": str(dataset_path),
            "checkpoint": str(checkpoint_path),
            "results_json": str(results_path),
            "summary_csv": str(summary_path),
        },
        "overall": results["overall"],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    overall = results["overall"]
    print(f"\n{'=' * 72}")
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(f"dataset={dataset_name}")
    print(f"output_dir={output_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"results={results_path}")
    print(f"summary={summary_path}")
    print(
        f"overall_fitness={overall['avg_fitness']:.6f} "
        f"distance={overall['avg_distance']:.3f} "
        f"profit={overall['avg_profit']:.3f} "
        f"scenarios={overall['scenario_count']} elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
