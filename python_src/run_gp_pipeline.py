#!/usr/bin/env python3
"""Train one GP policy per instance and write auditable result artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
FLEET_CONFIGS = {
    100: (10, 200.0),
    200: (50, 400.0),
    400: (100, 800.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run corrected per-instance GP, train on one scenario, evaluate "
            "the requested scenarios, and save JSON/CSV/manifest artifacts."
        )
    )
    parser.add_argument("target", help="dataset folder, e.g. datasets/h200_new")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=16)
    parser.add_argument("--train-scenarios", type=int, default=1)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="legacy shared depth override for both trees",
    )
    parser.add_argument("--routing-depth", type=int, default=None)
    parser.add_argument("--sequencing-depth", type=int, default=None)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--num-time-slots", type=float, default=20.0)
    parser.add_argument("--train-factor", type=float, default=2.0)
    parser.add_argument("--stress-factor", type=float, default=1.0)
    parser.add_argument(
        "--use-training-clone",
        action="store_true",
        help="use legacy clone_training transformation instead of exact CSV",
    )
    parser.add_argument(
        "--exclude-training-scenarios",
        action="store_true",
        help="exclude the first --train-scenarios files from final reporting",
    )
    return parser


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _customer_count(csv_path: str) -> int:
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = sum(
            1 for row in csv.reader(handle)
            if row and any(cell.strip() for cell in row)
        )
    return max(rows - 2, 0)


def _fleet(customer_count: int):
    size = min(FLEET_CONFIGS, key=lambda item: abs(item - customer_count))
    return FLEET_CONFIGS[size]


def _resolved_depths(args):
    routing_depth = (
        args.routing_depth if args.routing_depth is not None
        else args.max_depth if args.max_depth is not None
        else 8
    )
    sequencing_depth = (
        args.sequencing_depth if args.sequencing_depth is not None
        else args.max_depth if args.max_depth is not None
        else 6
    )
    return routing_depth, sequencing_depth


def _configure(args, output_dir: Path):
    routing_depth, sequencing_depth = _resolved_depths(args)
    values = {
        "SEED": args.seed,
        "NUM_GEN": args.generations,
        "POP_SIZE": args.population_size,
        "MAX_DEPTH": max(routing_depth, sequencing_depth),
        "MAX_DEPTH_ROUTING": routing_depth,
        "MAX_DEPTH_SEQUENCING": sequencing_depth,
        "CROSSOVER_RATE": args.crossover_rate,
        "MUTATION_RATE": args.mutation_rate,
        "WEIGHT": args.weight,
        "NUM_TIME_SLOT": args.num_time_slots,
        "TRAIN_FACTOR": args.train_factor,
        "STRESS_FACTOR": args.stress_factor,
        "PRINT_ROUTES": 0,
    }
    for name, value in values.items():
        os.environ[name] = str(value)
    log_path = str(output_dir / "gp.log")
    for name in ("LOG_GP", "LOG_LASTPOP", "LOG_LASTROUTE"):
        os.environ[name] = log_path
    return routing_depth, sequencing_depth


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_scenarios <= 0 or args.train_scenarios <= 0:
        raise SystemExit("scenario counts must be positive")
    if (
        args.exclude_training_scenarios
        and args.max_scenarios <= args.train_scenarios
    ):
        raise SystemExit(
            "held-out reporting requires --max-scenarios to exceed "
            "--train-scenarios"
        )
    if args.generations <= 0 or args.population_size <= 0:
        raise SystemExit("generations and population size must be positive")
    routing_depth, sequencing_depth = _resolved_depths(args)
    if routing_depth <= 0 or sequencing_depth <= 0:
        raise SystemExit("routing and sequencing depths must be positive")
    if not 0.0 <= args.weight <= 1.0:
        raise SystemExit("--weight must be in [0, 1]")

    target = _absolute(args.target)
    if not target.is_dir():
        raise SystemExit(f"dataset folder not found: {target}")
    if args.output_dir:
        output_dir = _absolute(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = REPO_ROOT / "gp_runs" / f"{target.name}_seed{args.seed}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    routing_depth, sequencing_depth = _configure(args, output_dir)

    # Import only after environment configuration because main.py reads its
    # experiment constants at import time.
    import main as gp_main
    from sim import mod as sim_mod
    from sim.problem import ProblemSet

    instances = gp_main.discover_instances(
        str(target), args.max_instances, args.max_scenarios
    )
    if not instances:
        raise SystemExit(f"no CSV instances found in {target}")

    started = time.time()
    all_results = {}
    total_objective_evaluations = 0
    total_training_rollouts = 0
    total_report_rollouts = 0

    for instance_name, scenario_files in instances.items():
        if args.train_scenarios > len(scenario_files):
            raise SystemExit(
                f"{instance_name} has only {len(scenario_files)} scenarios, "
                f"cannot train on {args.train_scenarios}"
            )
        customer_count = _customer_count(scenario_files[0])
        num_trucks, truck_capacity = _fleet(customer_count)
        problem_set = ProblemSet.load_from_csvs(
            scenario_files,
            truck_speed=1.0,
            truck_capacity=truck_capacity,
            num_trucks=num_trucks,
        )
        details = gp_main.gp(
            problem_set,
            training_scenario_count=args.train_scenarios,
            evaluate_full_each_generation=False,
            return_details=True,
            clone_training_scenarios=args.use_training_clone,
        )
        best = details["best"]
        total_objective_evaluations += details["objective_evaluations"]
        total_training_rollouts += details["training_simulator_rollouts"]

        start_idx = args.train_scenarios if args.exclude_training_scenarios else 0
        rows = []
        for scenario_idx in range(start_idx, len(problem_set)):
            problem = problem_set[scenario_idx]
            time_slot = problem.depot.close / gp_main.NUM_TIME_SLOT
            simulation = sim_mod.Simulation(
                problem, best.routing, best.sequencing
            )
            distance, profit = simulation.simulate_until(
                time_slot, float("inf")
            )
            routes = simulation.get_routes()
            rows.append({
                "scenario": scenario_idx + 1,
                "distance": round(float(distance), 6),
                "profit": round(float(profit), 6),
                "fitness": round(
                    float(gp_main.fitness(problem, (distance, profit))), 8
                ),
                "served": sum(
                    1 for route in routes for node in route if node != 0
                ),
                "dropped": len(simulation.dropped_requests),
                "routes": routes,
            })
            total_report_rollouts += 1

        summary = {
            "customers": customer_count,
            "num_trucks": num_trucks,
            "truck_capacity": truck_capacity,
            "training_scenarios": args.train_scenarios,
            "training_data_method": details["training_data_method"],
            "training_files": scenario_files[:args.train_scenarios],
            "objective_evaluations": details["objective_evaluations"],
            "training_simulator_rollouts": details["training_simulator_rollouts"],
            "scenario_count": len(rows),
            "avg_fitness": sum(row["fitness"] for row in rows) / len(rows),
            "avg_distance": sum(row["distance"] for row in rows) / len(rows),
            "avg_profit": sum(row["profit"] for row in rows) / len(rows),
            "avg_served": sum(row["served"] for row in rows) / len(rows),
            "avg_dropped": sum(row["dropped"] for row in rows) / len(rows),
            "routing": str(best.routing),
            "sequencing": str(best.sequencing),
            "routing_base64": best.routing.base64(),
            "sequencing_base64": best.sequencing.base64(),
            "scenario_results": rows,
        }
        all_results[instance_name] = summary
        print(
            f"{instance_name}: fitness={summary['avg_fitness']:.6f} "
            f"distance={summary['avg_distance']:.3f} "
            f"profit={summary['avg_profit']:.3f} "
            f"search_evals={summary['objective_evaluations']}"
        )

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
        "served_ratio": sum(row["served"] for row in scenario_rows) / total_customers,
        "dropped_ratio": sum(row["dropped"] for row in scenario_rows) / total_customers,
    }
    elapsed = time.time() - started
    result = {
        "algorithm": "corrected_genetic_programming",
        "protocol": "per_instance_fixed_training_scenarios",
        "target_path": str(target),
        "seed": args.seed,
        "routing_depth": routing_depth,
        "sequencing_depth": sequencing_depth,
        "training_scenarios_per_instance": args.train_scenarios,
        "exclude_training_scenarios": args.exclude_training_scenarios,
        "objective_evaluations": total_objective_evaluations,
        "training_simulator_rollouts": total_training_rollouts,
        "report_simulator_rollouts": total_report_rollouts,
        "elapsed_s": elapsed,
        "overall": overall,
        "instances": all_results,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    summary_path = output_dir / "summary.csv"
    fields = [
        "instance", "scenario_count", "avg_fitness", "avg_distance",
        "avg_profit", "avg_served", "avg_dropped", "objective_evaluations",
        "training_simulator_rollouts",
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
            "scenario_count": overall["scenario_count"],
            "avg_fitness": overall["avg_fitness"],
            "avg_distance": overall["avg_distance"],
            "avg_profit": overall["avg_profit"],
            "avg_served": overall["avg_served"],
            "avg_dropped": overall["avg_dropped"],
            "objective_evaluations": total_objective_evaluations,
            "training_simulator_rollouts": total_training_rollouts,
        })

    manifest = {
        "status": "complete",
        "algorithm": result["algorithm"],
        "protocol": result["protocol"],
        "dataset_name": target.name,
        "seed": args.seed,
        "routing_depth": routing_depth,
        "sequencing_depth": sequencing_depth,
        "parameters": vars(args),
        "target": str(target),
        "elapsed_s": elapsed,
        "objective_evaluations": total_objective_evaluations,
        "training_simulator_rollouts": total_training_rollouts,
        "report_simulator_rollouts": total_report_rollouts,
        "overall": overall,
        "artifacts": {
            "results_json": str(results_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "log": str((output_dir / "gp.log").resolve()),
        },
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"output_dir={output_dir.resolve()}")
    print(
        f"OVERALL fitness={overall['avg_fitness']:.6f} "
        f"distance={overall['avg_distance']:.3f} "
        f"profit={overall['avg_profit']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
