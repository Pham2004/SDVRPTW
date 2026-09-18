#!/usr/bin/env python3
"""Stage 1: build an SDVRPTW training dataset."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rl.data import dataset_from_scenario, generate_from_reference, save_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact one-scenario dataset from a CSV, or generate "
            "synthetic problems from a reference folder."
        )
    )
    parser.add_argument(
        "reference",
        help="one CSV scenario (exact mode) or a folder of reference CSVs",
    )
    parser.add_argument("output", help="output .pt training dataset")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--customers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--truck-speed", type=float, default=1.0)
    parser.add_argument("--num-trucks", type=int, default=None)
    parser.add_argument("--truck-capacity", type=float, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if os.path.isfile(args.reference):
        if not args.reference.lower().endswith(".csv"):
            raise SystemExit("a file reference must be a CSV scenario")
        dataset = dataset_from_scenario(
            scenario_path=args.reference,
            truck_speed=args.truck_speed,
            num_trucks=args.num_trucks,
            truck_capacity=args.truck_capacity,
        )
    else:
        dataset = generate_from_reference(
            reference_dir=args.reference,
            sample_count=args.samples,
            customer_count=args.customers,
            seed=args.seed,
            jitter=args.jitter,
            truck_speed=args.truck_speed,
            num_trucks=args.num_trucks,
            truck_capacity=args.truck_capacity,
        )
    save_dataset(dataset, args.output)
    shape = tuple(dataset["nodes"].shape)
    config = dataset["vehicle_config"]
    print(f"saved={os.path.abspath(args.output)}")
    print(f"shape={shape} node_features={dataset['node_features']}")
    print(
        "fleet="
        f"{config['num_trucks']} trucks, capacity={config['truck_capacity']:g}, "
        f"speed={config['truck_speed']:g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
