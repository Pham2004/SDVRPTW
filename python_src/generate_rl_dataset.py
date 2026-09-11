#!/usr/bin/env python3
"""Stage 1: generate a global SDVRPTW training dataset."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rl.data import generate_from_reference, save_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic SDVRPTW Problems from the empirical schema of "
            "a reference dataset. This does not run training."
        )
    )
    parser.add_argument("reference_dir", help="folder containing reference CSVs")
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
    dataset = generate_from_reference(
        reference_dir=args.reference_dir,
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
