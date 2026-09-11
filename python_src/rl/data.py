"""Synthetic SDVRPTW dataset format used by the global RL pipeline.

The generated artifact deliberately stores the repository's complete CSV
schema.  Training still runs through :class:`sim.mod.Simulation`; this module
only converts compact tensors back into the canonical ``Problem`` objects.
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from ..sim.problem import Problem, Request
except ImportError:
    from sim.problem import Problem, Request


DATASET_FORMAT = "sdvrptw_global_rl_v1"
NODE_FEATURES = (
    "x",
    "y",
    "demand",
    "open",
    "close",
    "service_time",
    "drone_serve",
    "reveal_time",
    "profit",
    "type",
)

VEHICLE_CONFIGS = {
    100: (10, 200.0),
    200: (50, 400.0),
    400: (100, 800.0),
}


def infer_vehicle_config(customer_count: int) -> Tuple[int, float]:
    """Use the same size-to-fleet mapping as ``main.py``."""
    size = min(VEHICLE_CONFIGS, key=lambda item: abs(item - customer_count))
    return VEHICLE_CONFIGS[size]


def fitness_value(problem: Problem, result: Tuple[float, float],
                  weight: float) -> float:
    """Exact repository fitness with an explicit, checkpointed weight."""
    distance, profit = result
    max_distance = (
        problem.truck_speed * problem.depot.close * problem.num_trucks
    )
    if max_distance <= 0:
        raise ValueError("fitness normalization requires positive horizon")
    max_profit = sum(float(req.profit) for req in problem.requests) or 1.0
    return (
        weight * distance / max_distance
        + (1.0 - weight) * (max_profit - profit) / max_profit
    )


def discover_csvs(reference_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(reference_dir, "**", "*.csv"),
                             recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(reference_dir, "*.csv")))
    return paths


def _float(row: Dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return float(default)


def _read_profile(path: str) -> Dict[str, object]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"reference scenario has no customers: {path}")

    def convert(row: Dict[str, str]) -> List[float]:
        return [
            _float(row, "x"),
            _float(row, "y"),
            _float(row, "demand"),
            _float(row, "open"),
            _float(row, "close"),
            _float(row, "servicetime", "service_time"),
            _float(row, "drone_serve"),
            _float(row, "time", "reveal_time"),
            _float(row, "profit"),
            _float(row, "type"),
        ]

    depot = convert(rows[0])
    return {
        "path": path,
        "horizon": float(depot[4]),
        "depot": depot,
        "customers": [convert(row) for row in rows[1:]],
    }


def generate_from_reference(
    reference_dir: str,
    sample_count: int,
    customer_count: Optional[int] = None,
    seed: int = 0,
    jitter: float = 0.03,
    truck_speed: float = 1.0,
    num_trucks: Optional[int] = None,
    truck_capacity: Optional[float] = None,
) -> Dict[str, object]:
    """Generate synthetic instances by empirical, horizon-stratified bootstrap.

    A customer row is sampled jointly so demand, time-window, reveal time,
    profit and type correlations are retained.  Continuous fields are jittered
    and clipped to valid bounds, so generated instances are not copies of a
    benchmark scenario.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 <= jitter <= 0.5:
        raise ValueError("jitter must be in [0, 0.5]")

    csv_paths = discover_csvs(reference_dir)
    if not csv_paths:
        raise ValueError(f"no CSV scenarios found in {reference_dir}")
    profiles = [_read_profile(path) for path in csv_paths]
    observed_counts = {len(profile["customers"]) for profile in profiles}
    if customer_count is None:
        if len(observed_counts) != 1:
            raise ValueError(
                "reference scenarios have different sizes; pass customer_count"
            )
        customer_count = observed_counts.pop()
    if customer_count <= 0:
        raise ValueError("customer_count must be positive")

    inferred_trucks, inferred_capacity = infer_vehicle_config(customer_count)
    num_trucks = inferred_trucks if num_trucks is None else int(num_trucks)
    truck_capacity = (
        inferred_capacity if truck_capacity is None else float(truck_capacity)
    )
    if num_trucks <= 0 or truck_capacity <= 0 or truck_speed <= 0:
        raise ValueError("vehicle configuration must be positive")

    by_horizon: Dict[float, List[List[float]]] = defaultdict(list)
    for profile in profiles:
        by_horizon[float(profile["horizon"])].extend(profile["customers"])

    all_customers = np.asarray(
        [row for rows in by_horizon.values() for row in rows], dtype=np.float64
    )
    x_min, x_max = float(all_customers[:, 0].min()), float(all_customers[:, 0].max())
    y_min, y_max = float(all_customers[:, 1].min()), float(all_customers[:, 1].max())
    coord_scale = max(x_max - x_min, y_max - y_min, 1.0)
    rng = np.random.default_rng(seed)
    nodes = np.empty(
        (sample_count, customer_count + 1, len(NODE_FEATURES)),
        dtype=np.float32,
    )

    for sample_idx in range(sample_count):
        profile = profiles[int(rng.integers(0, len(profiles)))]
        horizon = float(profile["horizon"])
        depot = np.asarray(profile["depot"], dtype=np.float64).copy()
        depot[2] = 0.0
        depot[3] = 0.0
        depot[7] = 0.0
        nodes[sample_idx, 0] = depot

        pool = np.asarray(by_horizon[horizon], dtype=np.float64)
        chosen = pool[rng.integers(0, len(pool), size=customer_count)].copy()
        if jitter > 0:
            chosen[:, 0:2] += rng.normal(
                0.0, jitter * coord_scale, size=(customer_count, 2)
            )
            chosen[:, 0] = np.clip(chosen[:, 0], x_min, x_max)
            chosen[:, 1] = np.clip(chosen[:, 1], y_min, y_max)

            time_noise = rng.normal(
                0.0, jitter * horizon, size=(customer_count, 3)
            )
            opens = np.clip(chosen[:, 3] + time_noise[:, 0], 0.0, horizon)
            widths = np.maximum(
                chosen[:, 4] - chosen[:, 3] + time_noise[:, 1], 1.0
            )
            closes = np.minimum(opens + widths, horizon)
            reveals = np.clip(chosen[:, 7] + time_noise[:, 2], 0.0, closes)
            chosen[:, 3], chosen[:, 4], chosen[:, 7] = opens, closes, reveals

            multipliers = rng.lognormal(
                mean=0.0, sigma=jitter, size=(customer_count, 2)
            )
            chosen[:, 2] = np.maximum(chosen[:, 2] * multipliers[:, 0], 1e-6)
            chosen[:, 8] = np.maximum(chosen[:, 8] * multipliers[:, 1], 0.0)

        # No request may exceed a vehicle's full capacity in the canonical sim.
        chosen[:, 2] = np.minimum(chosen[:, 2], truck_capacity)
        nodes[sample_idx, 1:] = chosen.astype(np.float32)

    return {
        "format": DATASET_FORMAT,
        "node_features": list(NODE_FEATURES),
        "nodes": torch.from_numpy(nodes),
        "vehicle_config": {
            "truck_speed": float(truck_speed),
            "truck_capacity": float(truck_capacity),
            "num_trucks": int(num_trucks),
        },
        "generation": {
            "method": "horizon_stratified_empirical_bootstrap",
            "reference_dir": str(Path(reference_dir).resolve()),
            "reference_scenarios": len(csv_paths),
            "sample_count": int(sample_count),
            "customer_count": int(customer_count),
            "seed": int(seed),
            "jitter": float(jitter),
        },
    }


def save_dataset(dataset: Dict[str, object], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output)


def load_dataset(dataset_path: str) -> Dict[str, object]:
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(dataset, dict) or dataset.get("format") != DATASET_FORMAT:
        raise ValueError("unsupported RL dataset format")
    if dataset.get("node_features") != list(NODE_FEATURES):
        raise ValueError("RL dataset feature schema mismatch")
    nodes = dataset.get("nodes")
    if not isinstance(nodes, torch.Tensor) or nodes.ndim != 3:
        raise ValueError("RL dataset nodes must have shape [B, N+1, F]")
    if nodes.shape[-1] != len(NODE_FEATURES):
        raise ValueError("RL dataset has the wrong feature dimension")
    return dataset


def problems_from_dataset(dataset: Dict[str, object]) -> List[Problem]:
    config = dataset["vehicle_config"]
    problems = []
    for instance in dataset["nodes"]:
        rows = instance.tolist()

        def request(row: Sequence[float], idx: int) -> Request:
            return Request(
                idx=idx,
                x=float(row[0]),
                y=float(row[1]),
                demand=float(row[2]),
                open=float(row[3]),
                close=float(row[4]),
                service_time=float(row[5]),
                time=float(row[7]),
                profit=float(row[8]),
                type=int(round(row[9])),
            )

        depot = request(rows[0], 0)
        customers = [request(row, idx) for idx, row in enumerate(rows[1:], 1)]
        problems.append(Problem(
            depot=depot,
            requests=customers,
            truck_speed=float(config["truck_speed"]),
            truck_capacity=float(config["truck_capacity"]),
            num_trucks=int(config["num_trucks"]),
        ))
    return problems


__all__ = [
    "DATASET_FORMAT",
    "NODE_FEATURES",
    "VEHICLE_CONFIGS",
    "discover_csvs",
    "fitness_value",
    "generate_from_reference",
    "infer_vehicle_config",
    "load_dataset",
    "problems_from_dataset",
    "save_dataset",
]
