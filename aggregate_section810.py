"""Aggregate logs for section-8.10 experiment.

Each log file is named:    result_<instance>_S<num_scenarios>.log
The script parses every JSON-per-line log under a directory (or list), extracts
the best GP fitness across generations per (instance, S), and writes a CSV
with columns: Instance, S, Distance, Profit, Fitness, BestGen.

Usage:
    python aggregate_section810.py <log_dir_or_glob> <output_csv>
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

LOG_NAME = re.compile(r"^result_(?P<inst>.+)_S(?P<s>\d+)\.log$", re.IGNORECASE)


def parse_log(path):
    """Return (best_dist, best_profit, best_fit, best_gen) from a GP log."""
    best = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("__") == "GP" and d.get("_") == "full_result":
                res = d.get("result")
                fit = d.get("fitness")
                gen = d.get("gen")
                if res and len(res) >= 2 and fit is not None:
                    if best is None or fit < best[2]:
                        best = (res[0], res[1], fit, gen)
    return best


def main():
    if len(sys.argv) < 3:
        print("usage: python aggregate_section810.py <log_dir_or_glob> <output_csv>")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2]

    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "result_*_S*.log")))
    else:
        files = sorted(glob.glob(src))

    rows = []
    for path in files:
        m = LOG_NAME.match(os.path.basename(path))
        if not m:
            continue
        inst = m.group("inst")
        s = int(m.group("s"))
        best = parse_log(path)
        if best is None:
            print(f"[WARN] no GP result in {path}")
            continue
        rows.append({
            "Instance": inst,
            "S": s,
            "Distance": best[0],
            "Profit": best[1],
            "Fitness": best[2],
            "BestGen": best[3],
        })

    rows.sort(key=lambda r: (r["Instance"], r["S"]))

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Instance", "S", "Distance", "Profit", "Fitness", "BestGen"])
        w.writeheader()
        w.writerows(rows)

    # Per-S average across instances (column Avg_Fitness vs S)
    by_s = defaultdict(list)
    for r in rows:
        by_s[r["S"]].append(r["Fitness"])
    print(f"\nWrote {len(rows)} rows to {out}")
    print(f"\n{'S':>4} {'n_inst':>7} {'avg_fitness':>14}")
    for s in sorted(by_s):
        vs = by_s[s]
        print(f"{s:>4} {len(vs):>7} {sum(vs)/len(vs):>14.6f}")


if __name__ == "__main__":
    main()
