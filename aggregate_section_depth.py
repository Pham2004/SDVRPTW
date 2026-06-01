"""Aggregate logs for the tree-depth grid experiment.

Each log file is named:   result_<instance>_rd<routing_depth>_sd<seq_depth>.log
where rd = MAX_DEPTH_ROUTING and sd = MAX_DEPTH_SEQUENCING used for that run.

Per (instance, routing_depth, seq_depth) the script extracts, at the generation
with the best TEST fitness (lower is better):

    - TestFitness  : best `full_result` fitness (evaluated on all |S| scenarios)
    - TrainFitness : the `new_gen` fitness of that same generation (training set,
                     which under SCENARIO_TRAIN_RATIO=0.05 is a single scenario)
    - Gap          : TestFitness - TrainFitness  (overfitting signal; grows with
                     depth if a deeper tree memorises the tiny training set)
    - BestGen      : generation index where the best test fitness was reached
                     (convergence-speed signal)
    - R/S Size,Depth : realised node count and realised depth of the best routing
                     and sequencing trees, decoded from the logged base64. The
                     program is a binary-heap-laid-out array (children of i at
                     2i+1, 2i+2), null nodes are byte 255, so:
                         size  = #(byte != 255)
                         depth = max floor(log2(idx+1)) over non-null indices
                     These show whether the max-depth cap actually binds (bloat).

Output CSV columns:
    Instance, RoutingDepth, SeqDepth, TrainFitness, TestFitness, Gap,
    Distance, Profit, BestGen, RSize, RDepth, SSize, SDepth

Usage:
    python aggregate_section_depth.py <log_dir_or_glob> <output_csv>
"""
import base64
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

LOG_NAME = re.compile(
    r"^result_(?P<inst>.+)_rd(?P<rd>\d+)_sd(?P<sd>\d+)\.log$", re.IGNORECASE
)

NULL_BYTE = 255


def prog_size_depth(b64):
    """Decode a logged program base64 -> (realised_size, realised_depth).

    Returns (None, None) if the string cannot be decoded.
    """
    try:
        raw = list(base64.b64decode(b64))
    except Exception:
        return None, None
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    # run-length decode: pairs (byte, count) where real count = stored + 1
    nodes = []
    for i in range(0, len(raw), 2):
        nodes.extend([raw[i]] * (raw[i + 1] + 1))
    active = [i for i, b in enumerate(nodes) if b != NULL_BYTE]
    if not active:
        return 0, 0
    size = len(active)
    depth = max(int(math.log2(i + 1)) for i in active)
    return size, depth


def parse_log(path):
    """Walk a GP log and return the snapshot at the best (lowest) test fitness.

    The GP loop emits, per generation and in this order:
        new_gen     (__=GP, _=new_gen)     -> gen, training fitness
        full_result (__=GP, _=full_result) -> test fitness, distance, profit
        base64      (__=GP, _=base64)       -> routing / sequencing base64
    We pair them by sequence (full_result/base64 belong to the most recent
    new_gen) and keep the generation whose test fitness is lowest.
    """
    cur_gen = None
    cur_train = None
    best = None  # dict snapshot at best test fitness

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("__") != "GP":
                continue
            ev = d.get("_")

            if ev == "new_gen":
                cur_gen = d.get("gen")
                cur_train = d.get("fitness")

            elif ev == "full_result":
                res = d.get("result")
                fit = d.get("fitness")
                if res and len(res) >= 2 and fit is not None:
                    if best is None or fit < best["test_fit"]:
                        best = {
                            "gen": cur_gen,
                            "train_fit": cur_train,
                            "test_fit": fit,
                            "dist": res[0],
                            "profit": res[1],
                            "r_size": None, "r_depth": None,
                            "s_size": None, "s_depth": None,
                        }

            elif ev == "base64" and best is not None and best["gen"] == cur_gen:
                # base64 follows the full_result of the same generation; only
                # attach it when that generation is the current best snapshot.
                rs, rd = prog_size_depth(d.get("routing", ""))
                ss, sd = prog_size_depth(d.get("sequencing", ""))
                best["r_size"], best["r_depth"] = rs, rd
                best["s_size"], best["s_depth"] = ss, sd

    return best


def main():
    if len(sys.argv) < 3:
        print("usage: python aggregate_section_depth.py <log_dir_or_glob> <output_csv>")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2]

    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "result_*_rd*_sd*.log")))
    else:
        files = sorted(glob.glob(src))

    fields = ["Instance", "RoutingDepth", "SeqDepth", "TrainFitness",
              "TestFitness", "Gap", "Distance", "Profit", "BestGen",
              "RSize", "RDepth", "SSize", "SDepth"]
    rows = []
    for path in files:
        m = LOG_NAME.match(os.path.basename(path))
        if not m:
            continue
        rd = int(m.group("rd"))
        sd = int(m.group("sd"))
        b = parse_log(path)
        if b is None:
            print(f"[WARN] no GP result in {path}")
            continue
        gap = (b["test_fit"] - b["train_fit"]) if b["train_fit"] is not None else None
        rows.append({
            "Instance": m.group("inst"),
            "RoutingDepth": rd,
            "SeqDepth": sd,
            "TrainFitness": b["train_fit"],
            "TestFitness": b["test_fit"],
            "Gap": gap,
            "Distance": b["dist"],
            "Profit": b["profit"],
            "BestGen": b["gen"],
            "RSize": b["r_size"],
            "RDepth": b["r_depth"],
            "SSize": b["s_size"],
            "SDepth": b["s_depth"],
        })

    rows.sort(key=lambda r: (r["Instance"], r["RoutingDepth"], r["SeqDepth"]))

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Per-(routing, seq) cell averages across instances/seeds present in this run.
    cell = defaultdict(lambda: {"test": [], "gap": [], "rd": [], "sd": []})
    for r in rows:
        c = cell[(r["RoutingDepth"], r["SeqDepth"])]
        c["test"].append(r["TestFitness"])
        if r["Gap"] is not None:
            c["gap"].append(r["Gap"])
        if r["RDepth"] is not None:
            c["rd"].append(r["RDepth"])
        if r["SDepth"] is not None:
            c["sd"].append(r["SDepth"])

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\nWrote {len(rows)} rows to {out}")
    print(f"\n{'rDep':>4} {'sDep':>4} {'n':>3} {'avg_test':>10} {'avg_gap':>9} "
          f"{'realR':>6} {'realS':>6}")
    for (rd, sd) in sorted(cell):
        c = cell[(rd, sd)]
        print(f"{rd:>4} {sd:>4} {len(c['test']):>3} {avg(c['test']):>10.6f} "
              f"{avg(c['gap']):>9.6f} {avg(c['rd']):>6.2f} {avg(c['sd']):>6.2f}")


if __name__ == "__main__":
    main()
