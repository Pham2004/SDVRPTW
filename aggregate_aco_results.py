"""aggregate_aco_results.py

Đọc file log JSON-per-line từ aco_dvrptw.py và tổng hợp kết quả.

Log events được emit bởi aco_dvrptw.py:
    {"__": "MAIN", "_": "instance_start",    "instance": "..."}
    {"__": "ACO",  "_": "scenario_result",   "scenario": N, "result": [dist, profit], "fitness": F, "served": S}
    {"__": "ACO",  "_": "instance_summary",  "avg_final_fitness": F, "best_fitness_search": F, "evaluations": N, "elapsed_s": T}

Output CSV:
    Instance, AvgFitness, BestFitnessSearch, Evaluations, ElapsedS,
    Scen1_Fitness, ..., ScenN_Fitness

Usage:
    python aggregate_aco_results.py <log_file> <output_csv>
"""

import json
import csv
import sys
import os
from collections import defaultdict


def aggregate(log_file: str, output_file: str):
    if not os.path.exists(log_file):
        print(f"Log file '{log_file}' not found.")
        return

    current_instance = "unknown"

    # { instance -> { avg_final_fitness, best_fitness_search, evaluations, elapsed_s } }
    summaries: dict = {}

    # { instance -> [ {scenario, distance, profit, fitness, served} ] }
    scenarios_by_instance: dict = defaultdict(list)

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            logger_name = data.get("__")
            event       = data.get("_")

            if logger_name == "MAIN" and event == "instance_start":
                current_instance = data.get("instance", current_instance)
                continue

            if logger_name == "ACO" and event == "scenario_result":
                res = data.get("result")  # [dist, profit]
                scenarios_by_instance[current_instance].append({
                    "scenario": data.get("scenario"),
                    "distance": res[0] if res else None,
                    "profit":   res[1] if res else None,
                    "fitness":  data.get("fitness"),
                    "served":   data.get("served"),
                })

            if logger_name == "ACO" and event == "instance_summary":
                summaries[current_instance] = {
                    "avg_final_fitness":   data.get("avg_final_fitness"),
                    "best_fitness_search": data.get("best_fitness_search"),
                    "evaluations":         data.get("evaluations"),
                    "elapsed_s":           data.get("elapsed_s"),
                }

    all_instances = sorted(
        set(summaries.keys()) | set(scenarios_by_instance.keys())
    )

    if not all_instances:
        print("No ACO results found in log.")
        return

    # Determine max number of scenarios across all instances
    max_scen = max(
        (len(scenarios_by_instance[inst]) for inst in all_instances),
        default=0,
    )

    scen_cols = []
    for i in range(max_scen):
        scen_cols += [f"Scen{i+1}_Fitness", f"Scen{i+1}_Profit", f"Scen{i+1}_Distance"]

    fieldnames = [
        "Instance", "AvgFitness", "AvgProfit", "AvgDistance",
        "BestFitnessSearch", "Evaluations", "ElapsedS",
    ] + scen_cols

    rows = []
    for inst in all_instances:
        summ  = summaries.get(inst, {})
        scens = scenarios_by_instance.get(inst, [])
        scens_sorted = sorted(scens, key=lambda x: x.get("scenario", 0))

        profits   = [s["profit"]   for s in scens_sorted if s.get("profit")   is not None]
        distances = [s["distance"] for s in scens_sorted if s.get("distance") is not None]
        avg_profit   = round(sum(profits)   / len(profits),   6) if profits   else None
        avg_distance = round(sum(distances) / len(distances), 6) if distances else None

        row = {
            "Instance":           inst,
            "AvgFitness":         summ.get("avg_final_fitness"),
            "AvgProfit":          avg_profit,
            "AvgDistance":        avg_distance,
            "BestFitnessSearch":  summ.get("best_fitness_search"),
            "Evaluations":        summ.get("evaluations"),
            "ElapsedS":           summ.get("elapsed_s"),
        }
        for i, s in enumerate(scens_sorted):
            row[f"Scen{i+1}_Fitness"]  = s.get("fitness")
            row[f"Scen{i+1}_Profit"]   = s.get("profit")
            row[f"Scen{i+1}_Distance"] = s.get("distance")
        rows.append(row)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to '{output_file}'  ({len(rows)} instances)")
    print()
    print(f"{'Instance':<22} {'AvgFitness':>12} {'BestSearch':>12} "
          f"{'Evals':>8} {'Time(s)':>8}")
    print("-" * 68)
    for row in rows:
        avg = row["AvgFitness"]
        bst = row["BestFitnessSearch"]
        ev  = row["Evaluations"]
        t   = row["ElapsedS"]
        print(f"{row['Instance']:<22} "
              f"{avg:>12.5f} " if avg is not None else f"{'N/A':>12} ",
              end="")
        print(f"{bst:>12.5f} " if bst is not None else f"{'N/A':>12} ",
              end="")
        print(f"{ev:>8d} " if ev is not None else f"{'N/A':>8} ", end="")
        print(f"{t:>8.1f}" if t is not None else "N/A")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "aco_result.log"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "summary_aco.csv"
    aggregate(log_path, out_path)
