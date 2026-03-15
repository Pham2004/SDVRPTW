"""aggregate_results.py

Đọc file log (JSON-per-line) và tổng hợp kết quả theo từng instance riêng biệt.

Output CSV có các cột:
    Instance, Method, Distance, Profit, Fitness

Usage:
    python aggregate_results.py <log_file> <output_csv>
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

    # ── Parse log ─────────────────────────────────────────────────────────────
    # Theo dõi instance hiện tại đang được xử lý dựa trên event instance_start
    current_instance = "unknown"

    # { instance_name -> { heuristic_name -> {Distance, Profit, Fitness} } }
    heuristics_by_instance = defaultdict(dict)

    # { instance_name -> [ {Distance, Profit, Fitness} ] }  (mỗi gen 1 entry)
    gp_by_instance = defaultdict(list)

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

            # Cập nhật instance hiện tại khi gặp event instance_start
            if logger_name == "MAIN" and event == "instance_start":
                current_instance = data.get("instance", current_instance)
                continue

            if logger_name == "HEU" and event == "heuristic_result":
                name = data.get("name", "?")
                res  = data.get("result")   # [dist, profit]
                fit  = data.get("fitness")
                if res and len(res) >= 2:
                    heuristics_by_instance[current_instance][name] = {
                        "Distance": res[0],
                        "Profit":   res[1],
                        "Fitness":  fit,
                    }

            if logger_name == "GP" and event == "full_result":
                res = data.get("result")    # [dist, profit]
                fit = data.get("fitness")
                if res and len(res) >= 2:
                    gp_by_instance[current_instance].append({
                        "Distance": res[0],
                        "Profit":   res[1],
                        "Fitness":  fit,
                    })

    # ── Build summary rows ─────────────────────────────────────────────────────
    all_instances = sorted(
        set(heuristics_by_instance.keys()) | set(gp_by_instance.keys())
    )

    if not all_instances:
        print("No results found to aggregate.")
        return

    summary = []
    for inst in all_instances:
        # Heuristic rows
        for heu_name, metrics in sorted(heuristics_by_instance[inst].items()):
            summary.append({
                "Instance": inst,
                "Method":   f"Heuristic: {heu_name}",
                "Distance": metrics["Distance"],
                "Profit":   metrics["Profit"],
                "Fitness":  metrics["Fitness"],
            })

        # GP: lấy best (fitness thấp nhất = tốt nhất)
        gp_gens = gp_by_instance.get(inst, [])
        if gp_gens:
            best_gp = min(gp_gens, key=lambda x: x["Fitness"])
            last_gp = gp_gens[-1]
            summary.append({
                "Instance": inst,
                "Method":   "GP Best",
                "Distance": best_gp["Distance"],
                "Profit":   best_gp["Profit"],
                "Fitness":  best_gp["Fitness"],
            })
            summary.append({
                "Instance": inst,
                "Method":   "GP Last",
                "Distance": last_gp["Distance"],
                "Profit":   last_gp["Profit"],
                "Fitness":  last_gp["Fitness"],
            })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fieldnames = ["Instance", "Method", "Distance", "Profit", "Fitness"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"Summary written to '{output_file}'  ({len(summary)} rows, {len(all_instances)} instances)")

    # ── Print quick overview to stdout ─────────────────────────────────────────
    print()
    print(f"{'Instance':<15} {'Method':<25} {'Distance':>12} {'Profit':>12} {'Fitness':>10}")
    print("-" * 78)
    for row in summary:
        print(f"{row['Instance']:<15} {row['Method']:<25} "
              f"{row['Distance']:>12.4f} {row['Profit']:>12.4f} {row['Fitness']:>10.6f}")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "result.log"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "summary_results.csv"
    aggregate(log_path, out_path)