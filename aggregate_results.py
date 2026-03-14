import json
import csv
import sys
import os

def aggregate(log_file, output_file):
    heuristics = {}
    gp_results = []
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                logger_name = data.get("__")
                event = data.get("_")
                
                if logger_name == "HEU" and event == "heuristic_result":
                    name = data.get("name")
                    res = data.get("result") # [dist, profit]
                    fit = data.get("fitness")
                    heuristics[name] = {
                        "Distance": res[0],
                        "Profit": res[1],
                        "Fitness": fit
                    }
                
                if logger_name == "GP" and event == "full_result":
                    res = data.get("result") # [dist, profit]
                    fit = data.get("fitness")
                    gp_results.append({
                        "Distance": res[0],
                        "Profit": res[1],
                        "Fitness": fit
                    })
            except:
                continue

    # Prepare summary
    summary = []
    for name, metrics in heuristics.items():
        summary.append({
            "Method": f"Heuristic: {name}",
            "Distance": metrics["Distance"],
            "Profit": metrics["Profit"],
            "Fitness": metrics["Fitness"]
        })
    
    if gp_results:
        # Get the best GP result (usually the last one or the one with best fitness)
        # Assuming higher fitness is better based on main.py logic (min fitness is best?)
        # Let's check fitness formula: distance/tot_dist * weight + ((max_profit - profit) / max_profit) * (1.0 - weight)
        # Lower fitness is better.
        best_gp = min(gp_results, key=lambda x: x["Fitness"])
        summary.append({
            "Method": "GP Best",
            "Distance": best_gp["Distance"],
            "Profit": best_gp["Profit"],
            "Fitness": best_gp["Fitness"]
        })

    # Write to CSV
    if summary:
        keys = summary[0].keys()
        with open(output_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary)
        print(f"Summary written to {output_file}")
    else:
        print("No results found to aggregate.")

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "result.log"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "summary_results.csv"
    aggregate(log_path, out_path)
