import pandas as pd
import glob
import os

def analyze():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(base_dir, 'datasets', '100', '*.csv')
    files = glob.glob(search_path)
    max_diffs = []
    drone_serve_counts = {0: 0, 1: 0}
    drone_serve_demand_max = {0: 0, 1: 0}
    
    for f in files:
        df = pd.read_csv(f)
        # Assuming row 0 is depot, maybe drone_serve=1
        if 'time' in df.columns and 'open' in df.columns:
            dynamic_nodes = df[df['time'] > 0]
            if not dynamic_nodes.empty:
                diffs = dynamic_nodes['open'] - dynamic_nodes['time']
                max_diff = diffs.max()
                max_diffs.append(max_diff)
            
        if 'drone_serve' in df.columns:
            for v in df['drone_serve'].unique():
                drone_serve_counts[v] += (df['drone_serve'] == v).sum()
                drone_serve_demand_max[v] = max(drone_serve_demand_max.get(v, 0), df[df['drone_serve'] == v]['demand'].max())

    print(f"Global max_diff: {max(max_diffs) if max_diffs else 0}")
    print(f"Average max_diff: {sum(max_diffs)/len(max_diffs) if max_diffs else 0}")
    print(f"Drone serve distribution: {drone_serve_counts}")
    print(f"Max demand for drone_serve: {drone_serve_demand_max}")

if __name__ == '__main__':
    analyze()
