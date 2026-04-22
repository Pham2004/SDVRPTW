import os
import glob
import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform

def generate_datasets(input_dir, output_dir):
    # Get all csv files in the directory
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not files:
        print(f"No .csv files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Starting generation...")

    for file_path in files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # Read the original dataset
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        # Check if empty
        if df.empty:
            print(f"Skipping empty file {file_path}")
            continue

        # Coordinate bounds and demand bounds from the *entire* file (or just requests?)
        # Usually depot is included in bounds or it defines the center. 
        # We will use the min/max of the existing data to define the range.
        min_x = df['x'].min()
        max_x = df['x'].max()
        min_y = df['y'].min()
        max_y = df['y'].max()
        min_demand = df['demand'].min()
        max_demand = df['demand'].max()
        
        # Requests are assumed to be all rows. 
        # However, usually row 0 is depot. 
        # The user said "dataset", "requests". 
        # I will assume: Row 0 is depot and should remain FIXED (position/demand usually 0 for depot).
        # We only modify rows 1 to end.
        
        # Determine number of requests
        # If the file has N rows, we assume 1 depot + (N-1) requests.
        requests_indices = df.index[1:] # Skip first row (depot)
        if len(requests_indices) == 0:
            # Maybe no header? The view_file showed a header.
            # If only 1 row, nothing to generate.
            continue
            
        n_max = len(requests_indices)

        # Step 1: select n_static
        # interval [1/3 * n_max, 2/3 * n_max]
        lower_bound = int(n_max / 3)
        upper_bound = int(2 * n_max / 3)
        
        # Ensure bounds are valid
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound # Swap if weird
        if lower_bound == upper_bound:
                n_static = lower_bound
        else:
            n_static = random.randint(lower_bound, upper_bound)
        
        n_stoc = n_max - n_static
        
        # Step 2: Choose static/random set
        # "Select random n_static requests to be static"
        # We sample indices from the requests
        static_indices = np.random.choice(requests_indices, size=n_static, replace=False)
        # The rest are stochastic
        stochastic_indices = np.setdiff1d(requests_indices, static_indices)
        
        # Pre-calculate fixed profits for static requests and depot based on original dataset
        coords_orig = df[['x', 'y']].values
        dist_matrix_orig = squareform(pdist(coords_orig, metric='euclidean'))
        np.fill_diagonal(dist_matrix_orig, np.inf)
        k_values_orig = dist_matrix_orig.min(axis=1)
        
        fixed_profits = {}
        for positional_idx, idx in enumerate(df.index):
            if idx in static_indices or positional_idx == 0:
                k = k_values_orig[positional_idx]
                fixed_profits[idx] = random.uniform(0.7 * k, 1.5 * k)

        # Step 3: Generate 50 testcases... this loop IS generating one of the 50.
        # For stochastic indices, randomize x, y, demand
        for i in range(1, 17): # Generate 50 files
            new_df = df.copy()
            new_df.loc[stochastic_indices, 'x'] = np.random.uniform(min_x, max_x, size=len(stochastic_indices))
            new_df.loc[stochastic_indices, 'y'] = np.random.uniform(min_y, max_y, size=len(stochastic_indices))
            new_df.loc[stochastic_indices, 'demand'] = np.random.uniform(min_demand, max_demand, size=len(stochastic_indices))
            
            # Step 4: Add profit
            # k = distance to nearest request in the dataset
            # We compute distance matrix for ALL points in new_df (including depot)
            coords = new_df[['x', 'y']].values
            dist_matrix = squareform(pdist(coords, metric='euclidean'))
            
            # Fill diagonal with infinity so we don't pick self as nearest
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Nearest neighbor distance for each point
            k_values = dist_matrix.min(axis=1) # axis 1 = min over columns for each row
            
            profits = []
            for positional_idx, k in enumerate(k_values):
                idx = df.index[positional_idx]
                if idx in fixed_profits:
                    profits.append(fixed_profits[idx])
                else:
                    # Random in [0.7 * k, 1.5 * k]
                    p = random.uniform(0.7 * k, 1.5 * k)
                    profits.append(p)
            
            new_df['profit'] = profits
            new_df.loc[stochastic_indices, 'type'] = 1
            new_df.loc[static_indices, 'type'] = 0
            new_df.loc[0, 'type'] = 0
            
            # Save file
            output_filename = f"{base_name}_{i}.csv"
            output_path = os.path.join(output_dir, output_filename)
            new_df.to_csv(output_path, index=False)
            
        print(f"Generated 50 files for {filename}")

if __name__ == "__main__":
    # Assuming the script is run from the project root or we specify the path
    # The user is in f:\GP-DVRPTW-main\GP-DVRPTW-main
    target_dir = os.path.join(os.getcwd(), 'datasets', 'h200')
    output_dir = os.path.join(os.getcwd(), 'datasets', 'h200_new')
    if os.path.exists(target_dir):
        generate_datasets(target_dir, output_dir)
    else:
        # Fallback or try absolute path if cwd is wrong
        target_dir = r"f:\GP-DVRPTW-main\GP-DVRPTW-main\datasets\h200"
        if os.path.exists(target_dir):
            generate_datasets(target_dir, output_dir)
        else:
            print(f"Directory not found: {target_dir}")
