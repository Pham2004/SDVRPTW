import os
import glob
import pandas as pd
import numpy as np

def parse_solomon_200(filepath):
    # Read the file line by line
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract the name
    name = lines[0].strip().lower() # e.g., c1_2_1
    
    # Extract customer data starting from the line after "CUST NO. ..."
    data_lines = []
    start_parsing = False
    for line in lines:
        if "CUST NO." in line:
            start_parsing = True
            continue
        if start_parsing:
            parts = line.strip().split()
            if len(parts) >= 7:
                data_lines.append(parts)
            if len(data_lines) == 401: # Depot + 400 customers
                break
                
    # Create DataFrame
    df = pd.DataFrame(data_lines, columns=['CUST_NO', 'XCOORD', 'YCOORD', 'DEMAND', 'READY_TIME', 'DUE_DATE', 'SERVICE_TIME'])
    df = df.astype(float)
    
    # Rename columns to match target format: x,y,demand,open,close,servicetime
    df = df.rename(columns={
        'XCOORD': 'x',
        'YCOORD': 'y',
        'DEMAND': 'demand',
        'READY_TIME': 'open',
        'DUE_DATE': 'close',
        'SERVICE_TIME': 'servicetime'
    })
    
    # Drop CUST_NO
    df = df.drop(columns=['CUST_NO'])
    
    return name, df

def generate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'datasets', '400')
    output_dir = os.path.join(base_dir, 'datasets', 'h400')
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, '*.TXT'))
    
    for filepath in files:
        name, df = parse_solomon_200(filepath)
        
        # Determine the target filename
        # name is something like 'c1_2_1' or 'rc2_2_10'
        # The user's naming convention is h100[distribution][type][index] -> h100c101
        # 'c1_2_1' -> c, 1, 01 => c101
        # Wait, the dataset is Solomon 200. So `c1_2_1` means type C, class 1, subset 2, id 1? 
        # Actually Solomon 200 files are named C1_2_1.TXT, C2_2_1.TXT.
        # Let's map e.g. 'c1_2_1' -> 'c1', '01'.
        parts = name.split('_')
        prefix = parts[0] # 'c1', 'rc2', etc.
        index_str = f"{int(parts[2]):02d}" # '01', '10'
        
        target_filename = f"h400{prefix}{index_str}.csv"
        
        # Generation Logic
        # 1. drone_serve: 1 if demand <= 10 else 0
        df['drone_serve'] = df['demand'].apply(lambda d: 1 if d <= 10.0 else 0)
        # Force depot to have drone_serve = 1
        df.loc[0, 'drone_serve'] = 1
        
        # 2. arrival_time (time)
        np.random.seed(42 + int(parts[2])) # Reproducible but varied
        
        times = []
        for i, row in df.iterrows():
            if i == 0:
                times.append(0.0) # Depot
            else:
                ready_time = row['open']
                lower_bound = max(0.0, ready_time - 180)
                t = np.random.uniform(lower_bound, ready_time)
                times.append(round(t, 2))
                    
        df['time'] = times
        
        # 3. Sort chronologically by time (keep depot at row 0)
        depot = df.iloc[[0]]
        customers = df.iloc[1:].copy()
        customers = customers.sort_values(by='time').reset_index(drop=True)
        
        # 4. Concatenate and save
        final_df = pd.concat([depot, customers], ignore_index=True)
        
        # Ensure column order
        final_df = final_df[['x', 'y', 'demand', 'open', 'close', 'servicetime', 'drone_serve', 'time']]
        
        out_path = os.path.join(output_dir, target_filename)
        final_df.to_csv(out_path, index=False)
        print(f"Generated {out_path}")

if __name__ == '__main__':
    generate()
