import pandas as pd
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Google Borg trace for CPU simulation")
    parser.add_argument("--input", type=str, default="data/google/raw_google_query.csv")
    parser.add_argument("--output", type=str, default="data/google/google_workload.csv")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-burst", type=int, default=100, help="Max synthetic burst time bounds")
    return parser.parse_args()

def preprocess():
    args = parse_args()
    np.random.seed(args.seed)
    
    df = pd.read_csv(args.input)
    initial_len = len(df)
    
    # 1. Derive Raw Execution Times (Microseconds -> Seconds)
    df['arrival_time_raw'] = df['submit_time'] / 1e6
    df['burst_time_raw'] = (df['finish_time'] - df['schedule_time']) / 1e6
    
    # 2. Filter Malformed Data
    df = df[df['burst_time_raw'] > 0]
    df = df.dropna(subset=['arrival_time_raw', 'burst_time_raw', 'priority'])
    
    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.seed)
        
    # 3. Preserve Temporal Order
    df = df.sort_values(by='arrival_time_raw').reset_index(drop=True)
    
    # 4. Normalize boundaries to match Synthetic state-space 
    min_arrival = df['arrival_time_raw'].min()
    df['arrival_time'] = df['arrival_time_raw'] - min_arrival
    
    # Min-max scaling for burst time to match synthetic RL bounds [1, max_burst]
    b_min, b_max = df['burst_time_raw'].min(), df['burst_time_raw'].max()
    df['burst_time'] = 1 + ((df['burst_time_raw'] - b_min) / (b_max - b_min)) * (args.max_burst - 1)
    
    # Scale arrival gaps to match synthetic workload density
    a_max = df['arrival_time'].max()
    target_arrival_max = args.sample_size * (args.max_burst * 0.1) 
    if a_max > 0:
        df['arrival_time'] = (df['arrival_time'] / a_max) * target_arrival_max
        
    df['arrival_time'] = np.floor(df['arrival_time']).astype(int)
    df['burst_time'] = np.ceil(df['burst_time']).astype(int)
    
    # Scale Priority from (0-360) -> (1-10)
    df['priority'] = np.ceil((df['priority'] / 360.0) * 10).astype(int)
    df['priority'] = df['priority'].clip(1, 10)
    
    df['pid'] = range(1, len(df) + 1)
    
    # 5. Methodological Validity Checks
    assert df['burst_time'].min() > 0, "Burst times must be positive"
    assert df['arrival_time'].min() >= 0, "Arrival time cannot be negative"
    assert df['pid'].is_unique, "PIDs must be unique"
    assert df['arrival_time'].is_monotonic_increasing, "Temporal order violated"
    
    # 6. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_cols = ['pid', 'arrival_time', 'burst_time', 'priority']
    df[final_cols].to_csv(args.output, index=False)
    print(f"Generated normalized dataset with {len(df)} tasks (Filtered {initial_len - len(df)}).")
    
if __name__ == "__main__":
    preprocess()