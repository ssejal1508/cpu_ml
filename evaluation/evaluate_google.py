import argparse
import pandas as pd
import numpy as np
# ---------------------------------------------------------
# ADAPT THESE IMPORTS TO YOUR REPOSITORY'S ACTUAL STRUCTURE
from environment import load_workload 
from schedulers import FCFS, SJF, RoundRobin
# from agent import load_trained_rl_model
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../data/google/google_workload.csv")
    parser.add_argument("--runs", type=int, default=5, help="Number of evaluation seeds")
    return parser.parse_args()

def run_external_evaluation():
    args = parse_args()
    
    # Load the RL model (already trained on synthetic data)
    # model = load_trained_rl_model("models/best_synthetic_model.pth")
    
    metrics_log = { "FCFS": [], "SJF": [], "RR": [], "RL": [] }
    
    for seed in range(args.runs):
        print(f"--- Running Google Evaluation Seed {seed} ---")
        # In reality, you'd shuffle/sub-sample the Google trace per seed here
        workload = load_workload(dataset="google", google_csv=args.csv)
        
        # 1. FCFS
        fcfs = FCFS()
        metrics_log["FCFS"].append(fcfs.evaluate(workload.copy()))
        
        # 2. SJF
        sjf = SJF()
        metrics_log["SJF"].append(sjf.evaluate(workload.copy()))
        
        # 3. Round Robin
        rr = RoundRobin(quantum=5)
        metrics_log["RR"].append(rr.evaluate(workload.copy()))
        
        # 4. RL Agent
        # rl_metrics = evaluate_rl_agent(model, workload.copy())
        # metrics_log["RL"].append(rl_metrics)

    # Calculate and Print Comparison Table
    print("\n### Google ClusterData2019 Generalization Results ###")
    print("| Algorithm | Avg Waiting Time | Avg Turnaround Time | Avg Response Time | Throughput |")
    print("|-----------|------------------|---------------------|-------------------|------------|")
    
    for algo, results in metrics_log.items():
        if not results: continue
        # Assuming evaluate() returns a dict of metrics
        wait = [r['avg_waiting_time'] for r in results]
        turn = [r['avg_turnaround_time'] for r in results]
        resp = [r['avg_response_time'] for r in results]
        thru = [r['throughput'] for r in results]
        
        print(f"| {algo:<9} | {np.mean(wait):.2f} ± {np.std(wait):.2f} | "
              f"{np.mean(turn):.2f} ± {np.std(turn):.2f} | "
              f"{np.mean(resp):.2f} ± {np.std(resp):.2f} | "
              f"{np.mean(thru):.2f} ± {np.std(thru):.2f} |")

if __name__ == "__main__":
    run_external_evaluation()