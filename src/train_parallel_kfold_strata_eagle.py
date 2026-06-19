import os
import sys
import time
import subprocess
import json
import argparse
import pandas as pd
import numpy as np

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():
    parser = argparse.ArgumentParser(
        description="Parallel Gaia EAGLE Cluster-Stratified K-Fold Runner across GPUs"
    )
    parser.add_argument("--k", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--mode", type=str, choices=["native", "10nm"], default="native",
                        help="EAGLE data mode (native or 10nm)")
    parser.add_argument("--unfreeze-epoch", type=int,
                        help="Unfreeze encoder at this epoch")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience")
    parser.add_argument("--epochs", type=int,
                        help="Override epochs per fold")
    parser.add_argument("--batch-size", type=int,
                        help="Override batch size")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated list of GPU device IDs to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splits")
    parser.add_argument("--test-run", action="store_true",
                        help="Quick test run forwarding to scripts")
    args = parser.parse_args()

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        print(f"{Colors.FAIL}[!] No GPU IDs specified in --gpus{Colors.ENDC}")
        sys.exit(1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(project_root, "src", "train_kfold_cluster_strata_eagle.py")

    num_folds = 2 if args.test_run else args.k
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  PARALLEL EAGLE STRATIFIED {num_folds}-FOLD CROSS-VALIDATION ({args.mode} mode){Colors.ENDC}")
    print(f"{Colors.HEADER}  Available GPUs: {', '.join(gpu_list)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    # Build queue of fold indices to run
    folds_to_run = list(range(num_folds))
    active_processes = {}  # maps process -> (fold_idx, gpu_id, log_file, log_file_path)
    available_gpus = list(gpu_list)
    fold_results = {}

    t0 = time.time()

    # Loop until all folds have run and completed
    while folds_to_run or active_processes:
        # Launch new processes on available GPUs
        while folds_to_run and available_gpus:
            fold_idx = folds_to_run.pop(0)
            gpu_id = available_gpus.pop(0)

            # Build command
            cmd = [
                sys.executable, script_path,
                "--mode", args.mode,
                "--k", str(args.k),
                "--seed", str(args.seed),
                "--fold", str(fold_idx)
            ]
            if args.unfreeze_epoch is not None:
                cmd.extend(["--unfreeze-epoch", str(args.unfreeze_epoch)])
            if args.patience is not None:
                cmd.extend(["--patience", str(args.patience)])
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.test_run:
                cmd.append("--test-run")

            # Environment variables setup
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            print(f"{Colors.OKBLUE}[*] Launching Fold {fold_idx+1}/{num_folds} on GPU {gpu_id}...{Colors.ENDC}")
            
            # Start process and direct logs to separate file
            log_dir = os.path.join(project_root, "reports", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, f"fold_{args.mode}_{fold_idx}.log")
            log_file = open(log_file_path, "w", encoding="utf-8")
            
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            active_processes[proc] = {
                'fold_idx': fold_idx,
                'gpu_id': gpu_id,
                'log_file': log_file,
                'log_file_path': log_file_path,
                'offset': 0
            }

        # Check for completed processes and stream log updates
        finished_procs = []
        for proc, info in active_processes.items():
            fold_idx = info['fold_idx']
            gpu_id = info['gpu_id']
            log_file_path = info['log_file_path']
            log_file = info['log_file']

            # Read new log lines
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(info['offset'])
                        new_content = f.read()
                        info['offset'] = f.tell()
                        
                        if new_content:
                            for line in new_content.splitlines():
                                line_str = line.strip()
                                if not line_str:
                                    continue
                                # Filter out tqdm progress bars to keep output clean
                                if "%|" in line_str or "it/s" in line_str:
                                    continue
                                # Print update with a distinct fold & GPU tag
                                print(f"{Colors.OKCYAN}[Fold {fold_idx+1:2d} (GPU {gpu_id})]{Colors.ENDC} {line}")
                except Exception:
                    pass

            ret = proc.poll()
            if ret is not None:
                finished_procs.append(proc)
                log_file.close()
                if ret == 0:
                    print(f"{Colors.OKGREEN}[OK] Fold {fold_idx+1} completed successfully on GPU {gpu_id}.{Colors.ENDC}")
                    # Load the results from the temporary JSON file
                    temp_path = os.path.join(project_root, "reports", f"temp_fold_{args.mode}_{fold_idx}.json")
                    if os.path.exists(temp_path):
                        with open(temp_path, "r") as f:
                            fold_results[fold_idx] = json.load(f)
                        try:
                            os.remove(temp_path)  # clean up
                        except OSError:
                            pass
                    else:
                        print(f"{Colors.WARNING}[!] Warning: Temp results file not found for fold {fold_idx+1}: {temp_path}{Colors.ENDC}")
                else:
                    print(f"{Colors.FAIL}[!] ERROR: Fold {fold_idx+1} failed with exit code {ret} on GPU {gpu_id}. Check log: {log_file_path}{Colors.ENDC}")

        # Clean up finished processes
        for proc in finished_procs:
            gpu_id = active_processes[proc]['gpu_id']
            del active_processes[proc]
            available_gpus.append(gpu_id)

        # Sleep briefly to avoid busy waiting
        time.sleep(1.0)

    elapsed = time.time() - t0

    # Aggregate and show summary if we have results
    if not fold_results:
        print(f"{Colors.FAIL}[!] No folds completed successfully.{Colors.ENDC}")
        sys.exit(1)

    results_list = [fold_results[i] for i in sorted(fold_results.keys()) if i in fold_results]
    df = pd.DataFrame(results_list)
    mean_r2 = df['r2'].mean()
    std_r2 = df['r2'].std() if len(df) > 1 else 0
    mean_rmse = df['rmse'].mean()
    std_rmse = df['rmse'].std() if len(df) > 1 else 0
    mean_mae = df['mae'].mean()
    std_mae = df['mae'].std() if len(df) > 1 else 0

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"  PARALLEL CLUSTER-STRATIFIED CROSS-VALIDATION SUMMARY (EAGLE {args.mode}){Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(df.to_string(index=False))
    print(f"{Colors.OKCYAN}{'-'*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean R2:   {mean_r2:.4f} +/- {std_r2:.4f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean MAE:  {mean_mae:.2f} +/- {std_mae:.2f}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  Total Time: {elapsed/60:.1f} minutes{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    # Save final report
    report_path = os.path.join(project_root, "reports", f"kfold_cluster_strata_results_eagle_{args.mode}.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"{Colors.OKGREEN}[OK] Aggregated results saved to: {report_path}{Colors.ENDC}")

if __name__ == "__main__":
    main()
