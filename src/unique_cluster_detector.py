import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_production_cluster_strata import MultiFlightBioScapeDataset, GaiaTransferModel, Colors
try:
    from src.s3_utils import get_s3_fs, get_s3_bytes_pulled, get_worker_statuses
except ImportError:
    from s3_utils import get_s3_fs, get_s3_bytes_pulled, get_worker_statuses

def run_single_session(run_idx, train_idx, val_idx, full_dataset, config, device, args):
    """Trains a single model on the given split. Returns the best validation metrics."""
    b_cfg = config['bioscape']
    batch_size = args.batch_size or b_cfg['batch_size']
    epochs = args.epochs or 40 # Default to 40 epochs for detector to save time
    lr = b_cfg['learning_rate']
    patch_size = b_cfg.get('patch_size', 16)

    # -- Target normalization (train-only) --
    train_richness = np.array([full_dataset.mappings[i][3] for i in train_idx])
    richness_mean = float(train_richness.mean())
    richness_std = float(train_richness.std()) + 1e-6

    # -- DataLoaders --
    num_workers = b_cfg.get('num_workers', 4)
    if args.mosaic:
        nc_dir = args.nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else "data/bioscape/30m_v2")
    else:
        nc_dir = args.nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else b_cfg['nc_dir_local'])

    if os.name == 'nt' and not nc_dir.startswith("s3"):
        num_workers = 0

    train_loader = DataLoader(
        Subset(full_dataset, train_idx.tolist()),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx.tolist()),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # -- Model --
    model = GaiaTransferModel(num_targets=1, patch_size=patch_size).to(device)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    foundation_ckpt = os.path.join(
        project_root, "checkpoints", "pretrained_ViTSpatialSpectral_200ep_enmap.pth"
    )
    if os.path.exists(foundation_ckpt):
        model.load_foundation_weights(foundation_ckpt, device)

    # -- Freeze encoder --
    if args.freeze:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.mlp_head.parameters():
            param.requires_grad = True

    # -- Optimizer / Scheduler --
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    best_r2 = -float('inf')
    best_rmse = float('inf')
    best_mae = float('inf')
    patience_counter = 0
    patience = 15 # Shorter patience for detector runs

    for epoch in range(1, epochs + 1):
        if args.freeze and args.unfreeze_epoch and epoch == args.unfreeze_epoch:
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.encoder.mlp_head.parameters(), 'lr': lr},
                {'params': [p for n, p in model.encoder.named_parameters()
                            if 'mlp_head' not in n], 'lr': lr * 0.3},
            ], weight_decay=0.05)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs - epoch + 1,
                steps_per_epoch=len(train_loader), pct_start=0.05
            )

        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels_norm = ((labels - richness_mean) / richness_std).to(device).float()
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                out_norm = model(images.to(device))
                out_real = (out_norm.cpu().numpy().flatten()
                            * richness_std + richness_mean)
                val_preds.extend(out_real)
                val_targets.extend(labels.numpy().flatten())

        r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0
        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        mae = mean_absolute_error(val_targets, val_preds)

        if r2 > best_r2:
            best_r2 = r2
            best_rmse = rmse
            best_mae = mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_r2, best_rmse, best_mae

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Gaia Unique Cluster Influence Detector"
    )
    parser.add_argument("-g", "--iterations", type=int, default=15,
                        help="Number of training sessions to run (default: 15)")
    parser.add_argument("-n", "--folds", type=int, default=6,
                        help="Number of folds/groups to split clusters into (default: 6)")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of epochs per training session (default: 40)")
    parser.add_argument("--batch_size", type=int,
                        help="Override batch size")
    parser.add_argument("--nc_dir", type=str,
                        help="Override data directory")
    parser.add_argument("--freeze", action="store_true", default=True,
                        help="Freeze encoder, train only head (default)")
    parser.add_argument("--no-freeze", dest="freeze", action="store_false",
                        help="Fine-tune all parameters")
    parser.add_argument("--unfreeze-epoch", type=int, default=None,
                        help="Unfreeze encoder at this epoch")
    parser.add_argument("--mosaic", action="store_true", help="Use Level 3 Mosaic tiles")
    parser.add_argument("--no-mask", dest="mask", action="store_false", help="Disable water vapor masking")
    parser.set_defaults(mask=True)
    parser.add_argument("--test-run", action="store_true",
                        help="Quick smoke test (2 iterations, 3 folds, 2 epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    # -- Load Config --
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    b_cfg = config['bioscape']
    richness_csv = os.path.join(project_root, b_cfg['richness_csv'])
    patch_size = b_cfg.get('patch_size', 16)
    device = torch.device(
        config.get('device', 'cuda') if torch.cuda.is_available() else "cpu"
    )

    if args.mosaic:
        nc_dir = args.nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else "data/bioscape/30m_v2")
        print(f"{Colors.WARNING}[*] MOSAIC MODE: Using Level 3 Tiles from {nc_dir}{Colors.ENDC}")
    else:
        nc_dir = args.nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else b_cfg['nc_dir_local'])

    if args.test_run:
        print(f"{Colors.FAIL}[!] TEST RUN: 2 iterations, 3 folds, 2 epochs{Colors.ENDC}")
        args.iterations = 2
        args.folds = 3
        args.epochs = 2

    # -- Discover NetCDFs --
    if nc_dir.startswith("s3://"):
        fs = get_s3_fs()
        nc_paths = ["s3://" + f for f in fs.ls(nc_dir) if f.endswith('.nc')]
    else:
        nc_paths = sorted([
            os.path.join(nc_dir, f)
            for f in os.listdir(nc_dir) if f.endswith('.nc')
        ])

    if not nc_paths:
        print(f"{Colors.FAIL}[!] No NetCDF files found in {nc_dir}{Colors.ENDC}")
        return

    if args.test_run:
        nc_paths = nc_paths[:2]

    if args.mosaic and not nc_dir.startswith("s3"):
        # Filter by tile_names.json if in mosaic mode
        tile_path = os.path.join(project_root, "tile_names.json")
        if os.path.exists(tile_path):
            import json
            try:
                with open(tile_path, 'r', encoding='utf-8') as f:
                    tiles = {t['tile'] for t in json.load(f)['tiles']}
            except (UnicodeDecodeError, json.JSONDecodeError):
                with open(tile_path, 'r', encoding='utf-16') as f:
                    tiles = {t['tile'] for t in json.load(f)['tiles']}
            
            nc_paths = [p for p in nc_paths if any(t in p for t in tiles)]

    # -- Build Dataset --
    cache_suffix = f"_p{patch_size}_strata.json"
    if args.mosaic: cache_suffix = "_mosaic" + cache_suffix
    cache_name = ("s3_mapping" if nc_dir.startswith("s3")
                  else "local_mapping") + cache_suffix
    mapping_cache = os.path.join(project_root, "data", "bioscape", cache_name)

    full_dataset = MultiFlightBioScapeDataset(
        nc_paths, richness_csv, patch_size=patch_size,
        augment=True, cache_path=mapping_cache, mask_water_vapor=args.mask
    )
    if len(full_dataset) == 0:
        print(f"{Colors.FAIL}[!] No training samples found. Stopping.{Colors.ENDC}")
        return

    # Compute band stats (cached)
    band_stats_name = ("s3_band_stats" if nc_dir.startswith("s3")
                       else "local_band_stats") + cache_suffix
    band_stats_cache = os.path.join(project_root, "data", "bioscape", band_stats_name)
    full_dataset.compute_band_stats(cache_path=band_stats_cache)

    # Clear h5py handles before forking workers
    for ds in full_dataset.dataset_cache.values():
        ds.close()
    full_dataset.dataset_cache.clear()

    # -- Discover Clusters --
    mappings = full_dataset.mappings
    sample_clusters = np.array([m[4] for m in mappings])
    unique_clusters = np.unique(sample_clusters)
    val_candidates = unique_clusters[unique_clusters != -1]

    if args.folds > len(val_candidates):
        args.folds = len(val_candidates)

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  GAIA UNIQUE CLUSTER INFLUENCE DETECTOR{Colors.ENDC}")
    print(f"  Running {args.iterations} iterations, holdout ratio 1/{args.folds} clusters")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    run_records = []
    
    # Track statistics for each cluster
    val_performances = {c: [] for c in val_candidates}
    train_performances = {c: [] for c in val_candidates}

    np.random.seed(args.seed)

    for i in range(1, args.iterations + 1):
        print(f"\n{Colors.OKBLUE}[*] Iteration {i}/{args.iterations} (Running training...){Colors.ENDC}")
        
        # Partition clusters randomly into N folds
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed + i)
        splits = list(kf.split(val_candidates))
        
        # Randomly choose one split to be the validation set
        chosen_split_idx = np.random.randint(0, args.folds)
        train_clusters_idx, val_clusters_idx = splits[chosen_split_idx]
        
        val_clusters = val_candidates[val_clusters_idx]
        train_clusters = val_candidates[train_clusters_idx]

        train_idx = np.where(~np.isin(sample_clusters, val_clusters))[0]
        val_idx = np.where(np.isin(sample_clusters, val_clusters))[0]

        t_start = time.time()
        r2, rmse, mae = run_single_session(i, train_idx, val_idx, full_dataset, config, device, args)
        elapsed = time.time() - t_start

        print(f"{Colors.OKGREEN}[OK] Iteration {i} Done in {elapsed/60:.1f}m - Val R2: {r2:.4f} | RMSE: {rmse:.2f}{Colors.ENDC}")

        # Record run
        run_records.append({
            'iteration': i,
            'val_clusters': ",".join(map(str, sorted(list(val_clusters)))),
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        })

        # Track cluster performance
        for c in val_candidates:
            if c in val_clusters:
                val_performances[c].append(r2)
            else:
                train_performances[c].append(r2)

    # -- Aggregate Cluster Performance Statistics --
    cluster_stats = []
    for c in val_candidates:
        v_scores = val_performances[c]
        t_scores = train_performances[c]
        
        mean_r2_val = np.mean(v_scores) if v_scores else np.nan
        mean_r2_train = np.mean(t_scores) if t_scores else np.nan
        diff_r2 = mean_r2_val - mean_r2_train if (v_scores and t_scores) else np.nan
        
        cluster_stats.append({
            'cluster_id': int(c),
            'times_in_val': len(v_scores),
            'times_in_train': len(t_scores),
            'mean_r2_when_val': round(mean_r2_val, 4) if not np.isnan(mean_r2_val) else np.nan,
            'mean_r2_when_train': round(mean_r2_train, 4) if not np.isnan(mean_r2_train) else np.nan,
            'influence_diff': round(diff_r2, 4) if not np.isnan(diff_r2) else np.nan
        })

    df_runs = pd.DataFrame(run_records)
    df_stats = pd.DataFrame(cluster_stats)

    # Save reports
    runs_path = os.path.join(project_root, "reports", "cluster_detector_runs.csv")
    stats_path = os.path.join(project_root, "reports", "cluster_influence_stats.csv")
    os.makedirs(os.path.dirname(runs_path), exist_ok=True)
    df_runs.to_csv(runs_path, index=False)
    df_stats.to_csv(stats_path, index=False)

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  CLUSTER INFLUENCE DIAGNOSTIC REPORT{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    # 1. Hardest Clusters to Predict (Lowest R2 when in Validation)
    print(f"\n{Colors.WARNING}  Top 5 Hardest Clusters to Predict (Lowest R2 when held-out in Validation):{Colors.ENDC}")
    hardest = df_stats.dropna(subset=['mean_r2_when_val']).sort_values('mean_r2_when_val').head(5)
    print(hardest[['cluster_id', 'times_in_val', 'mean_r2_when_val']].to_string(index=False))

    # 2. Most Critical Clusters for Training (Highest R2 when in Training)
    print(f"\n{Colors.OKGREEN}  Top 5 Most Critical Clusters for Training (Highest overall R2 when in Training set):{Colors.ENDC}")
    critical = df_stats.dropna(subset=['mean_r2_when_train']).sort_values('mean_r2_when_train', ascending=False).head(5)
    print(critical[['cluster_id', 'times_in_train', 'mean_r2_when_train']].to_string(index=False))

    # 3. Overall Diagnostic Stats
    print(f"\n{Colors.OKCYAN}{'-'*60}{Colors.ENDC}")
    print(f"  All runs log saved to: {runs_path}")
    print(f"  Cluster influence stats saved to: {stats_path}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

if __name__ == "__main__":
    main()
