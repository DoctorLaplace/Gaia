import os
import sys
import time
import copy
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
import matplotlib
matplotlib.use("Agg")  # headless-safe (nrp-nautilus JupyterHub has no display)
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eagle_dataset import MultiFlightEagleDataset, SingleEagleTiffDataset
from src.train_production_cluster_strata_eagle import GaiaTransferModel, Colors

def run_fold(fold_idx, k, train_idx, val_idx, val_clusters, full_dataset, config, device, args):
    """Train and evaluate a single fold. Returns a metrics dict."""
    b_cfg = config['bioscape']
    batch_size = args.batch_size or b_cfg['batch_size']
    epochs = args.epochs or b_cfg['epochs']
    lr = b_cfg['learning_rate']
    patch_size = b_cfg.get('patch_size', 16)

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  FOLD {fold_idx+1} / {k}  "
          f"(Train: {len(train_idx)} | Val: {len(val_idx)}){Colors.ENDC}")
    print(f"{Colors.HEADER}  Held out clusters: {sorted(list(val_clusters))}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    # -- Target normalization (train-only to prevent leakage) --
    train_richness = np.array([full_dataset.mappings[i][3] for i in train_idx])
    richness_mean = float(train_richness.mean())
    richness_std = float(train_richness.std()) + 1e-6
    print(f"{Colors.OKBLUE}[*] Fold {fold_idx+1} Richness: "
          f"mean={richness_mean:.1f}, std={richness_std:.1f}{Colors.ENDC}")

    # -- DataLoaders --
    # Windows doesn't handle multi-processing for rasterio cleanly, default to 0 on Windows
    num_workers = 0 if os.name == 'nt' else b_cfg.get('num_workers', 4)

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
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{Colors.OKBLUE}[*] Encoder FROZEN: "
              f"{trainable}/{total} params trainable "
              f"({trainable/total*100:.1f}%){Colors.ENDC}")

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
    best_metrics = {}
    best_state_dict = None
    patience_counter = 0
    patience = args.patience

    for epoch in range(1, epochs + 1):
        # -- Progressive unfreezing --
        if args.freeze and args.unfreeze_epoch and epoch == args.unfreeze_epoch:
            print(f"\n{Colors.WARNING}[*] Epoch {epoch}: "
                  f"UNFREEZING encoder for fine-tuning{Colors.ENDC}")
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

        # -- Train --
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader,
                    desc=f"F{fold_idx+1} Ep {epoch}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels_norm = ((labels - richness_mean) / richness_std).to(device).float()
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # -- Validate --
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
        avg_loss = train_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {Colors.OKCYAN}[Epoch {epoch:3d}] "
                  f"Loss: {avg_loss:.4f} | R2: {r2:7.4f} | "
                  f"RMSE: {rmse:.2f} | Best: {best_r2:7.4f}{Colors.ENDC}")

        if r2 > best_r2:
            best_r2 = r2
            best_metrics = {
                'fold': fold_idx + 1,
                'held_out_clusters': ",".join(map(str, sorted(list(val_clusters)))),
                'r2': round(r2, 4),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'best_epoch': epoch
            }
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {Colors.WARNING}[*] Early stopping at epoch {epoch} "
                      f"(best R2: {best_r2:.4f}){Colors.ENDC}")
                break

    print(f"{Colors.OKGREEN}[OK] Fold {fold_idx+1} Complete. "
          f"Best R2: {best_r2:.4f}{Colors.ENDC}")

    # -- Use the best-epoch weights (not just the last epoch) for predictions/plot --
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # -- Dense inference over the ENTIRE dataset (train + held-out) with this fold's model --
    model.eval()
    was_augmenting = full_dataset.augment
    full_dataset.augment = False  # deterministic predictions, no random rot/flip
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    all_preds = []
    with torch.no_grad():
        for images, labels in full_loader:
            out_norm = model(images.to(device))
            all_preds.extend(out_norm.cpu().numpy().flatten() * richness_std + richness_mean)
    full_dataset.augment = was_augmenting
    all_preds = np.array(all_preds)

    # -- Scatter plot: held-out (scored) vs train (context only, not scored) --
    actual = np.array([m[3] for m in full_dataset.mappings])
    held_out_mask = np.isin(np.array([m[4] for m in full_dataset.mappings]), val_clusters)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual[~held_out_mask], all_preds[~held_out_mask],
               color='gray', alpha=0.25, s=8, label='train (not scored)')
    ax.scatter(actual[held_out_mask], all_preds[held_out_mask],
               color='tab:blue', alpha=0.8, s=25, label='held-out (scored)')
    lims = [actual.min() - 2, actual.max() + 2]
    ax.plot(lims, lims, '--k', alpha=0.3, label='1:1')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual richness")
    ax.set_ylabel("Predicted richness")
    ax.set_title(f"Fold {fold_idx+1} — held-out clusters {sorted(list(val_clusters))}\n"
                 f"R² (held-out only) = {best_r2:.4f}", fontsize='small')
    ax.legend(fontsize='x-small', loc='upper left')
    fig.tight_layout()

    plots_dir = os.path.join(project_root, "reports", "kfold_plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"fold_{fold_idx+1}_scatter.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"{Colors.OKGREEN}[OK] Saved fold {fold_idx+1} scatter plot to: {plot_path}{Colors.ENDC}")

    return best_metrics, all_preds

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Gaia EAGLE Cluster-Stratified K-Fold Cross-Validation"
    )
    parser.add_argument("--k", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--epochs", type=int,
                        help="Override epochs per fold")
    parser.add_argument("--batch_size", type=int,
                        help="Override batch size")
    parser.add_argument("--tif_dir", type=str,
                        help="Override data directory")
    parser.add_argument("--freeze", action="store_true", default=True,
                        help="Freeze encoder, train only head (default)")
    parser.add_argument("--no-freeze", dest="freeze", action="store_false",
                        help="Fine-tune all parameters")
    parser.add_argument("--unfreeze-epoch", type=int, default=None,
                        help="Unfreeze encoder at this epoch")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (default: 25)")
    parser.add_argument("--mode", type=str, choices=["native", "10nm"], default="native",
                        help="Select EAGLE data mode: native (32 target tiles) or 10nm (1517 mosaic tiles)")
    parser.add_argument("--no-mask", dest="mask", action="store_false", help="Disable water vapor masking")
    parser.set_defaults(mask=True)
    parser.add_argument("--test-run", action="store_true",
                        help="Quick smoke test (2 folds, 2 epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for K-Fold splitting")
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

    # Select directory based on mode if not overridden
    tif_dir = args.tif_dir
    if not tif_dir:
        if args.mode == "native":
            tif_dir = os.path.join(project_root, "data", "eagle", "30m_native")
        else:
            tif_dir = os.path.join(project_root, "data", "eagle", "30m_10nm")

    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  GAIA EAGLE CLUSTER-STRATIFIED {args.k}-FOLD CROSS-VALIDATION ({args.mode} mode){Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    if args.test_run:
        print(f"{Colors.FAIL}[!] TEST RUN: 2 folds, 2 epochs{Colors.ENDC}")
        args.k = 2
        args.epochs = 2

    if not os.path.exists(tif_dir):
        print(f"{Colors.FAIL}[!] Directory not found: {tif_dir}. Run extract_eagle_data.py first.{Colors.ENDC}")
        return

    # Discover TIFFs
    tif_paths = sorted([
        os.path.join(tif_dir, f)
        for f in os.listdir(tif_dir) if f.endswith('.tif')
    ])

    if not tif_paths:
        print(f"{Colors.FAIL}[!] No .tif files found in {tif_dir}{Colors.ENDC}")
        return

    if args.test_run:
        tif_paths = tif_paths[:2]

    # -- Build Dataset --
    cache_suffix = f"_p{patch_size}_{args.mode}_strata.json"
    if args.test_run:
        cache_suffix = f"_p{patch_size}_{args.mode}_strata_test.json"
    mapping_cache = os.path.join(project_root, "data", "eagle", "mapping" + cache_suffix)

    full_dataset = MultiFlightEagleDataset(
        tif_paths, richness_csv, patch_size=patch_size,
        augment=True, cache_path=mapping_cache, mask_water_vapor=args.mask
    )
    if len(full_dataset) == 0:
        print(f"{Colors.FAIL}[!] No training samples found. Stopping.{Colors.ENDC}")
        return

    # Compute band stats (cached)
    band_stats_cache = os.path.join(project_root, "data", "eagle", "band_stats" + cache_suffix)
    full_dataset.compute_band_stats(cache_path=band_stats_cache)

    # Clear open files before DataLoader (critical on Windows to avoid process inheritance bugs)
    for ds in full_dataset.dataset_cache.values():
        ds.close()
    full_dataset.dataset_cache.clear()

    # -- Stratified Split logic --
    mappings = full_dataset.mappings
    sample_clusters = np.array([m[4] for m in mappings])
    unique_clusters = np.unique(sample_clusters)
    
    # Exclude noise (-1) from validation split options
    val_candidates = unique_clusters[unique_clusters != -1]

    if args.k > len(val_candidates):
        print(f"{Colors.WARNING}[!] Warning: Requested {args.k} folds, but only {len(val_candidates)} validation candidate clusters exist. Reducing k to {len(val_candidates)}.{Colors.ENDC}")
        args.k = len(val_candidates)

    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    print(f"\n{Colors.OKBLUE}[*] Cluster-Stratified {args.k}-Fold Split: "
          f"{len(full_dataset)} total sites, across {len(val_candidates)} clusters{Colors.ENDC}")

    # Site identifiers for the wide predictions matrix (column headers)
    site_ids = [m[5] if len(m) > 5 else f"site_{i}" for i, m in enumerate(mappings)]

    fold_results = []
    pred_matrix_rows = []
    t0 = time.time()

    for fold_idx, (train_clusters_idx, val_clusters_idx) in enumerate(kf.split(val_candidates)):
        val_clusters = val_candidates[val_clusters_idx]

        train_idx = np.where(~np.isin(sample_clusters, val_clusters))[0]
        val_idx = np.where(np.isin(sample_clusters, val_clusters))[0]

        if len(val_idx) == 0:
            print(f"{Colors.WARNING}[!] Fold {fold_idx+1} has 0 validation samples. Skipping.{Colors.ENDC}")
            continue

        metrics, all_preds = run_fold(
            fold_idx, args.k, train_idx, val_idx, val_clusters,
            full_dataset, config, device, args
        )
        fold_results.append(metrics)

        row = {'fold_idx': fold_idx + 1}
        row.update(dict(zip(site_ids, all_preds)))
        pred_matrix_rows.append(row)

    elapsed = time.time() - t0

    if pred_matrix_rows:
        actual_row = {'fold_idx': 'actual'}
        actual_row.update({site_id: mappings[i][3] for i, site_id in enumerate(site_ids)})
        pred_matrix_rows.append(actual_row)

        pred_matrix_df = pd.DataFrame(pred_matrix_rows)
        pred_matrix_path = os.path.join(
            project_root, "reports", f"kfold_predictions_matrix_eagle_{args.mode}.csv"
        )
        pred_matrix_df.to_csv(pred_matrix_path, index=False)
        print(f"{Colors.OKGREEN}[OK] Predictions matrix saved to: {pred_matrix_path}{Colors.ENDC}")

    if not fold_results:
        print(f"{Colors.FAIL}[!] No folds completed successfully.{Colors.ENDC}")
        return

    # -- Final Summary --
    df = pd.DataFrame(fold_results)
    mean_r2 = df['r2'].mean()
    std_r2 = df['r2'].std() if len(df) > 1 else 0
    mean_rmse = df['rmse'].mean()
    std_rmse = df['rmse'].std() if len(df) > 1 else 0
    mean_mae = df['mae'].mean()
    std_mae = df['mae'].std() if len(df) > 1 else 0

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"  CLUSTER-STRATIFIED {args.k}-FOLD CROSS-VALIDATION RESULTS (EAGLE {args.mode}){Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(df.to_string(index=False))
    print(f"{Colors.OKCYAN}{'-'*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean R2:   {mean_r2:.4f} +/- {std_r2:.4f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean MAE:  {mean_mae:.2f} +/- {std_mae:.2f}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  Total Time: {elapsed/60:.1f} minutes{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    # Save report
    report_path = os.path.join(project_root, "reports", f"kfold_cluster_strata_results_eagle_{args.mode}.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"{Colors.OKGREEN}[OK] Results saved to: {report_path}{Colors.ENDC}")

if __name__ == "__main__":
    main()
