import os
import sys
import time
import copy
import hashlib
import json
from datetime import datetime, timezone
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

from src.eagle_dataset import MultiFlightEagleDataset
from src.enmap_bands import INPUT_SPECTRAL_MAPPING
from src.model_transfer import GaiaTransferModel, Colors
from src.seed import set_seed

def run_fold(fold_idx, k, train_idx, val_idx, val_clusters, full_dataset, config, device, args):
    """Fit once, selecting weights on outer validation RMSE (not independent testing)."""
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Folding requires nonempty training and validation partitions")
    seed = (args.seed + fold_idx) % 2**32
    set_seed(seed)
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
    full_dataset.compute_band_stats(train_idx)
    for ds in full_dataset.dataset_cache.values():
        ds.close()
    full_dataset.dataset_cache.clear()

    # -- DataLoaders --
    # Windows doesn't handle multi-processing for rasterio cleanly, default to 0 on Windows
    # num_workers = 0 if os.name == 'nt' else b_cfg.get('num_workers', 4)
    num_workers = 0
    
    train_loader = DataLoader(
        Subset(full_dataset, train_idx.tolist()),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx.tolist()),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(seed)
    )

    # -- Model --
    model = GaiaTransferModel(num_targets=1, patch_size=patch_size).to(device)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    foundation_ckpt = os.path.join(
        project_root, "checkpoints", "pretrained_ViTSpatialSpectral_200ep_enmap.pth"
    )
    if not os.path.isfile(foundation_ckpt):
        raise FileNotFoundError(f"Foundation checkpoint not found: {foundation_ckpt}")
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
    # Omit only the zero-length warmup phase; keep the configured schedule horizon.
    pct_start = 0.1
    if pct_start * (epochs * len(train_loader)) == 1:
        pct_start = 0.0
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=pct_start
    )

    best_rmse = float('inf')
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
            pct_start = 0.05
            if pct_start * ((epochs - epoch + 1) * len(train_loader)) == 1:
                pct_start = 0.0
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=[lr, lr * 0.3], epochs=epochs - epoch + 1,
                steps_per_epoch=len(train_loader), pct_start=pct_start
            )

        # -- Train --
        # Shared augmentation flag requires sequential loaders with num_workers=0.
        full_dataset.augment = True
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
        full_dataset.augment = False
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                out_norm = model(images.to(device))
                out_real = (out_norm.cpu().numpy().flatten()
                            * richness_std + richness_mean)
                val_preds.extend(out_real)
                val_targets.extend(labels.numpy().flatten())

        if not np.isfinite(val_preds).all() or not np.isfinite(val_targets).all():
            raise ValueError(f"Fold {fold_idx+1} has nonfinite validation predictions or targets")
        rmse = float(np.sqrt(mean_squared_error(val_targets, val_preds)))
        if not np.isfinite(rmse):
            raise ValueError(f"Fold {fold_idx+1} has nonfinite validation RMSE")
        avg_loss = train_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {Colors.OKCYAN}[Epoch {epoch:3d}] "
                  f"Loss: {avg_loss:.4f} | RMSE: {rmse:.4f} | "
                  f"Best RMSE: {best_rmse:.4f}{Colors.ENDC}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_metrics = {'best_epoch': epoch}
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {Colors.WARNING}[*] Early stopping at epoch {epoch} "
                      f"(best RMSE: {best_rmse:.4f}){Colors.ENDC}")
                break

    print(f"{Colors.OKGREEN}[OK] Fold {fold_idx+1} Complete. "
          f"Best validation RMSE: {best_rmse:.4f}{Colors.ENDC}")

    # -- Use the best-epoch weights (not just the last epoch) for predictions/plot --
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # -- Dense inference over the ENTIRE dataset (train + held-out) with this fold's model --
    model.eval()
    full_dataset.augment = False  # deterministic predictions, no random rot/flip
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(seed))
    all_preds = []
    with torch.no_grad():
        for images, labels in full_loader:
            out_norm = model(images.to(device))
            all_preds.extend(out_norm.cpu().numpy().flatten() * richness_std + richness_mean)
    all_preds = np.asarray(all_preds, dtype=np.float64)
    outer_actual = np.array([full_dataset.mappings[i][3] for i in val_idx])
    if not np.isfinite(all_preds).all() or not np.isfinite(outer_actual).all():
        raise ValueError(f"Fold {fold_idx+1} has nonfinite dense predictions or outer targets")
    outer_preds = all_preds[val_idx]
    outer_r2 = (float(r2_score(outer_actual, outer_preds))
                if len(outer_actual) >= 2 and np.any(outer_actual != outer_actual[0])
                else np.nan)
    best_metrics.update({
        'fold': fold_idx + 1,
        'held_out_clusters': ",".join(map(str, sorted(list(val_clusters)))),
        'r2': outer_r2,
        'rmse': float(np.sqrt(mean_squared_error(outer_actual, outer_preds))),
        'mae': float(mean_absolute_error(outer_actual, outer_preds)),
    })

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
                 f"R² (outer-validation-selected) = {outer_r2:.4f}", fontsize='small')
    ax.legend(fontsize='x-small', loc='upper left')
    fig.tight_layout()

    plots_dir = os.path.join(args.output_dir, "kfold_plots")
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
    parser.add_argument("--test-run", action="store_true",
                        help="Quick smoke test (2 folds, 2 epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for cluster splits and training RNGs (default: 42)")
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


    # -- Build Dataset --
    cache_suffix = f"_p{patch_size}_{args.mode}_strata.json"
    mapping_cache = os.path.join(project_root, "data", "eagle", "mapping" + cache_suffix)
    if not os.path.isfile(mapping_cache):
        raise FileNotFoundError(
            f"Audited mapping cache is required; do not regenerate the cohort: {mapping_cache}"
        )

    full_dataset = MultiFlightEagleDataset(
        tif_paths, richness_csv, patch_size=patch_size,
        augment=True, cache_path=mapping_cache
    )
    if len(full_dataset) == 0:
        print(f"{Colors.FAIL}[!] No training samples found. Stopping.{Colors.ENDC}")
        return


    # -- Stratified Split logic --
    mappings = full_dataset.mappings
    sample_clusters = np.array([m[4] for m in mappings])
    unique_clusters = np.unique(sample_clusters)
    
    # Exclude noise (-1) from validation split options
    val_candidates = unique_clusters[unique_clusters != -1]
    if args.test_run:
        if len(val_candidates) < 2:
            raise ValueError("Smoke evaluation requires at least two eligible clusters")
        smoke_clusters = val_candidates[:4]
        full_dataset.mappings = [
            m for m in mappings if m[4] == -1 or m[4] in smoke_clusters
        ]
        mappings = full_dataset.mappings
        sample_clusters = np.array([m[4] for m in mappings])
        val_candidates = smoke_clusters

    if args.k > len(val_candidates):
        print(f"{Colors.WARNING}[!] Warning: Requested {args.k} folds, but only {len(val_candidates)} validation candidate clusters exist. Reducing k to {len(val_candidates)}.{Colors.ENDC}")
        args.k = len(val_candidates)
    if args.k < 2:
        raise ValueError("Outer evaluation requires k >= 2")

    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    outer_splits = []
    for _, val_clusters_idx in kf.split(val_candidates):
        val_clusters = val_candidates[val_clusters_idx]
        train_idx = np.where(~np.isin(sample_clusters, val_clusters))[0]
        val_idx = np.where(np.isin(sample_clusters, val_clusters))[0]
        if not len(val_idx):
            raise ValueError("Outer evaluation requires nonempty held-out partitions")
        if not len(train_idx):
            raise ValueError("Outer evaluation requires nonempty training partitions")
        outer_splits.append((train_idx, val_idx, val_clusters))

    # Capture inputs before any fitting; a completed manifest identifies this exact run.
    foundation_ckpt = os.path.join(
        project_root, "checkpoints", "pretrained_ViTSpatialSpectral_200ep_enmap.pth"
    )
    hash_paths = {
        'foundation_checkpoint': foundation_ckpt,
        'mapping': mapping_cache,
        'source_labels': richness_csv,
    }
    for name in ('train_kfold_cluster_strata_eagle.py', 'eagle_dataset.py',
                 'enmap_bands.py', 'model_transfer.py', 'seed.py'):
        hash_paths[f'src/{name}'] = os.path.join(project_root, 'src', name)
    input_hashes = {}
    for name, path in hash_paths.items():
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        input_hashes[name] = digest.hexdigest()
    while True:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        run_name = f"{timestamp}-eagle-{args.mode}-k{args.k}"
        if args.test_run:
            run_name += "-smoke"
        args.output_dir = os.path.join(project_root, "reports", run_name)
        try:
            os.makedirs(args.output_dir, exist_ok=False)
        except FileExistsError:
            continue
        break
    print(f"Run directory: {os.path.abspath(args.output_dir)}")
    print("Evaluation protocol: single fit; outer validation selects best weights, "
          "so these scores are not independent nested-test estimates.")

    print(f"\n{Colors.OKBLUE}[*] Cluster-Stratified {args.k}-Fold Split: "
          f"{len(full_dataset)} total sites, across {len(val_candidates)} clusters{Colors.ENDC}")

    # Site identifiers for the wide predictions matrix (column headers)
    site_ids = [m[5] if len(m) > 5 else f"site_{i}" for i, m in enumerate(mappings)]

    fold_results = []
    pred_matrix_rows = []
    oof_idx_parts = []   # held-out sample indices, per fold
    oof_pred_parts = []  # held-out predictions, per fold (for pooled R^2)
    t0 = time.time()

    for fold_idx, (train_idx, val_idx, val_clusters) in enumerate(outer_splits):

        metrics, all_preds = run_fold(
            fold_idx, args.k, train_idx, val_idx, val_clusters,
            full_dataset, config, device, args
        )
        fold_results.append(metrics)

        # Dense predictions use this fold's outer-validation-selected weights.
        # Only this fold's held-out indices contribute to pooled metrics.
        oof_idx_parts.append(val_idx)
        oof_pred_parts.append(np.asarray(all_preds)[val_idx])

        row = {'fold_idx': fold_idx + 1}
        row.update(dict(zip(site_ids, all_preds)))
        pred_matrix_rows.append(row)

    elapsed = time.time() - t0


    # -- Pooled ("concatenated") out-of-fold R²: every held-out prediction on one graph --
    oof_idx = np.concatenate(oof_idx_parts)
    oof_pred = np.concatenate(oof_pred_parts)
    oof_actual = np.array([mappings[i][3] for i in oof_idx])
    eligible_idx = np.flatnonzero(sample_clusters != -1)
    if not np.array_equal(np.sort(oof_idx), eligible_idx):
        raise ValueError("OOF coverage must contain every non-noise mapped index exactly once")
    if not np.isfinite(oof_actual).all() or not np.isfinite(oof_pred).all():
        raise ValueError("Pooled evaluation contains nonfinite targets or predictions")

    pooled_r2 = r2_score(oof_actual, oof_pred)
    pooled_rmse = np.sqrt(mean_squared_error(oof_actual, oof_pred))
    pooled_mae = mean_absolute_error(oof_actual, oof_pred)
    if not np.isfinite([pooled_r2, pooled_rmse, pooled_mae]).all():
        raise ValueError("Pooled evaluation metrics must be finite")

    actual_row = {'fold_idx': 'actual'}
    actual_row.update({site_id: mappings[i][3] for i, site_id in enumerate(site_ids)})
    pred_matrix_rows.append(actual_row)
    pred_matrix_path = os.path.join(
        args.output_dir, f"kfold_predictions_matrix_eagle_{args.mode}.csv"
    )
    pd.DataFrame(pred_matrix_rows).to_csv(pred_matrix_path, index=False)
    print(f"{Colors.OKGREEN}[OK] Predictions matrix saved to: {pred_matrix_path}{Colors.ENDC}")

    oof_df = pd.DataFrame({
        'site_id': [site_ids[i] for i in oof_idx],
        'actual': oof_actual,
        'predicted': oof_pred,
    })
    oof_path = os.path.join(args.output_dir, f"kfold_pooled_oof_eagle_{args.mode}.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"{Colors.OKGREEN}[OK] Pooled out-of-fold predictions saved to: {oof_path}{Colors.ENDC}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(oof_actual, oof_pred, color='tab:blue', alpha=0.6, s=18)
    lims = [oof_actual.min() - 2, oof_actual.max() + 2]
    ax.plot(lims, lims, '--k', alpha=0.3, label='1:1')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual richness")
    ax.set_ylabel("Predicted richness")
    ax.set_title(f"Pooled out-of-fold ({len(oof_idx)} sites)\n"
                 f"R² = {pooled_r2:.4f}  RMSE = {pooled_rmse:.2f}", fontsize='small')
    ax.legend(fontsize='x-small', loc='upper left')
    fig.tight_layout()
    pooled_plot_path = os.path.join(args.output_dir, "kfold_plots",
                                    f"pooled_oof_scatter_eagle_{args.mode}.png")
    os.makedirs(os.path.dirname(pooled_plot_path), exist_ok=True)
    fig.savefig(pooled_plot_path, dpi=150)
    plt.close(fig)
    print(f"{Colors.OKGREEN}[OK] Saved pooled out-of-fold scatter plot to: {pooled_plot_path}{Colors.ENDC}")

    # -- Final Summary --
    df = pd.DataFrame(fold_results)
    defined_r2 = df['r2'].dropna()
    mean_r2 = defined_r2.mean()
    std_r2 = defined_r2.std() if len(defined_r2) > 1 else (0.0 if len(defined_r2) else np.nan)
    mean_rmse = df['rmse'].mean()
    std_rmse = df['rmse'].std() if len(df) > 1 else 0
    mean_mae = df['mae'].mean()
    std_mae = df['mae'].std() if len(df) > 1 else 0

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"  CLUSTER-STRATIFIED {args.k}-FOLD CROSS-VALIDATION RESULTS (EAGLE {args.mode}){Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(df.to_string(index=False))
    print(f"{Colors.OKCYAN}{'-'*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean R2:   {mean_r2:.4f} +/- {std_r2:.4f} "
          f"({len(defined_r2)} defined folds){Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean MAE:  {mean_mae:.2f} +/- {std_mae:.2f}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Pooled R2:   {pooled_r2:.4f}   (all {len(oof_idx)} held-out preds, one graph){Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Pooled RMSE: {pooled_rmse:.2f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Pooled MAE:  {pooled_mae:.2f}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  Total Time: {elapsed/60:.1f} minutes{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    # Save report
    report_path = os.path.join(args.output_dir, f"kfold_cluster_strata_results_eagle_{args.mode}.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"{Colors.OKGREEN}[OK] Results saved to: {report_path}{Colors.ENDC}")
    manifest = {
        'command': sys.orig_argv.copy(),
        'k': args.k,
        'seed': args.seed,
        'evaluation_protocol': 'single_fit_outer_validation',
        'input_spectral_mapping': INPUT_SPECTRAL_MAPPING,
        'elapsed_seconds': time.time() - t0,
        'sha256': input_hashes,
        'metrics': {
            'n': len(oof_idx),
            'pooled_r2': float(pooled_r2),
            'rmse': float(pooled_rmse),
            'mae': float(pooled_mae),
        },
        'exit_status': 0,
    }
    with open(os.path.join(args.output_dir, 'run.json'), 'x') as stream:
        json.dump(manifest, stream, allow_nan=False, separators=(',', ':'))

if __name__ == "__main__":
    main()
