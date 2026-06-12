"""
Gaia K-Fold Cross-Validation

Runs K independent train/val cycles on the full BioSCape dataset,
each with a different fold held out for validation. Reports per-fold
R2, RMSE, and MAE, then prints the mean and standard deviation across
all folds to give a robust estimate of model performance.

Usage:
    python -m src.train_kfold                        # 5-fold (default)
    python -m src.train_kfold --k 10                 # 10-fold
    python -m src.train_kfold --k 3 --test-run       # quick smoke test
    python -m src.train_kfold --unfreeze-epoch 30    # progressive unfreezing
"""
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

from src.train_production import GaiaTransferModel, MultiFlightBioScapeDataset
try:
    from src.s3_utils import get_s3_fs, get_s3_bytes_pulled, get_worker_statuses
except ImportError:
    from s3_utils import get_s3_fs, get_s3_bytes_pulled, get_worker_statuses


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# ─────────────────────────────────────────────────────────────
#  Single Fold
# ─────────────────────────────────────────────────────────────


def run_fold(fold_idx, k, train_idx, val_idx, full_dataset,
             config, device, args):
    """Train and evaluate a single fold. Returns a metrics dict."""
    b_cfg = config['bioscape']
    batch_size = args.batch_size or b_cfg['batch_size']
    epochs = args.epochs or b_cfg['epochs']
    lr = b_cfg['learning_rate']
    patch_size = b_cfg.get('patch_size', 16)

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  FOLD {fold_idx+1} / {k}  "
          f"(Train: {len(train_idx)} | Val: {len(val_idx)}){Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    # -- Target normalization (train-only to prevent leakage) --
    train_richness = np.array([full_dataset.mappings[i][3] for i in train_idx])
    richness_mean = float(train_richness.mean())
    richness_std = float(train_richness.std()) + 1e-6
    print(f"{Colors.OKBLUE}[*] Fold {fold_idx+1} Richness: "
          f"mean={richness_mean:.1f}, std={richness_std:.1f}{Colors.ENDC}")

    # -- DataLoaders --
    num_workers = b_cfg.get('num_workers', 4)
    nc_dir = args.nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else b_cfg['nc_dir_local'])
    
    if args.mosaic:
        nc_dir = args.nc_dir or f"data/bioscape/{patch_size}m_v2"

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
    patience_counter = 0
    patience = 25

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
                'r2': round(r2, 4),
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'best_epoch': epoch
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  {Colors.WARNING}[*] Early stopping at epoch {epoch} "
                      f"(best R2: {best_r2:.4f}){Colors.ENDC}")
                break

    print(f"{Colors.OKGREEN}[OK] Fold {fold_idx+1} Complete. "
          f"Best R2: {best_r2:.4f}{Colors.ENDC}")
    return best_metrics


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Gaia K-Fold Cross-Validation"
    )
    parser.add_argument("--k", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--epochs", type=int,
                        help="Override epochs per fold")
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
                        help="Quick smoke test (2 folds, 2 epochs, 2 granules)")
    args = parser.parse_args()

    # -- Load Config --
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    b_cfg = config['bioscape']
    nc_dir = args.nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3']
                             else b_cfg['nc_dir_local'])
    richness_csv = os.path.join(project_root, b_cfg['richness_csv'])
    patch_size = b_cfg.get('patch_size', 16)
    device = torch.device(
        config.get('device', 'cuda') if torch.cuda.is_available() else "cpu"
    )

    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  GAIA {args.k}-FOLD CROSS-VALIDATION{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    if args.test_run:
        print(f"{Colors.FAIL}[!] TEST RUN: 2 folds, 2 epochs, 2 granules{Colors.ENDC}")
        args.k = 2
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
            with open(tile_path, 'r') as f:
                tiles = {t['tile'] for t in json.load(f)['tiles']}
            
            nc_paths = [p for p in nc_paths if any(t in p for t in tiles)]
            print(f"{Colors.OKBLUE}[*] Filtered to {len(nc_paths)} tiles from tile_names.json{Colors.ENDC}")

    # -- Build Dataset --
    cache_suffix = f"_p{patch_size}.json"
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

    # -- K-Fold Split --
    indices = np.arange(len(full_dataset))
    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)

    print(f"\n{Colors.OKBLUE}[*] {args.k}-Fold Split: "
          f"{len(full_dataset)} total sites{Colors.ENDC}")

    fold_results = []
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        metrics = run_fold(
            fold_idx, args.k, train_idx, val_idx,
            full_dataset, config, device, args
        )
        fold_results.append(metrics)

    elapsed = time.time() - t0

    # -- Final Summary --
    df = pd.DataFrame(fold_results)
    mean_r2 = df['r2'].mean()
    std_r2 = df['r2'].std()
    mean_rmse = df['rmse'].mean()
    std_rmse = df['rmse'].std()
    mean_mae = df['mae'].mean()
    std_mae = df['mae'].std()

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  {args.k}-FOLD CROSS-VALIDATION RESULTS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(df.to_string(index=False))
    print(f"{Colors.OKCYAN}{'-'*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean R2:   {mean_r2:.4f} +/- {std_r2:.4f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Mean MAE:  {mean_mae:.2f} +/- {std_mae:.2f}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  Total Time: {elapsed/60:.1f} minutes{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    # Save report
    report_path = os.path.join(project_root, "reports", "kfold_results.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"{Colors.OKGREEN}[OK] Results saved to: {report_path}{Colors.ENDC}")


if __name__ == "__main__":
    main()
