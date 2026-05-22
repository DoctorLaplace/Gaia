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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import yaml
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_production import GaiaTransferModel, MultiFlightBioScapeDataset

def run_fold(fold_idx, train_idx, val_idx, full_dataset, config, device, args):
    """Run training for a single fold."""
    print(f"\n{'='*60}")
    print(f"  STARTING FOLD {fold_idx+1}/{args.folds}")
    print(f"{'='*60}")
    
    b_cfg = config['bioscape']
    batch_size = args.batch_size or b_cfg['batch_size']
    epochs = args.epochs or b_cfg['epochs']
    lr = args.learning_rate or b_cfg['learning_rate']
    patch_size = b_cfg.get('patch_size', 16)
    
    # Compute target normalization stats ONLY from train_idx to avoid leakage
    train_richness = np.array([full_dataset.mappings[i][3] for i in train_idx])
    richness_mean = float(train_richness.mean())
    richness_std = float(train_richness.std()) + 1e-6
    
    # Create loaders
    # Note: Using num_workers=0 for local Windows to avoid H5PY pickling issues
    num_workers = 0 if os.name == 'nt' and (not args.nc_dir or not args.nc_dir.startswith("s3")) else 4
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, 
                              prefetch_factor=4 if num_workers > 0 else None, 
                              persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, 
                            prefetch_factor=4 if num_workers > 0 else None, 
                            persistent_workers=True if num_workers > 0 else False)
    
    # Initialize Model
    model = GaiaTransferModel(num_targets=1, patch_size=patch_size).to(device)
    foundation_ckpt = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "pretrained_ViTSpatialSpectral_200ep_enmap.pth")
    if os.path.exists(foundation_ckpt):
        model.load_foundation_weights(foundation_ckpt, device)
    
    # Freeze encoder
    if args.freeze:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.mlp_head.parameters():
            param.requires_grad = True
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.1)
    criterion = nn.MSELoss()
    
    best_fold_r2 = -float('inf')
    best_fold_metrics = {}
    patience_counter = 0
    patience = 25
    
    fold_ckpt_path = os.path.join("checkpoints", f"gaia_kfold_{fold_idx+1}.pth")
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx+1} Ep {epoch}", leave=False)
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
            
        # Eval
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                out_norm = model(images.to(device))
                out_real = out_norm.cpu().numpy().flatten() * richness_std + richness_mean
                val_preds.extend(out_real)
                val_targets.extend(labels.numpy().flatten())
        
        r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0
        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        mae = mean_absolute_error(val_targets, val_preds)
        
        if r2 > best_fold_r2:
            best_fold_r2 = r2
            best_fold_metrics = {'fold': fold_idx+1, 'r2': r2, 'rmse': rmse, 'mae': mae, 'epoch': epoch}
            patience_counter = 0
            torch.save(model.state_dict(), fold_ckpt_path)
        else:
            patience_counter += 1
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:3d}] Loss: {train_loss/len(train_loader):.4f} | R2: {r2:7.4f} | Best: {best_fold_r2:7.4f}")
            
        if patience_counter >= patience:
            print(f"  [*] Early stopping at epoch {epoch} (best R2: {best_fold_r2:.4f})")
            break
            
    print(f"[✔] Fold {fold_idx+1} Complete. Best R2: {best_fold_r2:.4f}")
    return best_fold_metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--freeze", action="store_true", default=True)
    parser.add_argument("--nc_dir", type=str)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
        
    b_cfg = config['bioscape']
    nc_dir = args.nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else b_cfg['nc_dir_local'])
    richness_csv = os.path.join(project_root, b_cfg['richness_csv'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset & Mappings
    if nc_dir.startswith("s3://"):
        from src.s3_utils import get_s3_fs
        fs = get_s3_fs()
        nc_paths = ["s3://" + f for f in fs.ls(nc_dir) if f.endswith('.nc')]
    else:
        nc_paths = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')])
    
    if args.test_run:
        nc_paths = nc_paths[:3]
        args.epochs = 2
        args.folds = 2
        
    cache_name = ("s3_mapping" if nc_dir.startswith("s3") else "local_mapping") + f"_p{b_cfg.get('patch_size', 16)}.json"
    mapping_cache = os.path.join(project_root, "data", "bioscape", cache_name)
    
    dataset = MultiFlightBioScapeDataset(nc_paths, richness_csv, cache_path=mapping_cache)
    
    # 2. Extract Groups (Flightlines)
    # mappings[i] = (nc_path, lat, lon, richness)
    flightlines = [os.path.basename(m[0]) for m in dataset.mappings]
    indices = np.arange(len(dataset))
    
    print(f"\n[*] Initializing {args.folds}-Fold Group Cross-Validation...")
    print(f"[*] Total Sites: {len(dataset)} | Unique Flightlines: {len(set(flightlines))}")
    
    # 3. K-Fold Loop
    gkf = GroupKFold(n_splits=args.folds)
    fold_results = []
    
    for i, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=flightlines)):
        res = run_fold(i, train_idx, val_idx, dataset, config, device, args)
        fold_results.append(res)
        
    # 4. Final Summary
    df = pd.DataFrame(fold_results)
    print(f"\n{'═'*60}")
    print(f"  CROSS-VALIDATION FINAL SUMMARY ({args.folds} Folds)")
    print(f"{'═'*60}")
    print(df.to_string(index=False))
    print(f"{'─'*60}")
    print(f"Mean R²:   {df['r2'].mean():.4f} ± {df['r2'].std():.4f}")
    print(f"Mean RMSE: {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f}")
    print(f"Mean MAE:  {df['mae'].mean():.2f} ± {df['mae'].std():.2f}")
    print(f"{'═'*60}")
    
    # Save Report
    report_path = os.path.join(project_root, "reports", "kfold_results.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"[✔] K-Fold results saved to: {report_path}")

if __name__ == "__main__":
    main()
