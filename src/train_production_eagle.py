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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import yaml

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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eagle_dataset import MultiFlightEagleDataset
from src.vit_spatial_spectral import ViTSpatialSpectral
from src.train_production import GaiaTransferModel

def train_eagle(tif_dir=None, richness_csv=None, epochs=None, batch_size=None,
                test_run=False, freeze_encoder=True, unfreeze_epoch=None,
                mode="native", mask_water_vapor=True, patience=25):
    # Load config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    
    b_cfg = config['bioscape']
    
    # Select directory based on mode if not overridden
    if not tif_dir:
        if mode == "native":
            tif_dir = os.path.join(project_root, "data", "eagle", "30m_native")
        else:
            tif_dir = os.path.join(project_root, "data", "eagle", "30m_10nm")

    richness_csv = richness_csv or os.path.join(project_root, b_cfg['richness_csv'])
    epochs = epochs or b_cfg['epochs']
    batch_size = batch_size or b_cfg['batch_size']
    patch_size = b_cfg.get('patch_size', 16)
    
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"{Colors.HEADER}===================================================={Colors.ENDC}\n{Colors.HEADER}Gaia EAGLE Fine-Tuning ({mode} mode) on {device}{Colors.ENDC}")
    print(f"[*] Target Directory: {tif_dir}")
    print(f"[*] Labels CSV: {richness_csv}")

    if test_run:
        print(f"{Colors.FAIL}[!] TEST RUN ENABLED: Using only 2 tiles and 1 epoch.{Colors.ENDC}")
        epochs = 1

    if not os.path.exists(tif_dir):
        print(f"{Colors.FAIL}[!] Directory not found: {tif_dir}. Run extract_eagle_data.py first.{Colors.ENDC}")
        return

    tif_paths = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')]
    
    if not tif_paths:
        print(f"{Colors.FAIL}[!] No .tif files found in {tif_dir}{Colors.ENDC}")
        return

    if test_run:
        tif_paths = tif_paths[:2]

    # Caching path for mapped dataset
    cache_suffix = f"_p{patch_size}_{mode}.json"
    if test_run:
        cache_suffix = f"_p{patch_size}_{mode}_test.json"
    mapping_cache = os.path.join(project_root, "data", "eagle", "mapping" + cache_suffix)
    
    full_dataset = MultiFlightEagleDataset(
        tif_paths, richness_csv, patch_size=patch_size,
        augment=True, cache_path=mapping_cache, mask_water_vapor=mask_water_vapor
    )
    
    if len(full_dataset) == 0:
        print(f"{Colors.FAIL}[!] No training samples found. Stopping.{Colors.ENDC}")
        return

    # Compute/Load global per-band normalization stats
    band_stats_cache = os.path.join(project_root, "data", "eagle", "band_stats" + cache_suffix)
    full_dataset.compute_band_stats(cache_path=band_stats_cache)

    # Clear open files before DataLoader (critical on Windows to avoid process inheritance bugs)
    for ds in full_dataset.dataset_cache.values():
        ds.close()
    full_dataset.dataset_cache.clear()

    # Compute target richness stats
    all_richness = np.array([m[3] for m in full_dataset.mappings])
    richness_mean = float(all_richness.mean())
    richness_std = float(all_richness.std()) + 1e-6
    print(f"{Colors.OKBLUE}[*] Richness Stats: mean={richness_mean:.1f}, std={richness_std:.1f}, min={all_richness.min():.0f}, max={all_richness.max():.0f}{Colors.ENDC}")

    indices = np.arange(len(full_dataset))
    val_split = b_cfg.get('val_split', 0.2)
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)

    # Windows multi-processing doesn't work well with rasterio, so use num_workers=0 on Windows
    num_workers = 0 if os.name == 'nt' else b_cfg.get('num_workers', 4)
    print(f"{Colors.OKBLUE}[*] Enabling {num_workers} dataloader workers...{Colors.ENDC}")

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = GaiaTransferModel(num_targets=1, patch_size=patch_size).to(device)
    foundation_ckpt = os.path.join(project_root, "checkpoints", "pretrained_ViTSpatialSpectral_200ep_enmap.pth")
    if os.path.exists(foundation_ckpt):
        model.load_foundation_weights(foundation_ckpt, device)
    else:
        print(f"{Colors.WARNING}[!] Pretrained foundation weights not found at {foundation_ckpt}. Initializing randomly.{Colors.ENDC}")

    # Freeze encoder
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.mlp_head.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{Colors.OKBLUE}[*] Encoder FROZEN: {trainable}/{total} params trainable ({trainable/total*100:.1f}%){Colors.ENDC}")
    else:
        print(f"{Colors.OKBLUE}[*] Full fine-tuning: all parameters trainable{Colors.ENDC}")

    lr = b_cfg['learning_rate']
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.15
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    best_r2 = -float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Progressive unfreezing
        if freeze_encoder and unfreeze_epoch and epoch == unfreeze_epoch:
            print(f"\n{Colors.WARNING}[*] Epoch {epoch}: UNFREEZING encoder for fine-tuning{Colors.ENDC}")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.encoder.mlp_head.parameters(), 'lr': lr},
                {'params': [p for n, p in model.encoder.named_parameters() if 'mlp_head' not in n], 'lr': lr * 0.1},
            ], weight_decay=0.15)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs - epoch + 1,
                steps_per_epoch=len(train_loader), pct_start=0.05
            )

        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
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
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                preds_norm = model(images.to(device))
                preds_real = preds_norm.cpu().numpy().flatten() * richness_std + richness_mean
                val_preds.extend(preds_real)
                val_targets.extend(labels.numpy().flatten())

        epoch_r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0
        avg_loss = train_loss / len(train_loader)
        print(f"{Colors.OKCYAN}[*] Epoch {epoch} - Loss: {avg_loss:.4f} | Val R²: {epoch_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}{Colors.ENDC}")

        if epoch_r2 > best_r2:
            best_r2 = epoch_r2
            patience_counter = 0
            ckpt_data = {
                'model_state_dict': model.state_dict(),
                'richness_mean': richness_mean,
                'richness_std': richness_std,
                'band_mean': full_dataset.band_mean.flatten().tolist() if full_dataset.band_mean is not None else None,
                'band_std': full_dataset.band_std.flatten().tolist() if full_dataset.band_std is not None else None,
            }
            torch.save(ckpt_data, os.path.join(project_root, "checkpoints", "gaia_eagle_best.pth"))
            print(f"{Colors.OKGREEN}[OK] New Best EAGLE Model Saved (R²: {best_r2:.4f}){Colors.ENDC}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{Colors.WARNING}[*] Early stopping at epoch {epoch} (no improvement for {patience} epochs){Colors.ENDC}")
                break

    print(f"\n{Colors.OKGREEN}[OK] Training Complete. Best R²: {best_r2:.4f}{Colors.ENDC}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaia EAGLE Training Production Script")
    parser.add_argument("--tif_dir", type=str, help="Override path to EAGLE TIFF tiles directory")
    parser.add_argument("--richness_csv", type=str, help="Override path to richness labels CSV")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--test-run", action="store_true", help="Quick test run (1 epoch on 2 tiles)")
    parser.add_argument("--freeze", action="store_true", default=True, help="Freeze encoder, train only head (default)")
    parser.add_argument("--no-freeze", dest="freeze", action="store_false", help="Fine-tune all parameters")
    parser.add_argument("--unfreeze-epoch", type=int, default=None, help="Unfreeze encoder at this epoch for progressive fine-tuning")
    parser.add_argument("--mode", type=str, choices=["native", "10nm"], default="native",
                        help="Select EAGLE data mode: native (32 target tiles) or 10nm (1517 mosaic tiles)")
    parser.add_argument("--no-mask", dest="mask", action="store_false", help="Disable water vapor masking")
    parser.add_argument("--patience", type=int, default=25, help="Patience for early stopping")
    parser.set_defaults(mask=True)
    
    args = parser.parse_args()
    train_eagle(
        tif_dir=args.tif_dir, richness_csv=args.richness_csv, epochs=args.epochs,
        batch_size=args.batch_size, test_run=args.test_run, freeze_encoder=args.freeze,
        unfreeze_epoch=args.unfreeze_epoch, mode=args.mode, mask_water_vapor=args.mask,
        patience=args.patience
    )
