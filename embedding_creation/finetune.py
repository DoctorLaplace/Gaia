"""
Fine-tune the AVIRIS-NG Foundation Model on BioSCape Richness

Loads the pre-trained encoder from pretrain.py and adds a regression head.
Uses native 370-band patches (no spectral resampling).

Usage:
    python -m embedding_creation.finetune                        # full run
    python -m embedding_creation.finetune --test-run              # quick test
    python -m embedding_creation.finetune --no-freeze             # full fine-tune
"""
import os
import sys
import time
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vit_spatial_spectral import ViTSpatialSpectral
from embedding_creation.dataset import AvirisRichnessDataset


def resolve_res(cfg, res):
    """Replace {res} placeholder in config paths with actual resolution."""
    res_str = str(res)
    cfg['data']['labeled_nc_dir'] = cfg['data']['labeled_nc_dir'].replace('{res}', res_str)
    cfg['data']['ssl_nc_dir'] = cfg['data']['ssl_nc_dir'].replace('{res}', res_str)
    return cfg


class GaiaAvirisModel(nn.Module):
    """
    Richness regression model built on native AVIRIS-NG encoder.
    Encoder is initialized from pre-trained SimMIM weights.
    """
    def __init__(self, num_channels, dim, depth, heads, mlp_dim,
                 spectral_patch_size, spatial_patch_size, image_size,
                 num_targets=1, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        self.encoder = ViTSpatialSpectral(
            image_size=image_size,
            spatial_patch_size=spatial_patch_size,
            spectral_patch_size=spectral_patch_size,
            num_classes=num_targets,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=num_channels,
            spectral_pos=list(range(num_channels // spectral_patch_size)),
            spectral_pos_embed=True,
            blockwise_patch_embed=True,
            spectral_only=False,
            pixelwise=False,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        # Replace classification head with 3-layer regression head
        self.encoder.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_targets),
        )

    def load_pretrained(self, checkpoint_path, device):
        """Load encoder weights from pre-training checkpoint."""
        print(f"[*] Loading pre-trained weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        # Filter out keys that don't belong to the encoder
        encoder_state = {}
        for k, v in state_dict.items():
            if 'mlp_head' in k:
                continue  # skip old classification head
            encoder_state[k] = v

        result = self.encoder.load_state_dict(encoder_state, strict=False)
        loaded = len(state_dict) - len(result.unexpected_keys)
        print(f"    Loaded {loaded} parameter tensors "
              f"(missing: {len(result.missing_keys)}, "
              f"unexpected: {len(result.unexpected_keys)})")
        return ckpt

    def forward(self, x):
        out = self.encoder(x)  # (B, H, W, num_targets) or (B, num_targets)
        if out.dim() > 2:
            out = out.mean(dim=(1, 2))  # spatial average pooling
        return out


def finetune(args):
    config_path = os.path.join(PROJECT_ROOT, "embedding_creation", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply test-run overrides
    if args.test_run:
        print("\n[!] TEST RUN MODE\n")
        cfg['finetune']['epochs'] = cfg['test_run']['finetune_epochs']
        cfg['finetune']['batch_size'] = cfg['test_run']['batch_size']
        max_granules = cfg['test_run']['max_granules']
    else:
        max_granules = args.max_granules

    if args.epochs:
        cfg['finetune']['epochs'] = args.epochs
    if args.batch_size:
        cfg['finetune']['batch_size'] = args.batch_size

    # Resolution
    res = args.res or cfg['data']['default_res']
    cfg = resolve_res(cfg, res)
    print(f"{Colors.OKCYAN}[*] Resolution: {res}m{Colors.ENDC}")

    epochs = cfg['finetune']['epochs']
    batch_size = cfg['finetune']['batch_size']
    lr = cfg['finetune']['lr']
    fp16 = cfg['finetune']['fp16']
    freeze_encoder = cfg['finetune']['freeze_encoder'] and not args.no_freeze
    unfreeze_epoch = args.unfreeze_epoch or cfg['finetune']['unfreeze_epoch']
    patience = cfg['finetune']['patience']
    ckpt_dir = os.path.join(PROJECT_ROOT, cfg['finetune']['checkpoint_dir'])
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{Colors.OKBLUE}[*] Device: {device}{Colors.ENDC}")

    # ── Dataset ──
    # Fine-tuning only uses the labeled granules (existing 535)
    nc_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'])
    richness_csv = os.path.join(PROJECT_ROOT, cfg['data']['richness_csv'])

    # Load band stats from pre-training
    pretrain_ckpt_path = os.path.join(ckpt_dir, "pretrained_aviris_best.pth")
    band_mean, band_std = None, None
    if os.path.exists(pretrain_ckpt_path):
        ckpt = torch.load(pretrain_ckpt_path, map_location='cpu', weights_only=False)
        if 'band_mean' in ckpt:
            band_mean = torch.tensor(ckpt['band_mean']).reshape(cfg['data']['num_bands'], 1, 1)
            band_std = torch.tensor(ckpt['band_std']).reshape(cfg['data']['num_bands'], 1, 1)
            print(f"{Colors.OKGREEN}[OK] Loaded band normalization stats from pre-training checkpoint{Colors.ENDC}")

    dataset = AvirisRichnessDataset(
        nc_dir=nc_dir,
        richness_csv=richness_csv,
        patch_size=cfg['data']['patch_size'],
        num_bands=cfg['data']['num_bands'],
        bands_to_trim=cfg['data']['bands_to_trim'],
        augment=True,
        max_granules=max_granules,
        band_mean=band_mean,
        band_std=band_std,
    )

    if len(dataset) == 0:
        print("[!] No training samples found.")
        return

    # Target normalization
    all_richness = np.array([m[3] for m in dataset.mappings])
    richness_mean = float(all_richness.mean())
    richness_std = float(all_richness.std()) + 1e-6
    print(f"{Colors.OKBLUE}[*] Richness: mean={richness_mean:.1f}, std={richness_std:.1f}, "
          f"range=[{all_richness.min():.0f}, {all_richness.max():.0f}]{Colors.ENDC}")

    # Train/val split
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg['finetune']['val_split'], random_state=42
    )

    # Close h5 handles before DataLoader (Windows multiprocessing safety)
    dataset.close()

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=cfg['finetune']['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=cfg['finetune']['num_workers'], pin_memory=True
    )

    # ── Model ──
    model = GaiaAvirisModel(
        num_channels=cfg['data']['num_bands'],
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        heads=cfg['model']['heads'],
        mlp_dim=cfg['model']['mlp_dim'],
        spectral_patch_size=cfg['model']['spectral_patch_size'],
        spatial_patch_size=cfg['model']['spatial_patch_size'],
        image_size=cfg['data']['patch_size'],
        num_targets=1,
        dropout=cfg['model']['dropout'],
        emb_dropout=cfg['model']['emb_dropout'],
    ).to(device)

    # Load pre-trained encoder
    if os.path.exists(pretrain_ckpt_path):
        model.load_pretrained(pretrain_ckpt_path, device)
    else:
        print(f"[!] No pre-trained checkpoint found at {pretrain_ckpt_path}")
        print("    Training from scratch (no foundation model).")

    # Freeze encoder
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.mlp_head.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{Colors.OKBLUE}[*] Encoder FROZEN: {trainable:,}/{total:,} params trainable "
              f"({trainable/total*100:.1f}%){Colors.ENDC}")

    # ── Optimizer ──
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg['finetune']['weight_decay']
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    scaler = torch.amp.GradScaler('cuda', enabled=fp16)

    # ── Training ──
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  AVIRIS-NG Richness Fine-tuning (370 native bands){Colors.ENDC}")
    print(f"{Colors.OKCYAN}  Epochs: {epochs}  |  Train: {len(train_idx)}  |  Val: {len(val_idx)}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  Freeze: {freeze_encoder}  |  Unfreeze at: {unfreeze_epoch}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    best_r2 = -float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Progressive unfreezing
        if freeze_encoder and unfreeze_epoch and epoch == unfreeze_epoch:
            print(f"\n{Colors.WARNING}[*] Epoch {epoch}: UNFREEZING encoder{Colors.ENDC}")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.encoder.mlp_head.parameters(), 'lr': lr},
                {'params': [p for n, p in model.encoder.named_parameters()
                           if 'mlp_head' not in n], 'lr': lr * 0.3},
            ], weight_decay=cfg['finetune']['weight_decay'])
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs - epoch + 1,
                steps_per_epoch=len(train_loader), pct_start=0.20
            )

        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"{Colors.OKCYAN}Epoch {epoch}/{epochs}{Colors.ENDC}", leave=False)

        for patches, labels in pbar:
            patches = patches.to(device)
            labels_norm = ((labels - richness_mean) / richness_std).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=fp16):
                preds = model(patches)
                loss = criterion(preds, labels_norm)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # ── Validation ──
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for patches, labels in val_loader:
                with torch.amp.autocast('cuda', enabled=fp16):
                    preds_norm = model(patches.to(device))
                preds_real = preds_norm.cpu().numpy().flatten() * richness_std + richness_mean
                val_preds.extend(preds_real)
                val_targets.extend(labels.numpy().flatten())

        epoch_r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0
        avg_loss = train_loss / max(len(train_loader), 1)
        print(f"{Colors.OKCYAN}[Epoch {epoch:3d}/{epochs}]{Colors.ENDC} "
              f"Loss: {Colors.BOLD}{avg_loss:.4f}{Colors.ENDC} | Val R2: {Colors.OKGREEN}{epoch_r2:.4f}{Colors.ENDC} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if epoch_r2 > best_r2:
            best_r2 = epoch_r2
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'richness_mean': richness_mean,
                'richness_std': richness_std,
                'band_mean': band_mean.flatten().tolist() if band_mean is not None else None,
                'band_std': band_std.flatten().tolist() if band_std is not None else None,
                'config': cfg,
                'best_r2': best_r2,
            }, os.path.join(ckpt_dir, "finetuned_aviris_best.pth"))
            print(f"    {Colors.OKGREEN}-> New best R2: {best_r2:.4f} (saved){Colors.ENDC}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{Colors.WARNING}[*] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs){Colors.ENDC}")
                break

    print(f"\n{Colors.OKGREEN}[OK] Fine-tuning complete. Best R2: {best_r2:.4f}{Colors.ENDC}")
    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVIRIS-NG Richness Fine-tuning")
    parser.add_argument("--test-run", action="store_true",
                        help="Minimal run: 3 epochs, 5 granules")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-granules", type=int)
    parser.add_argument("--res", type=int, help="Resolution in meters (default: from config, 30)")
    parser.add_argument("--no-freeze", action="store_true",
                        help="Fine-tune all parameters (not just head)")
    parser.add_argument("--unfreeze-epoch", type=int,
                        help="Epoch at which to unfreeze the encoder (overrides config)")
    args = parser.parse_args()
    finetune(args)
