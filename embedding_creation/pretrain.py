"""
SimMIM Pre-training for AVIRIS-NG Native Spectral Foundation Model

Masked Spatial-Spectral Transformer pre-training on 370 bands.
Uses the existing ViTSpatialSpectral architecture from src/ with a
lightweight linear reconstruction head (SimMIM approach).

Usage:
    python -m embedding_creation.pretrain                      # full run
    python -m embedding_creation.pretrain --test-run            # quick test (2 epochs, 5 granules)
    python -m embedding_creation.pretrain --epochs 50 --batch-size 2
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
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from einops import rearrange

# Add project root so we can import from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vit_spatial_spectral import ViTSpatialSpectral
from embedding_creation.dataset import AvirisSSLDataset


def resolve_res(cfg, res):
    """Replace {res} placeholder in config paths with actual resolution."""
    res_str = str(res)
    cfg['data']['labeled_nc_dir'] = cfg['data']['labeled_nc_dir'].replace('{res}', res_str)
    cfg['data']['ssl_nc_dir'] = cfg['data']['ssl_nc_dir'].replace('{res}', res_str)
    return cfg


# ──────────────────────────────────────────────────────────
# Model: Masked Spatial-Spectral Transformer (SimMIM style)
# ──────────────────────────────────────────────────────────

class MaskedSST(nn.Module):
    """
    Wraps ViTSpatialSpectral with:
      - A learnable mask token (replaces masked embeddings)
      - A single linear reconstruction head (SimMIM)
      - Blockwise spatial masking strategy
    """
    def __init__(self, num_channels, dim, depth, heads, mlp_dim,
                 spectral_patch_size, spatial_patch_size, image_size,
                 mask_ratio=0.6, mask_block_size=4,
                 dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.spectral_patch_size = spectral_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.image_size = image_size

        self.num_spectral_patches = num_channels // spectral_patch_size
        spatial_grid = image_size // spatial_patch_size
        self.num_spatial_patches = spatial_grid ** 2
        self.spatial_grid = spatial_grid
        self.num_patches = self.num_spectral_patches * self.num_spatial_patches
        self.pixels_per_patch = spectral_patch_size * spatial_patch_size * spatial_patch_size

        # Encoder (this gets transferred to fine-tuning)
        num_spec_groups = num_channels // spectral_patch_size
        self.encoder = ViTSpatialSpectral(
            image_size=image_size,
            spatial_patch_size=spatial_patch_size,
            spectral_patch_size=spectral_patch_size,
            num_classes=1,  # dummy -- we don't use mlp_head
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=num_channels,
            spectral_pos=list(range(num_spec_groups)),
            spectral_pos_embed=True,
            blockwise_patch_embed=True,
            spectral_only=False,
            pixelwise=False,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Reconstruction head (single linear layer — SimMIM)
        self.decoder = nn.Linear(dim, self.pixels_per_patch)

    def _generate_spatial_block_mask(self, batch_size, device):
        """
        Generate blockwise spatial mask. All spectral groups at a masked
        spatial location are masked together (prevents trivial spectral copying).
        """
        grid = self.spatial_grid  # e.g., 16
        block = self.mask_block_size  # e.g., 4
        block_grid = grid // block  # e.g., 4

        # How many blocks to mask
        num_blocks = block_grid * block_grid
        num_mask = int(num_blocks * self.mask_ratio)

        # Generate block-level mask
        noise = torch.rand(batch_size, num_blocks, device=device)
        ids = torch.argsort(noise, dim=1)
        block_mask = torch.zeros(batch_size, num_blocks, device=device)
        block_mask.scatter_(1, ids[:, :num_mask], 1.0)

        # Expand block mask to pixel-level spatial mask: (B, grid, grid)
        block_mask = block_mask.reshape(batch_size, block_grid, block_grid)
        spatial_mask = block_mask.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)
        # Flatten to (B, num_spatial_patches)
        spatial_mask = spatial_mask.reshape(batch_size, -1)

        # Repeat across all spectral groups: (B, num_spectral * num_spatial)
        # Token order in ViTSpatialSpectral: (c, h, w) -> groups first, spatial second
        # So token i = spectral_group * num_spatial + spatial_idx
        full_mask = spatial_mask.unsqueeze(1).expand(-1, self.num_spectral_patches, -1)
        full_mask = full_mask.reshape(batch_size, self.num_patches)

        return full_mask

    def forward(self, x):
        """
        x: (B, C, H, W) — raw 370-band hyperspectral patch
        Returns: (reconstructed_patches, mask, targets)
        """
        B = x.shape[0]

        # 1. Embed patches into tokens
        tokens = self.encoder.to_patch_embedding(x)  # (B, N, D)

        # 2. Add positional encoding
        if self.encoder.spectral_pos_embed:
            pos_embed = self.encoder.get_pos_embeddings()
        else:
            pos_embed = self.encoder.pos_embedding[:, :tokens.shape[1]]
        tokens = tokens + pos_embed
        tokens = self.encoder.dropout(tokens)

        # 3. Generate spatial-block mask
        mask = self._generate_spatial_block_mask(B, x.device)  # (B, N)

        # 4. Replace masked tokens with mask_token
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
        tokens = tokens * (1 - mask_expanded) + self.mask_token * mask_expanded

        # 5. Factorized spatial-spectral transformer
        tokens = self.encoder.spatial_spectral_transformer(tokens)  # (B, N, D)

        # 6. Reconstruct pixel values for all tokens
        reconstructed = self.decoder(tokens)  # (B, N, pixels_per_patch)

        # 7. Build reconstruction targets
        targets = rearrange(
            x,
            'b (c p0) (h p1) (w p2) -> b (c h w) (p0 p1 p2)',
            p0=self.spectral_patch_size,
            p1=self.spatial_patch_size,
            p2=self.spatial_patch_size,
        )

        return reconstructed, mask, targets

    def compute_loss(self, reconstructed, mask, targets):
        """L1 loss on masked tokens only."""
        loss = (reconstructed - targets).abs()
        loss = loss.mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss


# ──────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────

def pretrain(args):
    # Load config
    config_path = os.path.join(PROJECT_ROOT, "embedding_creation", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply test-run overrides
    if args.test_run:
        print("\n[!] TEST RUN MODE — minimal data, tiny epochs\n")
        cfg['pretrain']['epochs'] = cfg['test_run']['pretrain_epochs']
        cfg['pretrain']['batch_size'] = cfg['test_run']['batch_size']
        cfg['pretrain']['patches_per_granule'] = cfg['test_run']['patches_per_granule']
        max_granules = cfg['test_run']['max_granules']
    else:
        max_granules = args.max_granules

    # CLI overrides
    if args.epochs:
        cfg['pretrain']['epochs'] = args.epochs
    if args.batch_size:
        cfg['pretrain']['batch_size'] = args.batch_size

    # Resolution
    res = args.res or cfg['data']['default_res']
    cfg = resolve_res(cfg, res)
    print(f"[*] Resolution: {res}m")

    epochs = cfg['pretrain']['epochs']
    batch_size = cfg['pretrain']['batch_size']
    lr = cfg['pretrain']['lr']
    fp16 = cfg['pretrain']['fp16']
    num_workers = cfg['pretrain']['num_workers']
    ckpt_dir = os.path.join(PROJECT_ROOT, cfg['pretrain']['checkpoint_dir'])
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    if device.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name()}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Dataset ──
    # Read from BOTH the existing labeled granules AND the SSL-only archive
    labeled_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'])
    ssl_dir = os.path.join(PROJECT_ROOT, cfg['data']['ssl_nc_dir'])
    os.makedirs(ssl_dir, exist_ok=True)

    nc_dirs = [labeled_dir, ssl_dir]
    dir_summary = [f"{d} ({'exists' if os.path.isdir(d) else 'missing'})" for d in nc_dirs]
    print(f"[*] Loading SSL dataset from:")
    for s in dir_summary:
        print(f"    {s}")

    dataset = AvirisSSLDataset(
        nc_dirs=nc_dirs,
        patch_size=cfg['data']['patch_size'],
        num_bands=cfg['data']['num_bands'],
        bands_to_trim=cfg['data']['bands_to_trim'],
        patches_per_granule=cfg['pretrain']['patches_per_granule'],
        max_granules=max_granules,
        augment=True,
    )
    print(f"[OK] {len(dataset)} patches from {len(dataset.granule_meta)} granules")

    if len(dataset) == 0:
        print("[!] No data found. Check nc_dir path.")
        return

    # Compute or load band stats
    stats_path = os.path.join(ckpt_dir, "pretrain_band_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        dataset.band_mean = torch.tensor(stats['mean']).reshape(cfg['data']['num_bands'], 1, 1)
        dataset.band_std = torch.tensor(stats['std']).reshape(cfg['data']['num_bands'], 1, 1)
        print(f"[*] Loaded band stats from {stats_path}")
    else:
        mean, std = dataset.compute_band_stats(max_samples=min(2000, len(dataset)))
        with open(stats_path, 'w') as f:
            json.dump({'mean': mean.flatten().tolist(), 'std': std.flatten().tolist()}, f)
        print(f"[OK] Band stats saved to {stats_path}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)

    # ── Model ──
    model = MaskedSST(
        num_channels=cfg['data']['num_bands'],
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        heads=cfg['model']['heads'],
        mlp_dim=cfg['model']['mlp_dim'],
        spectral_patch_size=cfg['model']['spectral_patch_size'],
        spatial_patch_size=cfg['model']['spatial_patch_size'],
        image_size=cfg['data']['patch_size'],
        mask_ratio=cfg['pretrain']['mask_ratio'],
        mask_block_size=cfg['pretrain']['mask_block_size'],
        dropout=cfg['model']['dropout'],
        emb_dropout=cfg['model']['emb_dropout'],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model: {num_params:,} parameters ({num_params * 4 / 1024**2:.1f} MB fp32)")
    print(f"    Tokens per sample: {model.num_patches} "
          f"({model.num_spectral_patches} spectral x {model.num_spatial_patches} spatial)")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=cfg['pretrain']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler = GradScaler(enabled=fp16)

    # ── Warmup ──
    warmup_epochs = cfg['pretrain']['warmup_epochs']

    # ── Training ──
    print(f"\n{'='*60}")
    print(f"  AVIRIS-NG SimMIM Pre-training")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  Mask: {cfg['pretrain']['mask_ratio']*100:.0f}%  |  "
          f"Block: {cfg['pretrain']['mask_block_size']}x{cfg['pretrain']['mask_block_size']}  |  "
          f"FP16: {fp16}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        # Warmup: linear ramp for first N epochs
        if epoch <= warmup_epochs:
            warmup_lr = lr * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()

            with autocast(enabled=fp16):
                reconstructed, mask, targets = model(batch)
                loss = model.compute_loss(reconstructed, mask, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch > warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time

        print(f"[Epoch {epoch:3d}/{epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.0f}s")

        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.encoder.state_dict(),
                'mask_token': model.mask_token.data,
                'decoder_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': cfg,
                'band_mean': dataset.band_mean.flatten().tolist(),
                'band_std': dataset.band_std.flatten().tolist(),
            }, os.path.join(ckpt_dir, "pretrained_aviris_best.pth"))

        if epoch % cfg['pretrain']['checkpoint_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.encoder.state_dict(),
                'loss': avg_loss,
                'config': cfg,
                'band_mean': dataset.band_mean.flatten().tolist(),
                'band_std': dataset.band_std.flatten().tolist(),
            }, os.path.join(ckpt_dir, f"pretrained_aviris_ep{epoch}.pth"))

    total_time = time.time() - start_time
    print(f"\n[OK] Pre-training complete. Best loss: {best_loss:.4f}")
    print(f"     Total time: {total_time/60:.1f} min")
    print(f"     Checkpoint: {ckpt_dir}/pretrained_aviris_best.pth")

    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVIRIS-NG SimMIM Pre-training")
    parser.add_argument("--test-run", action="store_true",
                        help="Minimal run: 2 epochs, 5 granules, batch 2")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--max-granules", type=int, help="Limit number of granules")
    parser.add_argument("--res", type=int, help="Resolution in meters (default: from config, 30)")
    args = parser.parse_args()
    pretrain(args)
