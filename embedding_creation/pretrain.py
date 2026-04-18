"""
SimMIM Pre-training for AVIRIS-NG Native Spectral Foundation Model

Masked Spatial-Spectral Transformer pre-training on 370 bands.
Uses the existing ViTSpatialSpectral architecture from src/ with a
lightweight linear reconstruction head (SimMIM approach).

Usage:
    python -m embedding_creation.pretrain                      # full run
    python -m embedding_creation.pretrain --test-run            # quick test (2 epochs, 5 granules)
    python -m embedding_creation.pretrain --force-stats         # recompute normalization stats
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from einops import rearrange

# Project Root Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vit_spatial_spectral import ViTSpatialSpectral
from embedding_creation.dataset import AvirisSSLDataset

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

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

        # Encoder
        num_spec_groups = num_channels // spectral_patch_size
        self.encoder = ViTSpatialSpectral(
            image_size=image_size,
            spatial_patch_size=spatial_patch_size,
            spectral_patch_size=spectral_patch_size,
            num_classes=1,
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

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = nn.Linear(dim, self.pixels_per_patch)

    def _generate_spatial_block_mask(self, batch_size, device):
        grid = self.spatial_grid
        block = self.mask_block_size
        block_grid = grid // block
        num_blocks = block_grid * block_grid
        num_mask = int(num_blocks * self.mask_ratio)

        noise = torch.rand(batch_size, num_blocks, device=device)
        ids = torch.argsort(noise, dim=1)
        block_mask = torch.zeros(batch_size, num_blocks, device=device)
        block_mask.scatter_(1, ids[:, :num_mask], 1.0)

        block_mask = block_mask.reshape(batch_size, block_grid, block_grid)
        spatial_mask = block_mask.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)
        spatial_mask = spatial_mask.reshape(batch_size, -1)

        full_mask = spatial_mask.unsqueeze(1).expand(-1, self.num_spectral_patches, -1)
        full_mask = full_mask.reshape(batch_size, self.num_patches)
        return full_mask

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        batch_range = torch.arange(B, device=device)[:, None]

        # 1. Get raw patches BEFORE embedding (these are our reconstruction targets)
        raw_patches = self.encoder.to_patch_embedding.to_patch(x)
        # raw_patches shape: (B, num_spectral_groups, num_spatial_patches, pixels_per_patch)
        raw_patches = rearrange(raw_patches, 'b g n d -> b (g n) d')
        # raw_patches shape: (B, N, pixels_per_patch)

        # 2. Embed patches into tokens
        tokens = self.encoder.to_patch_embedding(x)  # (B, N, D)
        num_patches = tokens.shape[1]

        # 3. Add positional encoding
        if self.encoder.spectral_pos_embed:
            pos_embed = self.encoder.get_pos_embeddings()
        else:
            pos_embed = self.encoder.pos_embedding[:, :num_patches]
        tokens = tokens + pos_embed

        # 4. Generate mask and get masked indices
        mask_bool = self._generate_spatial_block_mask(B, device)  # (B, N) float
        num_masked = int(mask_bool[0].sum().item())

        # Convert bool mask to indices (for indexing like the paper)
        masked_indices = mask_bool.nonzero(as_tuple=False)
        # Reshape to (B, num_masked)
        masked_indices = masked_indices[:, 1].reshape(B, num_masked)

        # 5. Prepare mask tokens with positional encoding (as the paper does)
        mask_tokens = self.mask_token.expand(B, num_patches, -1) + pos_embed

        # 6. Replace masked tokens
        mask_expanded = mask_bool.unsqueeze(-1).bool()  # (B, N, 1)
        tokens = torch.where(mask_expanded, mask_tokens, tokens)

        tokens = self.encoder.dropout(tokens)

        # 7. Run through the factored spatial-spectral transformer
        encoded = self.encoder.spatial_spectral_transformer(tokens)  # (B, N, D)

        # 8. Extract ONLY the masked token representations (like the paper)
        encoded_mask_tokens = encoded[batch_range, masked_indices]  # (B, num_masked, D)

        # 9. Decode ONLY masked tokens to pixel values
        pred_pixel_values = self.decoder(encoded_mask_tokens)  # (B, num_masked, pixels_per_patch)

        # 10. Get the raw pixel targets for ONLY the masked patches
        masked_patches = raw_patches[batch_range, masked_indices]  # (B, num_masked, pixels_per_patch)

        return pred_pixel_values, masked_patches, num_masked, mask_bool

    def compute_loss(self, pred_pixel_values, masked_patches, num_masked):
        """L1 loss on masked tokens only (mean absolute error per pixel)."""
        return F.l1_loss(pred_pixel_values, masked_patches)


# ──────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────

def pretrain(args):
    config_path = os.path.join(PROJECT_ROOT, "embedding_creation", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.test_run:
        cfg['pretrain']['epochs'] = cfg['test_run']['pretrain_epochs']
        cfg['pretrain']['batch_size'] = cfg['test_run']['batch_size']
        cfg['pretrain']['patches_per_granule'] = cfg['test_run']['patches_per_granule']
        max_granules = cfg['test_run']['max_granules']
    else:
        max_granules = args.max_granules

    if args.epochs: cfg['pretrain']['epochs'] = args.epochs
    if args.batch_size: cfg['pretrain']['batch_size'] = args.batch_size

    res = args.res or cfg['data']['default_res']
    cfg = resolve_res(cfg, res)
    print(f"{Colors.OKCYAN}[*] Resolution: {res}m{Colors.ENDC}")

    epochs = cfg['pretrain']['epochs']
    batch_size = cfg['pretrain']['batch_size']
    lr = cfg['pretrain']['lr']
    fp16 = cfg['pretrain']['fp16']
    num_workers = cfg['pretrain']['num_workers']
    ckpt_dir = os.path.join(PROJECT_ROOT, cfg['pretrain']['checkpoint_dir'])
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{Colors.OKBLUE}[*] Device: {device}{Colors.ENDC}")

    labeled_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'])
    ssl_dir = os.path.join(PROJECT_ROOT, cfg['data']['ssl_nc_dir'])
    nc_dirs = [labeled_dir, ssl_dir]

    dataset = AvirisSSLDataset(
        nc_dirs=nc_dirs,
        patch_size=cfg['data']['patch_size'],
        num_bands=cfg['data']['num_bands'],
        bands_to_trim=cfg['data']['bands_to_trim'],
        patches_per_granule=cfg['pretrain']['patches_per_granule'],
        max_granules=max_granules,
        augment=True,
        nodata_threshold=cfg['data']['nodata_threshold']
    )
    print(f"{Colors.OKGREEN}[OK] {len(dataset)} valid patches from {len(dataset.granule_meta)} granules.{Colors.ENDC}")

    if len(dataset) == 0:
        print(f"{Colors.FAIL}[!] No data found.{Colors.ENDC}")
        return

    # Stats handling
    stats_path = os.path.join(ckpt_dir, "pretrain_band_stats.json")
    if os.path.exists(stats_path) and not args.force_stats:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        dataset.band_mean = torch.tensor(stats['mean']).reshape(-1, 1, 1)
        dataset.band_std = torch.tensor(stats['std']).reshape(-1, 1, 1)
        print(f"{Colors.OKBLUE}[*] Loaded band stats from {stats_path}{Colors.ENDC}")
    else:
        mean, std = dataset.compute_band_stats(max_samples=min(2000, len(dataset)))
        with open(stats_path, 'w') as f:
            json.dump({'mean': mean.flatten().tolist(), 'std': std.flatten().tolist()}, f)
        print(f"{Colors.OKGREEN}[OK] Band stats saved to {stats_path}{Colors.ENDC}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)

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
    print(f"{Colors.OKBLUE}[*] Model: {num_params:,} parameters ({num_params * 4 / 1024**2:.1f} MB fp32){Colors.ENDC}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['pretrain']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=fp16)
    warmup_epochs = cfg['pretrain']['warmup_epochs']

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  AVIRIS-NG Cleaned SimMIM Pre-training{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        if epoch <= warmup_epochs:
            warmup_lr = lr * (epoch / warmup_epochs)
            for pg in optimizer.param_groups: pg['lr'] = warmup_lr

        pbar = tqdm(loader, desc=f"{Colors.OKCYAN}Epoch {epoch}/{epochs}{Colors.ENDC}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=fp16):
                pred_pixels, target_pixels, num_masked, mask_bool = model(batch)
                loss = model.compute_loss(pred_pixels, target_pixels, num_masked)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch > warmup_epochs: scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        print(f"{Colors.OKCYAN}[Epoch {epoch:3d}/{epochs}]{Colors.ENDC} Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(ckpt_dir, "pretrained_aviris_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.encoder.state_dict(),
                'mask_token': model.mask_token.data,
                'decoder_state_dict': model.decoder.state_dict(),
                'config': cfg,
                'band_mean': dataset.band_mean.flatten().tolist(),
                'band_std': dataset.band_std.flatten().tolist(),
            }, save_path)
            print(f"    {Colors.OKGREEN}-> New Best Loss: {best_loss:.4f} (Saved){Colors.ENDC}")

    print(f"\n{Colors.OKGREEN}[OK] Pre-training complete.{Colors.ENDC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-granules", type=int)
    parser.add_argument("--res", type=int)
    parser.add_argument('--force-stats', action='store_true')
    args = parser.parse_args()
    pretrain(args)
