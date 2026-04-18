"""
AVIRIS-NG Reconstruction Visualization Tool (SimMIM)

Generates visual comparisons between Original, Masked, and Reconstructed
hyperspectral patches to verify the foundation model's learning progress.

Outputs:
  - reconstruction_rgb.png: True color comparisons (Original | Masked | Composite).
  - reconstruction_spectra.png: Spectral signature plots (Original vs Reconstructed).
  - reconstruction_sweep.gif: Animated sweep through the 3D spectral cube.

Usage:
    python -m embedding_creation.visualize_reconstruction
    python -m embedding_creation.visualize_reconstruction --ckpt path/to/checkpoint.pth
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange
import imageio

# Set project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from embedding_creation.pretrain import MaskedSST, resolve_res
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

def get_rgb(cube, r_idx=60, g_idx=40, b_idx=20):
    """Pick 3 bands and normalize to [0,1] for RGB display."""
    rgb = cube[[r_idx, g_idx, b_idx], :, :]
    # Per-channel normalization for better contrast
    for c in range(3):
        ch = rgb[c]
        lo, hi = np.percentile(ch[ch > 0], [2, 98]) if (ch > 0).any() else (0, 1)
        if hi - lo < 1e-6:
            rgb[c] = 0
        else:
            rgb[c] = (ch - lo) / (hi - lo)
    return rearrange(rgb, 'c h w -> h w c').clip(0, 1)


def visualize():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='embedding_creation/checkpoints/pretrained_aviris_best.pth')
    parser.add_argument('--res', type=int, default=30)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()

    # 1. Load Config
    config_path = os.path.join(PROJECT_ROOT, "embedding_creation", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_res(cfg, args.res)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{Colors.OKBLUE}[*] Using device: {device}{Colors.ENDC}")

    # 2. Setup Dataset (with NoData filtering so we only visualize real data)
    labeled_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'])
    ssl_dir = os.path.join(PROJECT_ROOT, cfg['data']['ssl_nc_dir'])
    dataset = AvirisSSLDataset(
        nc_dirs=[labeled_dir, ssl_dir],
        patch_size=cfg['data']['patch_size'],
        num_bands=cfg['data']['num_bands'],
        patches_per_granule=5,
        augment=False,
        nodata_threshold=cfg['data'].get('nodata_threshold', -9000)
    )

    # Load stats from checkpoint (authoritative source, not the JSON file)
    ckpt_path = os.path.join(PROJECT_ROOT, args.ckpt)
    if not os.path.exists(ckpt_path):
        print(f"{Colors.FAIL}[!] Checkpoint not found at {ckpt_path}{Colors.ENDC}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Use band stats from the checkpoint itself (guaranteed to match the model)
    if 'band_mean' in checkpoint and 'band_std' in checkpoint:
        dataset.band_mean = torch.tensor(checkpoint['band_mean']).reshape(-1, 1, 1)
        dataset.band_std = torch.tensor(checkpoint['band_std']).reshape(-1, 1, 1)
        print(f"{Colors.OKGREEN}[OK] Loaded normalization stats from checkpoint.{Colors.ENDC}")
    else:
        # Fallback to JSON
        stats_path = os.path.join(PROJECT_ROOT, cfg['pretrain']['checkpoint_dir'], "pretrain_band_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            dataset.band_mean = torch.tensor(stats['mean']).reshape(-1, 1, 1)
            dataset.band_std = torch.tensor(stats['std']).reshape(-1, 1, 1)
            print(f"{Colors.WARNING}[*] Loaded stats from JSON (checkpoint had none).{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}[!] No normalization stats found. Cannot denormalize.{Colors.ENDC}")
            return

    # 3. Load Model
    model = MaskedSST(
        num_channels=cfg['data']['num_bands'],
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        heads=cfg['model']['heads'],
        mlp_dim=cfg['model']['mlp_dim'],
        spectral_patch_size=cfg['model']['spectral_patch_size'],
        spatial_patch_size=cfg['model']['spatial_patch_size'],
        image_size=cfg['data']['patch_size']
    ).to(device)

    model.encoder.load_state_dict(checkpoint['model_state_dict'])
    model.mask_token.data = checkpoint['mask_token']
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()
    print(f"{Colors.OKGREEN}[OK] Model loaded from epoch {checkpoint.get('epoch', '?')}{Colors.ENDC}")

    if len(dataset) == 0:
        print(f"{Colors.FAIL}[!] No valid patches found. Check data paths.{Colors.ENDC}")
        return

    # 4. Generate Reconstructions
    loader = DataLoader(dataset, batch_size=args.samples, shuffle=True)
    batch = next(iter(loader)).to(device)

    with torch.no_grad():
        reconstructed_flat, mask, targets_flat = model(batch)

        B, N, _ = reconstructed_flat.shape
        p0 = model.spectral_patch_size
        p1 = model.spatial_patch_size
        p2 = model.spatial_patch_size
        C = cfg['data']['num_bands']
        H = cfg['data']['patch_size']
        W = cfg['data']['patch_size']
        h_grid = H // p1
        w_grid = W // p2
        c_grid = C // p0

        reconstructed = rearrange(
            reconstructed_flat,
            'b (c h w) (p0 p1 p2) -> b (c p0) (h p1) (w p2)',
            c=c_grid, h=h_grid, w=w_grid, p0=p0, p1=p1, p2=p2
        )

    # Denormalize
    band_mean = dataset.band_mean.to(device)
    band_std = dataset.band_std.to(device)

    def denorm(x):
        return (x * band_std + band_mean).cpu().numpy()

    orig_np = denorm(batch)
    pred_np = denorm(reconstructed)

    # Build the spatial mask at pixel level: (B, H, W)
    spatial_mask_pixel = []
    for i in range(B):
        # mask shape: (B, N) where N = c_grid * h_grid * w_grid
        # All spectral groups share the same spatial mask, so just take the first group
        sm = rearrange(mask[i], '(c h w) -> c h w', c=c_grid, h=h_grid, w=w_grid)[0]
        sm_up = sm.repeat_interleave(p1, dim=0).repeat_interleave(p2, dim=1).cpu().numpy()
        spatial_mask_pixel.append(sm_up)

    # Build COMPOSITE: original where visible, reconstruction where masked
    composite_np = np.copy(orig_np)
    for i in range(B):
        mask_3d = spatial_mask_pixel[i][None, :, :]  # (1, H, W)
        composite_np[i] = orig_np[i] * (1 - mask_3d) + pred_np[i] * mask_3d

    # ── Output 1: Static Comparison ──
    fig, axes = plt.subplots(args.samples, 3, figsize=(12, 4 * args.samples))
    if args.samples == 1:
        axes = [axes]

    for i in range(args.samples):
        # Column 1: Original
        axes[i][0].imshow(get_rgb(orig_np[i]))
        axes[i][0].set_title("Original RGB")
        axes[i][0].axis('off')

        # Column 2: Masked Input (what the model actually sees)
        masked_view = get_rgb(orig_np[i]) * (1 - spatial_mask_pixel[i][:, :, None])
        axes[i][1].imshow(masked_view)
        axes[i][1].set_title("Masked Input (60%)")
        axes[i][1].axis('off')

        # Column 3: Composite (original unmasked + reconstructed masked)
        axes[i][2].imshow(get_rgb(composite_np[i]))
        axes[i][2].set_title("Reconstructed (Composite)")
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig("reconstruction_rgb.png", dpi=150)
    plt.close()
    print(f"{Colors.OKGREEN}[OK] Saved reconstruction_rgb.png{Colors.ENDC}")

    # ── Output 2: Spectral Profile ──
    plt.figure(figsize=(10, 6))
    for i in range(min(2, args.samples)):
        # Find a pixel that WAS masked so we can compare the model's actual prediction
        masked_ys, masked_xs = np.where(spatial_mask_pixel[i] > 0.5)
        if len(masked_ys) > 0:
            # Pick the center-most masked pixel
            center_dist = (masked_ys - H//2)**2 + (masked_xs - W//2)**2
            best = np.argmin(center_dist)
            py, px = masked_ys[best], masked_xs[best]
        else:
            py, px = H // 2, W // 2

        orig_spec = orig_np[i, :, py, px]
        pred_spec = pred_np[i, :, py, px]

        plt.plot(orig_spec, label=f"Original (Sample {i}, px [{py},{px}])", alpha=0.7, linestyle='--')
        plt.plot(pred_spec, label=f"Reconstructed (Sample {i})", alpha=0.8)

    plt.title("Spectral Signature: Masked Pixel (Original vs Reconstructed)")
    plt.xlabel("Band Index")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reconstruction_spectra.png", dpi=150)
    plt.close()
    print(f"{Colors.OKGREEN}[OK] Saved reconstruction_spectra.png{Colors.ENDC}")

    # ── Output 3: Animated Sweep (GIF) ──
    print(f"{Colors.OKBLUE}[*] Generating animated spectral sweep GIF...{Colors.ENDC}")
    frames = []
    sample_idx = 0
    for b in range(0, C, 10):
        f_orig = orig_np[sample_idx, b]
        f_comp = composite_np[sample_idx, b]

        # Shared normalization so both sides use the same scale
        lo = min(f_orig.min(), f_comp.min())
        hi = max(f_orig.max(), f_comp.max())
        if hi - lo < 1e-8:
            hi = lo + 1

        f_orig_u8 = ((f_orig - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        f_comp_u8 = ((f_comp - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)

        # Separator column
        sep = np.ones((H, 3), dtype=np.uint8) * 128

        # Concatenate: Original | Composite
        combined = np.hstack([f_orig_u8, sep, f_comp_u8])

        # Add band label header
        header = np.zeros((20, combined.shape[1]), dtype=np.uint8)
        frame = np.vstack([header, combined])
        frames.append(frame)

    imageio.mimsave("reconstruction_sweep.gif", frames, fps=5, loop=0)
    print(f"{Colors.OKGREEN}[OK] Saved reconstruction_sweep.gif (looping){Colors.ENDC}")

    print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}  Visualization Complete!{Colors.ENDC}")
    print(f"  - reconstruction_rgb.png")
    print(f"  - reconstruction_spectra.png")
    print(f"  - reconstruction_sweep.gif")
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")


if __name__ == "__main__":
    visualize()
