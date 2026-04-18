"""
AVIRIS-NG Reconstruction Visualization Tool (SimMIM)

This tool generates visual comparisons between Original, Masked, and Reconstructed 
hyperspectral patches to verify the foundation model's learning progress.

Outputs:
  - reconstruction_rgb.png: True color and False Color Infrared comparisons.
  - reconstruction_spectra.png: Spectral signature plots (Original vs Reconstructed).
  - reconstruction_sweep.gif: Animated sweep through the 3D spectral cube.

Usage:
    python -m embedding_creation.visualize_reconstruction
    python -m embedding_creation.visualize_reconstruction --ckpt paths/to/checkpoint.pth
"""

import os
import sys
import yaml
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
    """Pick 3 bands and normalize for RGB display."""
    rgb = cube[[r_idx, g_idx, b_idx], :, :]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
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

    # 2. Setup Dataset
    labeled_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'])
    ssl_dir = os.path.join(PROJECT_ROOT, cfg['data']['ssl_nc_dir'])
    dataset = AvirisSSLDataset(
        nc_dirs=[labeled_dir, ssl_dir],
        patch_size=cfg['data']['patch_size'],
        num_bands=cfg['data']['num_bands'],
        patches_per_granule=5,
        augment=False
    )

    # Load stats if available
    stats_path = os.path.join(PROJECT_ROOT, cfg['pretrain']['checkpoint_dir'], "pretrain_band_stats.json")
    if os.path.exists(stats_path):
        import json
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        dataset.band_mean = torch.tensor(stats['mean']).reshape(-1, 1, 1)
        dataset.band_std = torch.tensor(stats['std']).reshape(-1, 1, 1)
        print(f"{Colors.OKGREEN}[OK] Loaded normalization stats.{Colors.ENDC}")

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

    ckpt_path = os.path.join(PROJECT_ROOT, args.ckpt)
    if not os.path.exists(ckpt_path):
        print(f"{Colors.FAIL}[!] Checkpoint not found at {ckpt_path}{Colors.ENDC}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.encoder.load_state_dict(checkpoint['model_state_dict'])
    model.mask_token.data = checkpoint['mask_token']
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()
    print(f"{Colors.OKGREEN}[OK] Model loaded from epoch {checkpoint.get('epoch', '?')}{Colors.ENDC}")

    # 4. Generate Reconstructions
    loader = DataLoader(dataset, batch_size=args.samples, shuffle=True)
    batch = next(iter(loader)).to(device)

    with torch.no_grad():
        reconstructed_flat, mask, targets_flat = model(batch)
        
        # Un-flatten back to cube (B, C, H, W)
        B, N, _ = reconstructed_flat.shape
        p0, p1, p2 = model.spectral_patch_size, model.spatial_patch_size, model.spatial_patch_size
        C, H, W = cfg['data']['num_bands'], cfg['data']['patch_size'], cfg['data']['patch_size']
        h_grid = H // p1
        w_grid = W // p2
        c_grid = C // p0

        reconstructed = rearrange(
            reconstructed_flat, 
            'b (c h w) (p0 p1 p2) -> b (c p0) (h p1) (w p2)',
            c=c_grid, h=h_grid, w=w_grid, p0=p0, p1=p1, p2=p2
        )
        targets = batch # The original normalized patch

    # Denormalize for visualization
    def denorm(x):
        return (x * dataset.band_std.to(device) + dataset.band_mean.to(device)).cpu().numpy()

    orig_np = denorm(targets)
    pred_np = denorm(reconstructed)

    # 5. Output 1: Static Comparison (RGB + CIR)
    fig, axes = plt.subplots(args.samples, 3, figsize=(12, 4 * args.samples))
    if args.samples == 1: axes = [axes]
    
    # Indices for AVIRIS-NG (Roughly 640nm, 550nm, 470nm for RGB)
    # NIR is around band 150+
    for i in range(args.samples):
        # Original
        axes[i][0].imshow(get_rgb(orig_np[i]))
        axes[i][0].set_title("Original RGB")
        axes[i][0].axis('off')

        # Masked Preview (show original with mask opacity)
        # We need to reshape mask to (H, W)
        spatial_mask = rearrange(mask[i], '(c h w) -> c h w', c=c_grid, h=h_grid, w=w_grid)[0]
        spatial_mask_up = spatial_mask.repeat_interleave(p1, dim=0).repeat_interleave(p2, dim=1).cpu().numpy()
        masked_view = get_rgb(orig_np[i]) * (1 - spatial_mask_up[:, :, None])
        axes[i][1].imshow(masked_view)
        axes[i][1].set_title("Masked Input")
        axes[i][1].axis('off')

        # Reconstructed
        axes[i][2].imshow(get_rgb(pred_np[i]))
        axes[i][2].set_title("Reconstructed RGB")
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig("reconstruction_rgb.png")
    print(f"{Colors.OKGREEN}[OK] Saved reconstruction_rgb.png{Colors.ENDC}")

    # 6. Output 2: Spectral Profile (Line Plot)
    plt.figure(figsize=(10, 6))
    for i in range(min(2, args.samples)):
        # Pick a center pixel that was masked
        h_mid, w_mid = H // 2, W // 2
        plt.plot(orig_np[i, :, h_mid, w_mid], label=f"Original (Sample {i})", alpha=0.5, linestyle='--')
        plt.plot(pred_np[i, :, h_mid, w_mid], label=f"Reconstructed (Sample {i})", alpha=0.8)
    
    plt.title("Spectral Signature Comparison (Original vs Reconstructed)")
    plt.xlabel("Band Index")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reconstruction_spectra.png")
    print(f"{Colors.OKGREEN}[OK] Saved reconstruction_spectra.png{Colors.ENDC}")

    # 7. Output 3: Animated Sweep (GIF)
    print(f"{Colors.OKBLUE}[*] Generating animated spectral sweep GIF...{Colors.ENDC}")
    frames = []
    # Pick first sample, select every 10th band for the GIF
    sample_idx = 0
    for b in range(0, C, 10):
        f_orig = orig_np[sample_idx, b]
        f_pred = pred_np[sample_idx, b]
        
        # Normalize for display
        vmax = max(f_orig.max(), f_pred.max(), 1e-8)
        f_orig = (f_orig / vmax * 255).astype(np.uint8)
        f_pred = (f_pred / vmax * 255).astype(np.uint8)
        
        # Concatenate horizontally
        combined = np.hstack([f_orig, np.ones((H, 5), dtype=np.uint8)*128, f_pred])
        
        # Add a text header safely
        frame_canvas = np.zeros((H + 30, combined.shape[1]), dtype=np.uint8)
        frame_canvas[30:, :] = combined
        
        frames.append(frame_canvas)
    
    imageio.mimsave("reconstruction_sweep.gif", frames, fps=5, loop=0)
    print(f"{Colors.OKGREEN}[OK] Saved reconstruction_sweep.gif{Colors.ENDC}")

if __name__ == "__main__":
    visualize()
