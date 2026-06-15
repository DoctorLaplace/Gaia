"""
AVIRIS-NG Patch and Attention Patch Visualizer

Generates a three-panel plot showing how a hyperspectral flightline (map) is:
1. Cropped into 32x32 training/pre-training patches.
2. Divided into 4x4 spatial attention patches (tokens).
3. Sliced spectral-wise into a 3D cube with missing vertical chunks (60% masking).

Output:
  - embedding_creation/checkpoints/patch_demarcation_demo.png
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def get_rgb(cube, r_idx=60, g_idx=40, b_idx=20):
    """Normalize bands to [0,1] for RGB display."""
    rgb = cube[[r_idx, g_idx, b_idx], :, :]
    
    # Clip extreme values and normalize channels individually
    for c in range(3):
        ch = rgb[c]
        # Ignore NoData
        valid_pixels = ch[ch > -9000]
        if len(valid_pixels) > 0:
            lo, hi = np.percentile(valid_pixels, [2, 98])
            if hi - lo > 1e-6:
                rgb[c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
            else:
                rgb[c] = 0.0
        else:
            rgb[c] = 0.0
    return np.transpose(rgb, (1, 2, 0))

def main():
    # 1. Locate NetCDF file
    nc_dir = os.path.join(PROJECT_ROOT, "data", "bioscape", "30m")
    if not os.path.exists(nc_dir):
        print(f"[!] Data directory not found: {nc_dir}")
        return

    nc_files = [f for f in os.listdir(nc_dir) if f.endswith('.nc')]
    if not nc_files:
        print("[!] No NetCDF files found in 30m directory.")
        return

    nc_path = os.path.join(nc_dir, sorted(nc_files)[0])
    print(f"[*] Loading flightline from: {nc_path}")

    # 2. Extract a valid 128x128 region
    with h5py.File(nc_path, 'r') as f:
        if 'reflectance/reflectance' in f:
            cube_ds = f['reflectance/reflectance']
        else:
            cube_ds = f['reflectance']
        
        bands, h, w = cube_ds.shape
        print(f"    Flightline shape: {bands} bands, {h}x{w} spatial pixels")

        # Scan for a valid 128x128 spatial window without NoData
        crop_size = 128
        found = False
        y_start, x_start = 0, 0
        
        # Search starting from center to find interesting ground features
        for y in range(h // 4, h - crop_size, 32):
            for x in range(w // 4, w - crop_size, 32):
                window = cube_ds[0, y:y+crop_size, x:x+crop_size]
                if (window > -9000).all() and (window > 0.01).any():
                    y_start, x_start = y, x
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Fallback to absolute center
            y_start = max(0, h // 2 - crop_size // 2)
            x_start = max(0, w // 2 - crop_size // 2)
            print("[*] Could not find fully valid window. Falling back to center.")

        print(f"[*] Extracting 128x128 region at y:[{y_start}:{y_start+crop_size}], x:[{x_start}:{x_start+crop_size}]")
        raw_cube = cube_ds[:, y_start:y_start+crop_size, x_start:x_start+crop_size]

    # Clean NoData values
    raw_cube[raw_cube <= -9000] = 0.0
    if raw_cube.max() > 20:
        # Scale down if scaled reflectance (e.g. 10000)
        raw_cube = raw_cube / 10000.0

    # 3. Create RGB composites
    full_rgb = get_rgb(raw_cube)

    # Extract a single 32x32 patch from the center of our 128x128 block
    patch_y_offset = 32
    patch_x_offset = 64
    patch_rgb = full_rgb[patch_y_offset:patch_y_offset+32, patch_x_offset:patch_x_offset+32]

    # 4. Generate Visualization Plot
    fig = plt.figure(figsize=(18, 6.5))
    plt.suptitle("Gaia Hyperspectral Block Masking and Tokenization", fontsize=16, fontweight='bold', y=0.96)

    # -------------------------------------------------------------
    # Subplot 1: Map division into 32x32 Patches
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(full_rgb)
    ax1.set_title("1. Map Sliced into Patches (32x32 pixels)", fontsize=12, fontweight='bold')
    
    # Draw 32x32 grid lines in red, aligned with pixel boundaries (-0.5 offset)
    for y in range(0, crop_size + 1, 32):
        ax1.axhline(y - 0.5, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    for x in range(0, crop_size + 1, 32):
        ax1.axvline(x - 0.5, color='red', linestyle='--', linewidth=1.2, alpha=0.8)

    # Highlight the zoom patch
    rect = Rectangle((patch_x_offset - 0.5, patch_y_offset - 0.5), 32, 32, linewidth=3.0, edgecolor='yellow', facecolor='none', linestyle='-')
    ax1.add_patch(rect)
    ax1.text(patch_x_offset + 2, patch_y_offset + 6, "ZOOM TARGET", color='yellow', fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))

    # Label the individual crops
    for i in range(crop_size // 32):
        for j in range(crop_size // 32):
            ax1.text(j*32 + 3, i*32 + 28, f"Patch ({i},{j})", color='white', fontsize=8, alpha=0.9, fontweight='semibold')

    ax1.set_xlabel("Easting (X pixels)", fontsize=10)
    ax1.set_ylabel("Northing (Y pixels)", fontsize=10)
    ax1.grid(False)

    # -------------------------------------------------------------
    # Subplot 2: Patch division into 4x4 Attention Patches (Tokens)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(patch_rgb)
    ax2.set_title("2. Zoom Target: Spatial Tokens (4x4 pixels)", fontsize=12, fontweight='bold')

    # Draw 4x4 token grid lines in yellow, aligned with pixel boundaries (-0.5 offset)
    for y in range(0, 32 + 1, 4):
        ax2.axhline(y - 0.5, color='yellow', linestyle=':', linewidth=1.0, alpha=0.9)
    for x in range(0, 32 + 1, 4):
        ax2.axvline(x - 0.5, color='yellow', linestyle=':', linewidth=1.0, alpha=0.9)

    # Highlight one single spatial token (4x4 pixels)
    token_y, token_x = 12, 16
    token_rect = Rectangle((token_x - 0.5, token_y - 0.5), 4, 4, linewidth=2.0, edgecolor='cyan', facecolor='none')
    ax2.add_patch(token_rect)
    ax2.text(token_x + 5, token_y + 3, "Spatial Token\n(4x4 pixels)", color='cyan', fontsize=9, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))

    ax2.set_xlabel("Local X pixels", fontsize=10)
    ax2.set_ylabel("Local Y pixels", fontsize=10)
    ax2.grid(False)

    # -------------------------------------------------------------
    # Subplot 3: 3D Masked Hyperspectral Cube (SimMIM Masking Demo)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title("3. SimMIM Blockwise Masking (60% Masked)", fontsize=12, fontweight='bold')

    # Set up grid dimensions: 8x8 spatial tokens, Z dimension represents 370 bands (represented as height 40)
    grid_size = 8
    height = 40
    dz_step = 10  # Stack heights

    # Generate a fixed block mask of 60% masked columns
    np.random.seed(42)  # Consistent layout
    mask = np.random.rand(grid_size, grid_size) < 0.6  # 60% True (masked)

    # Color stack for unmasked active columns
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd'] # VIS-Blue, NIR-Green, SWIR1-Red, SWIR2-Purple

    # Iterate over the 8x8 spatial grid
    for x in range(grid_size):
        for y in range(grid_size):
            if mask[x, y]:
                # MASKED COLUMN: Draw a semi-transparent gray wireframe chunk
                # This demonstrates the "missing vertical chunk" spanning all bands
                ax3.bar3d(x, y, 0, 0.8, 0.8, height, 
                          color='gray', edgecolor='#7f7f7f', 
                          alpha=0.04, linestyle='--')
            else:
                # UNMASKED COLUMN: Draw active data stack representing spectral bands
                for i in range(4):
                    z_base = i * dz_step
                    ax3.bar3d(x, y, z_base, 0.8, 0.8, dz_step, 
                              color=colors[i], edgecolor='black', 
                              alpha=0.8, linewidth=0.2)

    # Adjust axis viewing angles and labels
    ax3.set_xlim(-1, 8)
    ax3.set_ylim(-1, 8)
    ax3.set_zlim(0, 45)
    
    ax3.set_xlabel("Spatial X (Tokens)")
    ax3.set_ylabel("Spatial Y (Tokens)")
    ax3.set_zlabel("Spectral Band Axis")
    
    # Hide axes tick labels for a cleaner, premium design feel
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    
    # Perspective angles for a clear view of the "cutout" columns
    ax3.view_init(elev=28, azim=-45)

    # 5. Save the output image
    output_dir = os.path.join(PROJECT_ROOT, "embedding_creation", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "patch_demarcation_demo.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Demarcation visualization saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
