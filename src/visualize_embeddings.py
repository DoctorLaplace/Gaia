import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bioscape_dataset import BioScapeNetCDFDataset
from src.train_production import GaiaTransferModel

def visualize_global_embeddings():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="checkpoints/gaia_bioscape_best.pth")
    parser.add_argument("--samples", type=int, default=1000, help="Total random patches to extract")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path Logic
    nc_dir = args.nc_dir
    if nc_dir is None:
        for candidate in ["data/bioscape/30m_v2", "data/bioscape/30m", "data/bioscape"]:
            if os.path.exists(candidate) and any(f.endswith('.nc') for f in os.listdir(candidate)):
                nc_dir = candidate
                break
    
    if nc_dir is None or not os.path.exists(nc_dir):
        print(f"[!] No valid data directory found.")
        return

    print(f"[*] Analyzing Global Semantic Space in: {nc_dir}")
    print(f"[*] Using Model Checkpoint: {args.ckpt}")

    # 1. Load Model
    model = GaiaTransferModel(num_targets=1).to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("[✔] Loaded fine-tuned weights.")
    else:
        print("[!] No checkpoint found. Visualizing RAW EnMap Foundation.")
    model.eval()

    def get_embedding(x):
        with torch.no_grad():
            tokens = model.encoder.to_patch_embedding(x)
            b, n, _ = tokens.shape
            tokens += model.encoder.pos_embedding[:, :n]
            tokens = model.encoder.dropout(tokens)
            x_feat = model.encoder.spatial_spectral_transformer(tokens)
            if model.encoder.pool == "mean":
                x_feat = x_feat.mean(dim=1)
            else:
                x_feat = x_feat[:, 0]
            return x_feat

    # 2. Collect Random Samples
    nc_paths = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')])
    if not nc_paths:
        print(f"[!] No .nc files found in {nc_dir}")
        return
        
    patches_per_tile = max(1, args.samples // len(nc_paths))
    all_embeddings = []
    
    print(f"[*] Scanning {len(nc_paths)} tiles for embeddings...")
    first_tile = True
    for nc in tqdm(nc_paths):
        if len(all_embeddings) >= args.samples: break
        
        ds = BioScapeNetCDFDataset(nc, richness_csv=None, quiet=True)
        
        if first_tile:
            print(f"\n[DEBUG] Tile: {os.path.basename(nc)}")
            print(f"[DEBUG] Dimensions: {ds.width}x{ds.height}")
            print(f"[DEBUG] Bands: {len(ds.bands)}")
            first_tile = False

        for _ in range(patches_per_tile):
            ry = np.random.randint(20, ds.height - 20)
            rx = np.random.randint(20, ds.width - 20)
            patch_data = ds.get_patch(rx, ry)
            
            if patch_data is not None:
                patch, mask = patch_data
                if patch is not None:
                    emb = get_embedding(patch.unsqueeze(0).to(device))
                    all_embeddings.append(emb.cpu().numpy().flatten())
        del ds

    if not all_embeddings:
        print("[!] No embeddings extracted.")
        return

    X = np.array(all_embeddings)
    print(f"[*] Successfully extracted {len(X)} embeddings.")
    
    # 3. Dimensionality Reduction (t-SNE)
    print(f"[*] Projecting to 2D space...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42)
    X_embedded = tsne.fit_transform(X)

    # 4. Semantic Clustering
    kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X)

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters, cmap='Spectral', alpha=0.8, s=60, edgecolors='white', linewidths=0.5)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Semantic Clusters")
    ax.add_artist(legend1)

    ax.set_title("Gaia Global Embedding Space (EnMap Foundation)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    ax.text(0.02, 0.02, f"Samples: {len(X)}\nData: {os.path.basename(nc_dir)}", 
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    out_path = 'reports/global_semantic_clusters.png'
    os.makedirs('reports', exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[✔] Global Semantic Map saved to {out_path}")

if __name__ == "__main__":
    visualize_global_embeddings()
