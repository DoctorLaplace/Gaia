import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bioscape_dataset import BioScapeNetCDFDataset
from src.train_production import GaiaTransferModel

def quick_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")
    
    # 1. Path Configuration
    checkpoint_path = "checkpoints/gaia_bioscape_best.pth"
    val_sites_csv = "reports/validation_sites_used.csv"
    nc_dir = "data/bioscape"
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] Checkpoint not found: {checkpoint_path}")
        return
    
    # 2. Load Model
    model = GaiaTransferModel(num_targets=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"[✔] Loaded Gaia Model from {checkpoint_path}")
    
    # 4. Reconstruct the exact 0.46 R2 Split
    nc_paths = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')])
    richness_df = pd.read_csv("data/bioscape/western_cape_site_embeddings.csv").rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    
    # Pre-loading (re-implement identical mapping logic)
    mappings = []
    for nc in tqdm(nc_paths, desc="Mapping Sites (Deterministic)"):
        try:
            ds = BioScapeNetCDFDataset(nc, richness_csv=None)
            for _, row in richness_df.iterrows():
                easting, northing = ds.transformer.transform(row['lon'], row['lat'])
                x_idx = int((easting - ds.origin_x) / ds.pixel_w)
                y_idx = int((northing - ds.origin_y) / ds.pixel_h)
                p = 8 # patch_size // 2
                if y_idx >= p and y_idx < ds.height - p and x_idx >= p and x_idx < ds.width - p:
                    mappings.append((nc, row['lat'], row['lon'], row['richness'], row.get('site_id', f"Site_{row['lat']}_{row['lon']}")))
            del ds
        except: continue
        
    indices = np.arange(len(mappings))
    from sklearn.model_selection import train_test_split
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    results = []
    print(f"[*] Accurately evaluating {len(val_idx)} validation sites from the 0.46 R2 run...")
    
    for idx in tqdm(val_idx, desc="Evaluating Sites"):
        nc_path, lat, lon, richness, site_id = mappings[idx]
        try:
            ds = BioScapeNetCDFDataset(nc_path, richness_csv=None)
            patch = ds.get_patch_at_latlon(lat, lon)
            if patch is not None:
                input_tensor = patch.unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(input_tensor).item()
                results.append({'site_id': site_id, 'actual': richness, 'predicted': pred, 'residual': richness - pred})
            del ds
        except: continue
            
    # 5. Summary and Metrics
    df = pd.DataFrame(results)
    if df.empty:
        print("[!] No predictions were made.")
        return
        
    r2 = r2_score(df['actual'], df['predicted'])
    print(f"\n" + "="*40)
    print(f"FINAL QUICK-CHECK RESULTS")
    print(f"="*40)
    print(f"Samples: {len(df)}")
    print(f"Validation R² Score: {r2:.4f}")
    print(f"Mean Abs Error: {np.abs(df['residual']).mean():.4f}")
    
    print("\n[Sample Predictions]")
    print(df[['site_id', 'actual', 'predicted']].head(10))
    
    # Save results
    df.to_csv("reports/evaluation_results.csv", index=False)
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(df['actual'], df['predicted'], alpha=0.5, color='teal')
    plt.plot([df['actual'].min(), df['actual'].max()], [df['actual'].min(), df['actual'].max()], 'r--')
    plt.title(f'Gaia Richness Predictions (R²: {r2:.4f})')
    plt.xlabel('Actual Richness')
    plt.ylabel('Predicted Richness')
    plt.grid(True)
    plt.savefig("reports/evaluation_plot.png")
    print(f"\n[✔] Evaluation plot saved to reports/evaluation_plot.png")

if __name__ == "__main__":
    quick_check()
