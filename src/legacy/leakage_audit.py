"""
LEAKAGE AUDIT: Fast check for geographic data pollution.
Instead of retraining 81 models, we analyze the existing trained model's
validation split for geographic overlap with the training split.
"""
import os, sys, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bioscape_dataset import BioScapeNetCDFDataset
from src.train_production import GaiaTransferModel

def audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc_dir = "data/bioscape"
    patch_size = 16

    # 1. Load the trained model
    model = GaiaTransferModel(num_targets=1).to(device)
    model.load_state_dict(torch.load("checkpoints/gaia_bioscape_best.pth", map_location=device, weights_only=True))
    model.eval()
    print("[✔] Loaded gaia_bioscape_best.pth")

    # 2. Reconstruct EXACT mapping (same logic as train_production.py)
    nc_paths = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')])
    richness_df = pd.read_csv("data/bioscape/western_cape_site_embeddings.csv")
    richness_df = richness_df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})

    mappings = []
    for nc in tqdm(nc_paths, desc="Mapping"):
        try:
            ds = BioScapeNetCDFDataset(nc, richness_csv=None, patch_size=patch_size)
            for _, row in richness_df.iterrows():
                easting, northing = ds.transformer.transform(row['lon'], row['lat'])
                x_idx = int((easting - ds.origin_x) / ds.pixel_w)
                y_idx = int((northing - ds.origin_y) / ds.pixel_h)
                p = patch_size // 2
                if y_idx >= p and y_idx < ds.height - p and x_idx >= p and x_idx < ds.width - p:
                    mappings.append((nc, row['lat'], row['lon'], row['richness']))
            del ds
        except: continue

    # 3. Reconstruct EXACT split (same random_state=42)
    indices = np.arange(len(mappings))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # 4. Identify geographic overlap
    train_coords = set()
    for i in train_idx:
        lat, lon = mappings[i][1], mappings[i][2]
        train_coords.add((round(lat, 5), round(lon, 5)))

    clean_val = []  # Sites with NO geographic match in training
    leaked_val = [] # Sites with a geographic match in training

    for i in val_idx:
        nc, lat, lon, richness = mappings[i]
        coord = (round(lat, 5), round(lon, 5))
        entry = (nc, lat, lon, richness, i)
        if coord in train_coords:
            leaked_val.append(entry)
        else:
            clean_val.append(entry)

    print(f"\n{'='*50}")
    print(f"LEAKAGE AUDIT RESULTS")
    print(f"{'='*50}")
    print(f"Total Validation Samples: {len(val_idx)}")
    print(f"  → Geographically UNIQUE (clean):    {len(clean_val)}")
    print(f"  → Geographically OVERLAPPING (risk): {len(leaked_val)}")

    # 5. Run inference on both groups separately
    def evaluate_group(entries, label):
        if not entries:
            print(f"\n[{label}] No samples in this group.")
            return
        preds, actuals = [], []
        for nc, lat, lon, richness, idx in entries:
            try:
                ds = BioScapeNetCDFDataset(nc, richness_csv=None, patch_size=patch_size)
                patch = ds.get_patch_at_latlon(lat, lon)
                if patch is not None:
                    with torch.no_grad():
                        pred = model(patch.unsqueeze(0).to(device)).item()
                    preds.append(pred)
                    actuals.append(richness)
                del ds
            except: continue

        if len(actuals) > 1:
            r2 = r2_score(actuals, preds)
            mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))
            print(f"\n[{label}] Samples: {len(actuals)} | R²: {r2:.4f} | MAE: {mae:.2f}")
        else:
            print(f"\n[{label}] Not enough samples to compute R².")

    evaluate_group(clean_val, "CLEAN (Never-Seen Locations)")
    evaluate_group(leaked_val, "OVERLAPPING (Same GPS in Train)")

    # Also compute combined for reference
    all_val = clean_val + leaked_val
    evaluate_group(all_val, "COMBINED (All Val)")

if __name__ == "__main__":
    audit()
