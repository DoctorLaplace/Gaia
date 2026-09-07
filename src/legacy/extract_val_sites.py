import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bioscape_dataset import BioScapeNetCDFDataset

def extract_sites():
    nc_dir = "data/bioscape"
    richness_csv = "data/bioscape/western_cape_site_embeddings.csv"
    patch_size = 16
    
    # 1. Re-gather mapping (identical logic to train_production.py)
    nc_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
    richness_df = pd.read_csv(richness_csv).rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    
    mappings = []
    print(f"[*] Mapping sites...")
    for nc in nc_paths:
        try:
            ds = BioScapeNetCDFDataset(nc, richness_csv=None, patch_size=patch_size)
            for _, row in richness_df.iterrows():
                easting, northing = ds.transformer.transform(row['lon'], row['lat'])
                x_idx = int((easting - ds.origin_x) / ds.pixel_w)
                y_idx = int((northing - ds.origin_y) / ds.pixel_h)
                p = patch_size // 2
                if y_idx >= p and y_idx < ds.height - p and x_idx >= p and x_idx < ds.width - p:
                    mappings.append({
                        'nc': os.path.basename(nc),
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'richness': row['richness'],
                        'site_id': row.get('site_id', f"Site_{row['lat']}_{row['lon']}")
                    })
            del ds
        except: continue
        
    indices = np.arange(len(mappings))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    val_sites = [mappings[i] for i in val_idx]
    val_df = pd.DataFrame(val_sites)
    
    print(f"\n[✔] Identified {len(val_df)} Validation Sites.")
    print(val_df[['site_id', 'lat', 'lon', 'richness']].head(20))
    val_df.to_csv("reports/validation_sites_used.csv", index=False)
    print(f"[*] Saved full list to reports/validation_sites_used.csv")

if __name__ == "__main__":
    extract_sites()
