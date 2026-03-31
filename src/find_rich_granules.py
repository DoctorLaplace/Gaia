import os
import pandas as pd
import torch
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bioscape_dataset import BioScapeNetCDFDataset

def find_rich_granules(nc_dir, richness_csv):
    df = pd.read_csv(richness_csv)
    df = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'richness': 'richness', 'Richness': 'richness'})
    
    nc_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
    
    results = []
    print(f"[*] Checking {len(nc_paths)} local flightlines for site overlaps...")
    
    for nc in tqdm(nc_paths):
        try:
            # We use a temporary dataset just to check metadata/coverage
            ds = BioScapeNetCDFDataset(nc, richness_csv=None, patch_size=16)
            site_count = 0
            for _, row in df.iterrows():
                easting, northing = ds.transformer.transform(row['lon'], row['lat'])
                x_idx = int((easting - ds.origin_x) / ds.pixel_w)
                y_idx = int((northing - ds.origin_y) / ds.pixel_h)
                p = 8
                if y_idx >= p and y_idx < ds.height - p and x_idx >= p and x_idx < ds.width - p:
                    site_count += 1
            
            if site_count > 0:
                results.append((os.path.basename(nc), site_count))
            del ds
        except:
            continue
            
    print("\n[✔] Identification Complete.")
    if results:
        print("\nFlightlines with ground-truth sites:")
        for name, count in sorted(results, key=lambda x: x[1], reverse=True):
            print(f"  - {name}: {count} sites")
    else:
        print("  - No overlaps found in the local subset.")

if __name__ == "__main__":
    find_rich_granules("data/bioscape", "data/bioscape/western_cape_site_embeddings.csv")
