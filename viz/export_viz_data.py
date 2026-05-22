import os
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import sys

# Add project root to path
# This script is in viz/, so project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.bioscape_dataset import BioScapeNetCDFDataset

def export_viz_data(nc_dir=None, mapping_json=None, output_dir=None):
    if output_dir is None:
        # Save to viz/data relative to the script location
        output_dir = os.path.join(os.path.dirname(__file__), "data")
        
    # Determine mapping file
    if mapping_json is None:
        # Prioritize local mapping for faster local dev
        local_m = os.path.join(project_root, "data", "bioscape", "local_mapping.json")
        s3_m = os.path.join(project_root, "data", "bioscape", "s3_mapping.json")
        mapping_json = local_m if os.path.exists(local_m) else s3_m
    
    if not os.path.exists(mapping_json):
        print(f"[!] Mapping file not found: {mapping_json}")
        print("[!] Tip: Run 'python viz/regenerate_mapping.py' first.")
        return

    with open(mapping_json, 'r') as f:
        mappings = json.load(f)
    
    print(f"[*] Exporting {len(mappings)} patches from {os.path.basename(mapping_json)}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We'll store all patches as a single uint8 binary buffer [N, C, H, W]
    all_patches = []
    metadata = []
    
    # Preload evaluation results and extra metadata if available
    eval_csv_path = os.path.join(project_root, "reports", "evaluation_results.csv")
    eval_data = {}
    if os.path.exists(eval_csv_path):
        print(f"[*] Found evaluation results at {eval_csv_path}, merging additional metadata...")
        eval_df = pd.read_csv(eval_csv_path)
        for _, row in eval_df.iterrows():
            k = (round(row['lat'], 5), round(row['lon'], 5))
            eval_data[k] = row.to_dict()
    
    datasets = {}
    
    for nc_path, lat, lon, richness in tqdm(mappings, desc="Fetching Patches"):
        # Resolve path relative to project root if it's not absolute
        full_nc_path = nc_path if os.path.isabs(nc_path) else os.path.join(project_root, nc_path)
        
        if full_nc_path not in datasets:
            try:
                datasets[full_nc_path] = BioScapeNetCDFDataset(full_nc_path, richness_csv=None, patch_size=16, quiet=True)
            except:
                continue
        
        ds = datasets[full_nc_path]
        patch, bounds = ds.get_patch_at_latlon(lat, lon)
        
        if patch is not None:
            # Check for zero-fill (Black squares)
            # Band 50 is used to avoid water vapor absorption gaps (like Band 100)
            zero_ratio = (patch[50] == 0).float().mean()
            if zero_ratio > 0.5:
                continue

            # Normalize each band independently using 2%-98% percentiles for robust visibility
            uint8_patch_list = []
            for b in range(patch.shape[0]):
                band = patch[b].numpy()
                p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
                if p98 - p2 < 1e-6:
                    uint8_band = np.zeros_like(band, dtype=np.uint8)
                else:
                    norm_band = np.clip((band - p2) / (p98 - p2), 0, 1)
                    uint8_band = (norm_band * 255).astype(np.uint8)
                uint8_patch_list.append(uint8_band)
            
            uint8_patch = np.stack(uint8_patch_list) # [200, 16, 16]
            
            all_patches.append(uint8_patch)
            lat_round = round(float(lat), 5)
            lon_round = round(float(lon), 5)
            
            site_meta = {
                "lat": float(lat),
                "lon": float(lon),
                "richness": float(richness),
                "bounds": bounds,
                "nc": os.path.basename(full_nc_path)
            }
            
            # Merge extended metadata
            if (lat_round, lon_round) in eval_data:
                ed = eval_data[(lat_round, lon_round)]
                for k, v in ed.items():
                    if k not in ['lat', 'lon', 'actual', 'nc_path']:
                        # handle NaNs gracefully
                        if pd.isna(v): v = None
                        elif isinstance(v, float) and np.isnan(v): v = None
                        site_meta[k] = v
                        
            metadata.append(site_meta)
            
    if not all_patches:
        print("[!] No valid patches were collected. Check if data files are zero-filled.")
        return

    # Convert to one large binary block
    patches_vol = np.stack(all_patches) # [N, 200, 16, 16]
    patches_vol.tofile(os.path.join(output_dir, "patches.dat"))
    
    # Target wavelengths
    wavelengths = np.linspace(400, 2450, 200).tolist()
    
    # Save metadata as JSON
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump({
            "count": len(metadata),
            "bands": 200,
            "h": 16,
            "w": 16,
            "wavelengths": wavelengths,
            "sites": metadata
        }, f, indent=2)
        
    print(f"[✔] Export Complete! Data saved to {output_dir}")
    print(f"    - Binary: {os.path.join(output_dir, 'patches.dat')} ({patches_vol.nbytes/1024/1024:.1f} MB)")
    print(f"    - Metadata: {os.path.join(output_dir, 'metadata.json')}")

if __name__ == "__main__":
    export_viz_data()
