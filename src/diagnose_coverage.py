import os
import sys
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import yaml

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.bioscape_dataset import BioScapeNetCDFDataset

def diagnose_coverage():
    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    b_cfg = config['bioscape']
    nc_dir = os.path.join(project_root, b_cfg['nc_dir_local'])
    richness_csv = os.path.join(project_root, b_cfg['richness_csv'])
    mapping_cache = os.path.join(project_root, "data", "bioscape", "local_mapping.json")
    
    # 1. Load All Potential Sites
    df = pd.read_csv(richness_csv)
    df = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'richness': 'richness', 'Richness': 'richness'})
    all_sites = set()
    for _, row in df.iterrows():
        all_sites.add((round(row['lat'], 5), round(row['lon'], 5)))
    
    print(f"[*] Total Potential Sites in CSV: {len(all_sites)}")

    # 2. Load Mapped Sites
    if not os.path.exists(mapping_cache):
        print("[!] No mapping cache found. Run regenerate_mapping.py first.")
        return
        
    with open(mapping_cache, 'r') as f:
        mappings = json.load(f)
    
    mapped_sites = set()
    for item in mappings:
        mapped_sites.add((round(item[1], 5), round(item[2], 5)))
    
    print(f"[*] Total Mapped Sites: {len(mapped_sites)}")
    
    missing_sites = all_sites - mapped_sites
    print(f"[*] Total Missing Sites: {len(missing_sites)}")
    
    if not missing_sites:
        print("[✔] No sites are missing!")
        return

    # 3. Analyze Bounding Boxes of all flightlines
    nc_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
    print(f"[*] Analyzing bounding boxes of {len(nc_paths)} flightlines...")
    
    boxes = []
    for nc in tqdm(nc_paths, desc="Scanning BBoxes"):
        try:
            ds = BioScapeNetCDFDataset(nc, richness_csv=None, quiet=True)
            boxes.append(ds.get_bounds())
            ds.close()
        except:
            continue
            
    # 4. For each missing site, check if it falls in ANY box
    out_of_bounds = []
    too_close_to_edge = []
    
    for lat, lon in missing_sites:
        in_any_box = False
        for min_lon, min_lat, max_lon, max_lat in boxes:
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                in_any_box = True
                break
        
        if not in_any_box:
            out_of_bounds.append((lat, lon))
        else:
            too_close_to_edge.append((lat, lon))
            
    print("\n" + "="*50)
    print("  COVERAGE DIAGNOSTIC RESULTS")
    print("="*50)
    print(f"  Out of Bounds (No Flightline):  {len(out_of_bounds)}")
    print(f"  Inside Box but Failed Mapping: {len(too_close_to_edge)}")
    print("="*50)
    
    if out_of_bounds:
        print("\n[*] Sample Out-of-Bounds Sites (Lat, Lon):")
        for lat, lon in list(out_of_bounds)[:5]:
            print(f"  - {lat}, {lon}")
            
    if too_close_to_edge:
        print("\n[*] Sites inside boxes that failed (Edge/Data issues):")
        for lat, lon in list(too_close_to_edge)[:5]:
            print(f"  - {lat}, {lon}")

if __name__ == "__main__":
    diagnose_coverage()
