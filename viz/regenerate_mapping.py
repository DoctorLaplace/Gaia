import os
import sys
import yaml

# Add project root to path
# This script is in viz/, so project root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.train_production import MultiFlightBioScapeDataset

def regenerate_mapping():
    # Load config from project root
    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    b_cfg = config['bioscape']
    nc_dir = os.path.abspath(os.path.join(project_root, b_cfg['nc_dir_local']))
    richness_csv = os.path.abspath(os.path.join(project_root, b_cfg['richness_csv']))
    mapping_cache = os.path.abspath(os.path.join(project_root, "data", "bioscape", "local_mapping.json"))
    
    print(f"[*] Scanning {nc_dir}...")
    if not os.path.exists(nc_dir):
        print(f"[!] Error: Data directory not found at {nc_dir}")
        return

    nc_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
    print(f"[*] Found {len(nc_paths)} local granules.")
    
    # Force regeneration by deleting old cache
    if os.path.exists(mapping_cache):
        print(f"[*] Updating existing mapping: {mapping_cache}")
        os.remove(mapping_cache)
    
    print(f"[*] Mapping sites from {os.path.basename(richness_csv)}...")
    # This automatically writes the cache file in __init__
    dataset = MultiFlightBioScapeDataset(nc_paths, richness_csv, augment=False, cache_path=mapping_cache)
    
    print(f"\n[✔] Mapping Complete: {len(dataset.mappings)} sites found across {len(nc_paths)} granules.")
    print(f"[*] Run 'python src/export_viz_data.py' next to update the visualizer images.")

if __name__ == "__main__":
    regenerate_mapping()
