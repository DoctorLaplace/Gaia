"""
GAIA SMART INVENTORY: Map biodiversity site coordinates to EarthData granules.
Generates 'labeled_inventory.txt' containing ONLY high-value flightlines.
"""
import earthaccess
import pandas as pd
import os
from tqdm import tqdm

def generate_labeled_inventory(richness_csv, output_path="labeled_inventory.txt"):
    print("\n" + "="*60)
    print("GAIA SMART INVENTORY: Querying NASA Earthdata for Labeled Granules...")
    print("="*60)
    
    # 1. Load Richness Labels
    df = pd.read_csv(richness_csv)
    # Handle different column namings
    lat_col = 'Latitude' if 'Latitude' in df.columns else 'lat'
    lon_col = 'Longitude' if 'Longitude' in df.columns else 'lon'
    
    sites = df[[lat_col, lon_col]].values
    print(f"[*] Loaded {len(sites)} biodiversity sites from {richness_csv}")

    # 2. Login to EarthData
    earthaccess.login(persist=True)
    
    # 3. Search for ALL available granules
    print("[*] Searching for all granules in 'BioSCape_AVNG_L2B_BRDF_GCFR_2385'...")
    granules = earthaccess.search_data(
        short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385",
        count=5000 
    )
    
    if not granules:
        print("[!] No granules found.")
        return

    print(f"[✔] Found {len(granules)} candidate granules.")

    # 4. Map Sites to Granules
    labeled_granules = set()
    site_coverage = 0
    
    print("[*] Matching {len(sites)} sites against granule footprints...")
    for g in tqdm(granules):
        # Extract bounding box from granule metadata
        # Earthaccess granules have a 'spatial' key
        try:
            # Simplistic check: If site is within the bounding box
            # Usually: g['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['BoundingRectangles']
            bbox = g['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['BoundingRectangles'][0]
            west = bbox['WestBoundingCoordinate']
            east = bbox['EastBoundingCoordinate']
            north = bbox['NorthBoundingCoordinate']
            south = bbox['SouthBoundingCoordinate']
            
            native_id = g['meta']['native-id']
            
            # Check for overlap
            found_site = False
            for lat, lon in sites:
                if south <= lat <= north and west <= lon <= east:
                    labeled_granules.add(native_id)
                    found_site = True
            
            if found_site:
                site_coverage += 1
        except Exception as e:
            # Some granules might have complex polygons instead of boxes
            continue

    # 5. Save Results
    with open(output_path, 'w') as f:
        for gid in sorted(list(labeled_granules)):
            f.write(gid + "\n")
            
    print(f"\n[SUCCESS] Labeled Inventory saved to: {output_path}")
    print(f"[*] Identified {len(labeled_granules)} granules covering your richness sites.")
    print(f"[*] This optimization reduces your total download batch significantly.")
    print("="*60)

if __name__ == "__main__":
    generate_labeled_inventory("data/bioscape/western_cape_site_embeddings.csv")
