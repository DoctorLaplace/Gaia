import earthaccess
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def prioritize_coverage(richness_csv, nc_dir, budget=20):
    print("[*] Accessing NASA Earthdata...")
    earthaccess.login(persist=True)
    
    # 1. Load Sites
    df = pd.read_csv(richness_csv)
    df = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    sites = df[['lat', 'lon']].values
    print(f"[*] Loaded {len(df)} sites.")
    
    # 2. Search for granules
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()
    print(f"[*] Searching for granules in region...")
    results = earthaccess.search_data(
        short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385",
        bounding_box=(min_lon, min_lat, max_lon, max_lat)
    )
    
    # 3. Map Sites to Paths
    # Grouped by flightline base name (first 17 chars of the timestamp section)
    path_to_sites = defaultdict(set)
    path_to_granules = defaultdict(list)
    
    existing_files = [f for f in os.listdir(nc_dir) if f.endswith('.nc')]
    existing_paths = set(f.split('_L2B')[0][:17] for f in existing_files)
    
    print(f"[*] Analyzing {len(results)} granules for path-site mapping...")
    for g in tqdm(results):
        try:
            spatial = g['umm'].get('SpatialExtent', {}).get('HorizontalSpatialDomain', {}).get('Geometry', {})
            boxes = spatial.get('BoundingRectangles', [])
            if not boxes: continue
            
            g_url = g.data_links()[0]
            g_name = g_url.split('/')[-1]
            # Key = angYYYYMMDDtHHMMSS
            path_id = g_name.split('_')[0].split('.')[-1]
            
            if path_id in existing_paths: continue
            
            hit_sites = []
            for box in boxes:
                west, south, east, north = box['WestBoundingCoordinate'], box['SouthBoundingCoordinate'], box['EastBoundingCoordinate'], box['NorthBoundingCoordinate']
                mask = (sites[:, 0] >= south) & (sites[:, 0] <= north) & (sites[:, 1] >= west) & (sites[:, 1] <= east)
                hit_sites.extend(df.index[mask].tolist())
            
            if hit_sites:
                path_to_sites[path_id].update(hit_sites)
                path_to_granules[path_id].append(g_name)
        except:
            continue

    # 4. Rank Paths by Unique Site Count
    sorted_paths = sorted(path_to_sites.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("\n" + "="*40)
    print("TOP 20 FLIGHTLINES BY SITE DENSITY")
    print("="*40)
    
    final_granules = []
    total_sites_covered = 0
    covered_indices = set()
    
    for i, (path_id, site_indices) in enumerate(sorted_paths[:budget]):
        new_sites = site_indices - covered_indices
        print(f"Rank {i+1}: {path_id} | Total Sites: {len(site_indices)} | New: {len(new_sites)}")
        final_granules.extend(path_to_granules[path_id])
        covered_indices.update(site_indices)
        
    print(f"\n[✔] Selected {budget} paths.")
    print(f"[✔] Total Unique Sites to be Covered: {len(covered_indices)}")
    print(f"[✔] Total Granule Segments to Download: {len(final_granules)}")

    # 5. Write to mega_batch.txt
    with open('mega_batch.txt', 'w') as f:
        for g in final_granules:
            f.write(f"{g}\n")
    
    print(f"[✔] Updated mega_batch.txt with prioritized essential files.")

if __name__ == "__main__":
    prioritize_coverage("data/bioscape/western_cape_site_embeddings.csv", "data/bioscape")
