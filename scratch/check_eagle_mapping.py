import os
import re
import pandas as pd
import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm

def main():
    project_root = r"E:\Git Repositories\Laboratory\Models\Gaia"
    tif_dir = os.path.join(project_root, "data", "eagle", "30m_10nm")
    richness_csv = os.path.join(project_root, "data", "bioscape", "western_cape_site_embeddings.csv")
    
    if not os.path.exists(tif_dir):
        print(f"Directory not found: {tif_dir}")
        return
    if not os.path.exists(richness_csv):
        print(f"CSV not found: {richness_csv}")
        return
        
    df = pd.read_csv(richness_csv)
    df = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'richness': 'richness', 'Richness': 'richness'})
    print(f"Loaded {len(df)} richness sites.")
    
    tif_files = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')]
    print(f"Found {len(tif_files)} TIFF files in {tif_dir}")
    
    # Let's see how many tiles intersect with sites
    intersections = []
    mapped_sites = set()
    
    for tif_path in tqdm(tif_files, desc="Checking bounds"):
        try:
            with rasterio.open(tif_path) as src:
                bounds = src.bounds
                crs = src.crs
                inv_transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
                transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
                
                # Get bounds in WGS84
                lon1, lat1 = inv_transformer.transform(bounds.left, bounds.bottom)
                lon2, lat2 = inv_transformer.transform(bounds.right, bounds.top)
                min_lon, min_lat = min(lon1, lon2), min(lat1, lat2)
                max_lon, max_lat = max(lon1, lon2), max(lat1, lat2)
                
                # Check sites in bounds
                mask = (df['lat'] >= min_lat) & (df['lat'] <= max_lat) & \
                       (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
                relevant_sites = df[mask]
                
                if len(relevant_sites) > 0:
                    for idx, row in relevant_sites.iterrows():
                        # Transform lat/lon to map coordinates
                        easting, northing = transformer.transform(row['lon'], row['lat'])
                        col_idx, row_idx = ~src.transform * (easting, northing)
                        col_idx, row_idx = int(col_idx), int(row_idx)
                        
                        # Check bounds of pixel offsets
                        p = 8  # patch half size
                        is_inside = (row_idx >= p and row_idx < src.height - p and
                                     col_idx >= p and col_idx < src.width - p)
                        
                        # Read single band pixel to check nodata
                        nodata_val = None
                        if is_inside:
                            val = src.read(1, window=((row_idx-1, row_idx+2), (col_idx-1, col_idx+2)))
                            nodata_val = np.isnan(val).mean()
                            
                        intersections.append({
                            'tile': os.path.basename(tif_path),
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'SiteID': row.get('SiteID', f"Site_{idx}"),
                            'is_inside_pixel_bounds': is_inside,
                            'nodata_ratio': nodata_val
                        })
                        if is_inside and (nodata_val is None or nodata_val < 0.5):
                            mapped_sites.add((row['lat'], row['lon']))
        except Exception as e:
            print(f"Error reading {os.path.basename(tif_path)}: {e}")
            
    df_inter = pd.DataFrame(intersections)
    print(f"\nTotal intersection records found: {len(df_inter)}")
    if len(df_inter) > 0:
        print(f"Inside pixel bounds: {df_inter['is_inside_pixel_bounds'].sum()}")
        print(f"Unique coordinates inside bounds with valid pixels: {len(mapped_sites)}")
        
if __name__ == "__main__":
    main()
