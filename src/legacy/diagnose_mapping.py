import h5py
import numpy as np
import pandas as pd
from src.s3_utils import get_s3_fs
import os

def diagnose_mapping():
    print("[*] Loading richness CSV...")
    df = pd.read_csv('data/bioscape/western_cape_site_embeddings.csv')
    print(df[['Latitude', 'Longitude']].describe())
    
    fs = get_s3_fs()
    files = [f for f in fs.ls('gaia-test-500gb') if f.endswith('.nc')]
    print(f"[*] Found {len(files)} NetCDF files in S3.")
    
    for f_path in files[:5]: # Check first 5
        print(f"\n--- Checking: {f_path} ---")
        try:
            with fs.open(f_path, 'rb') as f_obj:
                with h5py.File(f_obj, 'r') as ds:
                    # AVIRIS-NG NetCDF structure:
                    # Lats/Lons are often in 'location/lat' and 'location/lon'
                    # or in the root.
                    if 'location' in ds:
                        lats = ds['location/lat'][:]
                        lons = ds['location/lon'][:]
                    else:
                        # Fallback to search for lat/lon keys
                        lat_key = [k for k in ds.keys() if 'lat' in k.lower()]
                        lon_key = [k for k in ds.keys() if 'lon' in k.lower()]
                        if lat_key and lon_key:
                            lats = ds[lat_key[0]][:]
                            lons = ds[lon_key[0]][:]
                        else:
                            print("[!] Could not find lat/lon in NetCDF.")
                            continue
                            
                    lat_min, lat_max = np.min(lats), np.max(lats)
                    lon_min, lon_max = np.min(lons), np.max(lons)
                    
                    print(f"    Extent: Lat ({lat_min:.4f} to {lat_max:.4f}), Lon ({lon_min:.4f} to {lon_max:.4f})")
                    
                    # Check overlap with CSV
                    overlap = df[
                        (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
                        (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)
                    ]
                    print(f"    Overlap: {len(overlap)} sites found in this granule.")
                    
        except Exception as e:
            print(f"    [!] Error: {e}")

if __name__ == "__main__":
    diagnose_mapping()
