import sys
import os
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bioscape_dataset import BioScapeNetCDFDataset
from src.s3_utils import get_s3_fs, get_cached_s3_fs

def test_s3_pull():
    print("[*] Testing Raw S3 Connection...")
    try:
        fs = get_s3_fs()
        files = fs.ls("s3://gaia-datasets/30m/")
        print(f"[OK] Found {len(files)} files in S3.")
        test_file = "s3://" + [f for f in files if f.endswith('.nc')][0]
        print(f"[*] Attempting to open: {test_file}")
        
        ds = BioScapeNetCDFDataset(test_file, richness_csv=None, patch_size=16, quiet=False, use_cache=False)
        print(f"[OK] Metadata loaded: {ds.width}x{ds.height}")
        cube = ds.h5_file['reflectance/reflectance']
        print(f"[*] Dataset Chunks: {cube.chunks}")
        print(f"[*] Dataset Compression: {cube.compression}")
        print("[*] Pulling a patch (Raw)...")
        # Center coordinates roughly for BioSCape
        lat, lon = -34.0, 19.0 
        # Actually, let's just use the first possible site from the dataset if we had one
        # But we can just use pixel indices
        ds._ensure_open()
        patch_raw = ds.h5_file['reflectance/reflectance'][:, 100:116, 100:116]
        print(f"[OK] Raw patch pull successful. Shape: {patch_raw.shape}")
        ds.close()

    except Exception as e:
        print(f"[!] Raw S3 Error: {e}")

    print("\n[*] Testing Cached S3 Connection (Blockcache)...")
    try:
        ds_cached = BioScapeNetCDFDataset(test_file, richness_csv=None, patch_size=16, quiet=False, use_cache=True)
        print("[*] Pulling a patch (Cached)...")
        # This will trigger get_patch_at_coord which uses the cache
        # We need a lat/lon that is valid. Let's use the bounds it just loaded.
        min_lon, min_lat, max_lon, max_lat = ds_cached.get_bounds()
        mid_lat, mid_lon = (min_lat + max_lat)/2, (min_lon + max_lon)/2
        
        patch, bounds = ds_cached.get_patch_at_latlon(mid_lat, mid_lon)
        if patch is not None:
            print(f"[OK] Cached patch pull successful. Shape: {patch.shape}")
        else:
            print("[!] Could not find valid patch at mid-point. Trying pixel indices.")
            ds_cached._ensure_open()
            cube = ds_cached.h5_file['reflectance/reflectance']
            p = cube[:, 200:216, 200:216]
            print(f"[OK] Cached pixel-index pull successful. Shape: {p.shape}")
        ds_cached.close()
    except Exception as e:
        print(f"[!] Cached S3 Error: {e}")

if __name__ == "__main__":
    test_s3_pull()
