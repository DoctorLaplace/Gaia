"""
GAIA SMART SYNC: Stream ONLY labeled granules from EarthData to S3 (or local).
Uses 'labeled_inventory.txt' by default to save 400GB+ of bandwidth.
(Legacy Sequential Version)
"""
import earthaccess
import os
import time
import sys
import subprocess
import argparse
from datetime import datetime

# --- PATHS ---
S3CMD_PATH = r"C:\Users\silve\AppData\Roaming\Python\Python312\Scripts\s3cmd"
CONFIG_PATH = r"C:\Users\silve\AppData\Roaming\s3cmd.ini"

# --- ANSI COLORS ---
GREEN = "\033[32m"
PURPLE = "\033[35m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_banner():
    print(f"\n{BOLD}{PURPLE}" + "="*70)
    print(f"      GAIA SMART SYNC: TARGETED DATA ACQUISITION")
    print("="*70 + f"{RESET}\n")

def get_existing_s3_files(res=30):
    s3_url = f"s3://gaia-datasets/{res}m/"
    print(f"{CYAN}[*] Checking current S3 state ({s3_url})...{RESET}")
    try:
        result = subprocess.run(['python', S3CMD_PATH, '-c', CONFIG_PATH, 'ls', s3_url], stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL)
        files = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                files.append(os.path.basename(parts[3]))
        return set(files)
    except:
        return set()

# --- BBL & PROCESSING ---
# Dr. Clark's Bad Band List (0 = bad, 1 = good) for 425 bands
BBL = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]

def downsample_netcdf(input_path, output_path, res_target):
    import h5py
    import numpy as np
    
    print(f"\n{CYAN}[*] Post-Processing: Downsampling to {res_target}m + Filtering Bad Bands...{RESET}")
    
    good_indices = np.array([i for i, val in enumerate(BBL) if val == 1])
    factor = max(1, int(res_target // 5))
    
    with h5py.File(input_path, 'r') as src:
        # Read the reflectance cube and wavelengths
        cube = src['reflectance/reflectance'][()]   # (bands, northing, easting)
        wavelengths = src['reflectance/wavelength'][()]
        fwhm = src['reflectance/fwhm'][()]
        
        # 1. Apply BBL mask
        cube = cube[good_indices]
        wavelengths = wavelengths[good_indices]
        fwhm = fwhm[good_indices]
        
        # 2. Spatial downsampling via block averaging
        if factor > 1:
            b, h, w = cube.shape
            h_new = (h // factor) * factor
            w_new = (w // factor) * factor
            cube = cube[:, :h_new, :w_new]
            cube = cube.reshape(b, h_new // factor, factor, w_new // factor, factor)
            cube = cube.mean(axis=(2, 4))
        
        # 3. Read and update GeoTransform
        gt_str = src['projection'].attrs.get('GeoTransform', None)
        if gt_str is not None:
            if isinstance(gt_str, bytes): gt_str = gt_str.decode()
            gt = [float(x) for x in gt_str.split()]
            # gt = [origin_x, pixel_w, 0, origin_y, 0, pixel_h]
            gt[1] *= factor   # pixel_w: 5m -> 30m
            gt[5] *= factor   # pixel_h: -5m -> -30m
            gt_out = " ".join(str(v) for v in gt)
        else:
            gt_out = None
        
        # Also grab the CRS WKT if present
        proj_wkt = src['projection'].attrs.get('spatial_ref', None)
        
    # 4. Write output in the same grouped HDF5 structure
    with h5py.File(output_path, 'w') as dst:
        # Root-level projection dataset (matches original structure)
        proj_ds = dst.create_dataset('projection', data=np.uint8(0))
        if gt_out:
            proj_ds.attrs['GeoTransform'] = gt_out
        if proj_wkt is not None:
            proj_ds.attrs['spatial_ref'] = proj_wkt
        
        # Reflectance group
        grp = dst.create_group('reflectance')
        grp.create_dataset('reflectance', data=cube.astype(np.float32), compression='gzip', compression_opts=4)
        grp.create_dataset('wavelength', data=wavelengths.astype(np.float32))
        grp.create_dataset('fwhm', data=fwhm.astype(np.float32))
    
    orig_mb = os.path.getsize(input_path) / 1024 / 1024
    new_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  {GREEN}Processed: {cube.shape[0]} bands, {cube.shape[1]}x{cube.shape[2]} pixels ({orig_mb:.0f}MB -> {new_mb:.0f}MB){RESET}")

def smart_sync(inventory_path, limit=None, s3_upload=True, local_dir=None, res=30):
    print_banner()
    
    if not os.path.exists(inventory_path):
        print(f"{RED}[!] Error: {inventory_path} not found. Run generate_labeled_inventory.py first.{RESET}")
        return

    # 1. Login
    earthaccess.login(persist=True)
    
    # 2. Get existing state (S3 or local)
    if s3_upload:
        existing = get_existing_s3_files(res=res)
    elif local_dir and os.path.isdir(local_dir):
        existing = {f for f in os.listdir(local_dir) if f.endswith('.nc')}
        # Also keep full filenames for exact matching in the transfer loop
        existing_files = {f for f in os.listdir(local_dir) if f.endswith('.nc')}
    else:
        existing = set()
        existing_files = set()
    
    if not s3_upload:
        existing_files = existing_files if 'existing_files' in dir() else set()
    
    # 3. Load Targets
    with open(inventory_path, 'r') as f:
        target_ids = [l.strip() for l in f if l.strip()]
    
    def normalize(nid):
        return nid.replace("BioSCape_AVNG_L2B_BRDF_GCFR.", "")

    new_ids = [nid for nid in target_ids if normalize(nid) not in existing]
    
    print(f"{GREEN}[OK] Loaded {len(target_ids)} target granules. {len(new_ids)} are remaining.{RESET}")
    if local_dir and not s3_upload:
        already = len(target_ids) - len(new_ids)
        if already > 0:
            print(f"  {CYAN}(Skipping {already} already downloaded in {local_dir}){RESET}")
    
    if limit:
        new_ids = new_ids[:limit]
        
    if not new_ids:
        print(f"{BOLD}{GREEN}All targeted granules are already in sync!{RESET}")
        return

    # 4. CMR Discovery
    print(f"{CYAN}[*] Discovering {len(new_ids)} granules on NASA CMR...{RESET}")
    granules = []
    chunk_size = 50
    for i in range(0, len(new_ids), chunk_size):
        chunk = new_ids[i:i + chunk_size]
        results = earthaccess.search_data(short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385", granule_name=chunk)
        granules.extend(results)
    
    print(f"{GREEN}[OK] Discovery Complete: {len(granules)} granules ready for transfer.{RESET}")

    # 5. Transfer Loop
    session = earthaccess.get_requests_https_session()
    
    for i, g in enumerate(granules):
        url = g.data_links()[0]
        filename = os.path.basename(url)
        
        # Skip if already downloaded locally
        if not s3_upload and filename in existing_files:
            print(f"\n{CYAN}--- [{i+1}/{len(granules)}] SKIP (exists): {filename} ---{RESET}")
            continue
        
        print(f"\n{BOLD}{PURPLE}--- [{i+1}/{len(granules)}] SYNCING: {filename} ---{RESET}")
        
        try:
            # If downsampling, we must download locally first
            if res > 5:
                tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                pid = os.getpid()
                temp_file = os.path.join(tmp_dir, f"temp_{pid}_{filename}")
            else:
                temp_file = None
            
            with session.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                # Streaming Output
                if not temp_file:
                    if s3_upload:
                        s3_path = f"s3://gaia-datasets/{res}m/{filename}"
                        cmd = ['python', S3CMD_PATH, '-c', CONFIG_PATH, 'put', '-', s3_path]
                        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    else:
                        os.makedirs(local_dir, exist_ok=True)
                        dest_f = open(os.path.join(local_dir, filename), 'wb')
                else:
                    dest_f = open(temp_file, 'wb')

                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024*1024*10):
                    if chunk:
                        if not temp_file and s3_upload:
                            proc.stdin.write(chunk)
                        else:
                            dest_f.write(chunk)
                        downloaded += len(chunk)
                        print(f"\r  Download: {downloaded/1024/1024:.0f}MB / {total_size/1024/1024:.0f}MB", end="")
                
                if not temp_file and s3_upload:
                    proc.stdin.close()
                    proc.wait()
                else:
                    dest_f.close()

            # Optional Processing
            if temp_file:
                pid = os.getpid()
                processed_file = os.path.join(tmp_dir, f"proc_{pid}_{filename}")
                downsample_netcdf(temp_file, processed_file, res)
                
                # Upload processed result
                if s3_upload:
                    s3_path = f"s3://gaia-datasets/{res}m/{filename}"
                    subprocess.run(['python', S3CMD_PATH, '-c', CONFIG_PATH, 'put', processed_file, s3_path])
                elif local_dir:
                    os.makedirs(local_dir, exist_ok=True)
                    os.replace(processed_file, os.path.join(local_dir, filename))
                
                # Cleanup
                if os.path.exists(temp_file): os.remove(temp_file)
                if os.path.exists(processed_file) and os.path.isfile(processed_file): os.remove(processed_file)

            print(f"\n{GREEN}[OK] SUCCESS{RESET}")
                
        except Exception as e:
            print(f"\n{RED}[!] ERROR: {e}{RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="labeled_inventory.txt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local", type=str, nargs='?', const='auto',
                        help="Download locally. Omit path for auto (data/bioscape/<res>m/)")
    parser.add_argument("--res", type=int, default=30,
                        help="Resolution in meters (default 30, use 5 for original)")
    args = parser.parse_args()
    
    # Auto-route to resolution subfolder
    local_dir = args.local
    if local_dir == 'auto':
        local_dir = f"data/bioscape/{args.res}m"
    
    smart_sync(
        args.inventory, 
        limit=args.limit, 
        s3_upload=(args.local is None),
        local_dir=local_dir,
        res=args.res
    )
