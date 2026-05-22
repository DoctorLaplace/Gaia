"""
GAIA SMART SYNC: Stream ONLY labeled granules from EarthData to S3 (or local).
Uses 'labeled_inventory.txt' by default to save 400GB+ of bandwidth.
"""
import earthaccess
import os
import time
import sys
import subprocess
import argparse
import threading
import uuid
import signal
import queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np

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

# --- CONCURRENCY & INTERRUPTS ---
print_lock = threading.Lock()
stop_event = threading.Event()
worker_seats = queue.Queue()

# --- PERFORMANCE TRACKING ---
stats_lock = threading.Lock()
total_downloaded_bytes = 0

def add_stats(bytes_count):
    global total_downloaded_bytes
    with stats_lock:
        total_downloaded_bytes += bytes_count

def safe_print(msg):
    """Thread-safe print that plays nice with tqdm."""
    with print_lock:
        tqdm.write(msg)

def signal_handler(sig, frame):
    """Graceful Ctrl+C handling."""
    if not stop_event.is_set():
        safe_print(f"\n{RED}[!] Interrupt received. Cleaning up and exiting...{RESET}")
        stop_event.set()

signal.signal(signal.SIGINT, signal_handler)

def print_banner():
    safe_print(f"\n{BOLD}{PURPLE}" + "="*70)
    safe_print(f"      GAIA SMART SYNC: TARGETED DATA ACQUISITION")
    safe_print("="*70 + f"{RESET}\n")

def get_existing_s3_files(res=30):
    s3_url = f"s3://gaia-datasets/{res}m/"
    safe_print(f"{CYAN}[*] Checking current S3 state ({s3_url})...{RESET}")
    try:
        result = subprocess.run([sys.executable, S3CMD_PATH, '-c', CONFIG_PATH, 'ls', s3_url], stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL)
        files = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                files.append(os.path.basename(parts[3]))
        return set(files)
    except:
        return set()

# --- BBL & PROCESSING ---
BBL = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]

def downsample_netcdf(input_path, output_path, res_target, log_prefix):
    import h5py
    import numpy as np

    if stop_event.is_set():
        return

    filename = os.path.basename(input_path)

    safe_print(f"{log_prefix} Processing (CF-grid): {filename}")

    good_indices = np.array([i for i, val in enumerate(BBL) if val == 1])

    factor = max(1, int(res_target // 5))

    try:
        with h5py.File(input_path, 'r') as src:

            # -------------------------
            # LOAD CORE DATASETS
            # -------------------------
            grp = src['reflectance']
            cube = grp['reflectance'][good_indices]
            wavelengths = grp['wavelength'][good_indices]
            fwhm = grp['fwhm'][good_indices]

            easting = src['easting'][:]
            northing = src['northing'][:]

            proj_var = None
            proj_attrs = None

            if 'transverse_mercator' in src:
                proj_var = src['transverse_mercator'][()]
                proj_attrs = dict(src['transverse_mercator'].attrs)

            safe_print("[DEBUG] successfully loaded datasets and applied spectral filter")
            # -------------------------
            # SPATIAL DOWNSAMPLE (cube)
            # -------------------------
            b, h, w = cube.shape

            h_new = (h // factor) * factor
            w_new = (w // factor) * factor

            cube = cube[:, :h_new, :w_new]

            cube = cube.reshape(
                b,
                h_new // factor,
                factor,
                w_new // factor,
                factor
            ).mean(axis=(2, 4))
            safe_print("[DEBUG] successfully spatially downsampled")
            # -------------------------
            # DOWNSAMPLE COORDINATES
            # -------------------------
            easting = easting[:w_new]
            northing = northing[:h_new]

            easting = easting.reshape(-1, factor).mean(axis=1)
            northing = northing.reshape(-1, factor).mean(axis=1)
            #print("[DEBUG] successfully downsampled coordinates")
            # -------------------------
            # WRITE OUTPUT
            # -------------------------
            #print(f"[DEBUG] output_path: {output_path}")
            with h5py.File(output_path, 'w') as dst:
                grp_out = dst.create_group('reflectance')
                grp_out.create_dataset(
                    'reflectance',
                    data=cube.astype(np.float32),
                    compression=None
                )
                grp_out.create_dataset(
                    'wavelength',
                    data=wavelengths.astype(np.float32)
                )
                grp_out.create_dataset(
                    'fwhm',
                    data=fwhm.astype(np.float32)
                )
                # CF-style geolocation (IMPORTANT)
                dst.create_dataset('easting', data=easting.astype(np.float32))
                dst.create_dataset('northing', data=northing.astype(np.float32))

                # Restore CRS variable
                if proj_var is not None:
                    tm = dst.create_dataset('transverse_mercator', data=proj_var)
                    for k, v in proj_attrs.items():
                        tm.attrs[k] = v
                grp_out['reflectance'].attrs['grid_mapping'] = 'transverse_mercator'

                # metadata (optional but useful)
                dst.attrs['resolution_factor'] = factor
                dst.attrs['source_file'] = filename


        orig_mb = os.path.getsize(input_path) / 1024 / 1024
        new_mb = os.path.getsize(output_path) / 1024 / 1024

        safe_print(
            f"{GREEN}[OK] Downsampled:{RESET} "
            f"{filename} ({orig_mb:.0f}MB -> {new_mb:.0f}MB)"
        )

    except Exception as e:
        if not stop_event.is_set():
            safe_print(f"{RED}[!] Processing Error {filename}: {e}{RESET}")
        raise

def downsample_streaming(cube, factor, block=256):
    b, h, w = cube.shape

    h_new = h // factor
    w_new = w // factor

    out = np.zeros((b, h_new, w_new), dtype=np.float32)

    for i in range(0, h, block):
        for j in range(0, w, block):

            block_data = cube[
                :,
                i:min(i+block, h),
                j:min(j+block, w)
            ]

            # trim to factor multiples
            hh = (block_data.shape[1] // factor) * factor
            ww = (block_data.shape[2] // factor) * factor
            block_data = block_data[:, :hh, :ww]

            if hh == 0 or ww == 0:
                continue

            reduced = block_data.reshape(
                b,
                hh // factor,
                factor,
                ww // factor,
                factor
            ).mean(axis=(2, 4))

            out_i = i // factor
            out_j = j // factor

            out[
                :,
                out_i:out_i + reduced.shape[1],
                out_j:out_j + reduced.shape[2]
            ] = reduced

    return out

def sync_worker(granule_info):
    """Worker function to handle download and processing for a single granule."""
    g, idx, total, session, s3_upload, local_dir, res = granule_info
    links = g.data_links()

    rfl_links = []
    for l in links:
        if l.endswith("_RFL.nc"):
            rfl_links.append(l)

    if not rfl_links:
        safe_print(f"{RED}[!] No RFL .nc file found for granule{RESET}")
        return

    url = rfl_links[0]
    filename = os.path.basename(url)
    
    if stop_event.is_set(): return

    # Get a "seat" for the tqdm bar position
    seat = worker_seats.get()
    log_prefix = f"{BOLD}{PURPLE}[{idx}/{total}]{RESET}"
    
    unique_id = uuid.uuid4().hex[:8]
    tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_file = os.path.join(tmp_dir, f"temp_{unique_id}_{filename}")
    processed_file = os.path.join(tmp_dir, f"proc_{unique_id}_{filename}")
    
    try:
        # 1. Download
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            # Short filename for the bar
            short_name = filename.split('_L3_')[1][:5] if '_L3_' in filename else filename[:15]
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"DWN {short_name}", position=seat, leave=False) as pbar:
                with open(temp_file, 'wb') as dest_f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                        if stop_event.is_set():
                            dest_f.close()
                            return
                        if chunk:
                            dest_f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                    add_stats(downloaded)
                
                # 2. Process (Update bar to reflect processing)
                pbar.set_description(f"PRC {short_name}")
                downsample_netcdf(temp_file, processed_file, res, log_prefix)
        
        if stop_event.is_set(): return

        # 3. Deliver
        #print(f"[DEBUG] local dir = {local_dir}")
        #if s3_upload:
        #    s3_path = f"s3://gaia-datasets/{res}m/{filename}"
        #    res_proc = subprocess.run([sys.executable, S3CMD_PATH, '-c', CONFIG_PATH, 'put', processed_file, s3_path],
        #                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #    if res_proc.returncode != 0:
        #        raise Exception(f"S3 Upload failed: {res_proc.stderr}")
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            target_path = os.path.join(local_dir, filename)
            if os.path.exists(target_path): os.remove(target_path)
            os.replace(processed_file, target_path)
        
        safe_print(f"{GREEN}{log_prefix} SUCCESS:{RESET} {filename}")
        
    except Exception as e:
        if not stop_event.is_set():
            safe_print(f"{RED}{log_prefix} ERROR {filename}:{RESET} {e}")
    finally:
        worker_seats.put(seat)
        for f in [temp_file, processed_file]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

def smart_sync(inventory_path, limit=None, s3_upload=False, local_dir=None, res=30, workers=4):
    print_banner()
    start_time = time.time()
    
    if not os.path.exists(inventory_path):
        safe_print(f"{RED}[!] Error: {inventory_path} not found. Run generate_labeled_inventory.py first.{RESET}")
        return

    earthaccess.login(persist=True)
    
    if s3_upload:
        existing = get_existing_s3_files(res=res)
    elif local_dir and os.path.isdir(local_dir):
        existing = {f.replace('_RFL.nc', '').split('_L3_')[1] for f in os.listdir(local_dir) if f.endswith('.nc')}
    else:
        existing = set()
    
    target_ids = load_inventory(inventory_path)

    normalize = lambda nid: nid.replace("BioSCape_AVNG_L2B_BRDF_GCFR.", "")
    new_ids = [nid for nid in target_ids if normalize(nid) not in existing]
    
    safe_print(f"{GREEN}[OK] Loaded {len(target_ids)} target granules. {len(new_ids)} are remaining.{RESET}")
    if limit: new_ids = new_ids[:limit]
    if not new_ids:
        safe_print(f"{BOLD}{GREEN}All targeted granules are already in sync!{RESET}")
        return

    safe_print(f"{CYAN}[*] Discovering {len(new_ids)} granules on NASA CMR...{RESET}")
    safe_print(f"{CYAN}[*] Fetching full granule list from CMR...{RESET}")

    all_granules = earthaccess.search_data(
        short_name="BioSCape_ANG_V02_L3_RFL_Mosaic_2427",
        version="2"
    )

    collections = earthaccess.search_datasets(keyword="bioscape v02 mosaic rfl")

    for c in collections:
        meta = c["umm"]
        print(
            meta.get("ShortName"),
            meta.get("Version"),
            meta.get("EntryTitle")
        )

    safe_print(f"{GREEN}[OK] Retrieved {len(all_granules)} total granules.{RESET}")

    tile_set = set(new_ids)
    granules = []

    for g in all_granules:
        try:
            name = g["umm"]["GranuleUR"]
            parts = name.split('_')

            # Correct tile extraction
            tile = f"{parts[-2]}_{parts[-1]}"

            if tile in tile_set:
                granules.append(g)

        except Exception:
            continue

    safe_print(f"{GREEN}[OK] Discovery Complete: {len(granules)} granules ready for transfer.{RESET}\n")

    # Initialize worker seats (0-indexed for tqdm)
    for i in range(workers):
        worker_seats.put(i)

    session = earthaccess.get_requests_https_session()
    work_items = [(g, i+1, len(granules), session, s3_upload, local_dir, res) for i, g in enumerate(granules)]
    
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(sync_worker, work_items))
    except (KeyboardInterrupt, SystemExit):
        stop_event.set()
        safe_print(f"\n{RED}[!] CANCELLING: Waiting for workers to clean up...{RESET}")
        executor.shutdown(wait=False)
        sys.exit(1)

    if not stop_event.is_set():
        elapsed = time.time() - start_time
        total_mb = total_downloaded_bytes / 1024 / 1024
        safe_print(f"\n{BOLD}{GREEN}--- ALL SYNC OPERATIONS COMPLETE ---{RESET}")
        if elapsed > 0:
            speed = total_mb / elapsed
            safe_print(f"      Total Time:     {elapsed:.1f}s")
            safe_print(f"      Total Data:     {total_mb:.1f} MB")
            safe_print(f"      Avg Throughput: {speed:.2f} MB/s")
        safe_print(f"{BOLD}{GREEN}-------------------------------------{RESET}")

import json

def load_inventory(path):
    """Load tile list from JSON or fallback TXT."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)

        if "tiles" not in data:
            raise ValueError("JSON must contain 'tiles' key")

        # Extract tile strings like "14_15"
        return [t["tile"] for t in data["tiles"] if "tile" in t]

    else:
        with open(path, "r") as f:
            return [l.strip() for l in f if l.strip()]

def tile_to_pattern(tile):
    # matches any granule containing the tile string
    return f"*{tile}*"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="tile_names.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local", type=str, nargs='?', const='auto')
    parser.add_argument("--res", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    local_dir = args.local
    if local_dir == 'auto': local_dir = f"data/bioscape/{args.res}m_v2"
    
    smart_sync(
        args.inventory, 
        limit=args.limit, 
        s3_upload=(args.local is None),
        local_dir=local_dir,
        res=args.res,
        workers=args.workers
    )
