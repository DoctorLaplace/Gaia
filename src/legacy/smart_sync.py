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

# --- PATHS ---
S3CMD_PATH = r"C:\Users\silve\AppData\Roaming\Python\Python312\Scripts\s3cmd"
CONFIG_PATH = r"C:\Users\silve\AppData\Roaming\s3cmd.ini"

# --- ANSI COLORS ---
GREEN = "\033[92m"
PURPLE = "\033[1;95m"
CYAN = "\033[96m"
RED = "\033[91m"
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

def enforce_dataset_boundaries(labeled_dir, ssl_dir, inventory_path):
    import shutil
    safe_print(f"\n{PURPLE}[*] Auditing dataset boundaries (Labeled vs SSL)...{RESET}")
    
    if not os.path.exists(inventory_path):
        safe_print(f"  {RED}[!] inventory file not found. Skipping deduplication.{RESET}")
        return
        
    # Ensure directories exist before routing
    from pathlib import Path
    Path(labeled_dir).mkdir(parents=True, exist_ok=True)
    Path(ssl_dir).mkdir(parents=True, exist_ok=True)
        
    with open(inventory_path, 'r') as f:
        labeled_ids = set([l.strip().replace("BioSCape_AVNG_L2B_BRDF_GCFR.", "").replace(".nc", "") for l in f if l.strip()])
        
    actions_taken = 0
    scanned_files = 0
        
    # Check Labeled Dir
    if os.path.exists(labeled_dir):
        for f in os.listdir(labeled_dir):
            if not f.endswith('.nc'): continue
            scanned_files += 1
            f_id = f.replace('.nc', '').replace("BioSCape_AVNG_L2B_BRDF_GCFR.", "")
            
            if f_id not in labeled_ids:
                src = os.path.join(labeled_dir, f)
                dst = os.path.join(ssl_dir, f)
                try:
                    if os.path.exists(dst):
                        os.remove(src)
                        safe_print(f"  {RED}[-] Deduplicated: Deleted {f} from Labeled (already in SSL){RESET}")
                    else:
                        shutil.move(src, dst)
                        safe_print(f"  {CYAN}[->] Routed to SSL: {f} (Unlabeled){RESET}")
                    actions_taken += 1
                except PermissionError:
                    safe_print(f"  {RED}[!] Locked: Cannot move {f} (File is currently open/in-use){RESET}")
                
    # Check SSL Dir
    if os.path.exists(ssl_dir):
        for f in os.listdir(ssl_dir):
            if not f.endswith('.nc'): continue
            scanned_files += 1
            f_id = f.replace('.nc', '').replace("BioSCape_AVNG_L2B_BRDF_GCFR.", "")
            
            if f_id in labeled_ids:
                src = os.path.join(ssl_dir, f)
                dst = os.path.join(labeled_dir, f)
                try:
                    if os.path.exists(dst):
                        os.remove(src)
                        safe_print(f"  {RED}[-] Deduplicated: Deleted {f} from SSL (already in Labeled){RESET}")
                    else:
                        shutil.move(src, dst)
                        safe_print(f"  {GREEN}[->] Routed to Labeled: {f} (Ground-Truth){RESET}")
                    actions_taken += 1
                except PermissionError:
                    safe_print(f"  {RED}[!] Locked: Cannot move {f} (File is currently open/in-use){RESET}")
                
    if actions_taken == 0:
        safe_print(f"  {GREEN}[OK] Dataset boundaries are perfectly fit ({scanned_files} files verified).{RESET}")
    else:
        safe_print(f"  {GREEN}[OK] Re-aligned {actions_taken} files to their correct architectural boundaries.{RESET}")


def get_existing_s3_files(res=30):
    s3_url = f"s3://gaia-datasets/{res}m/"
    safe_print(f"{CYAN}[*] Checking current S3 state ({s3_url})...{RESET}")
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
BBL = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]

def downsample_netcdf(input_path, output_path, res_target, log_prefix):
    import h5py
    import numpy as np
    if stop_event.is_set(): return
    
    filename = os.path.basename(input_path).replace('temp_', '')
    safe_print(f"{log_prefix} {CYAN}Processing:{RESET} {filename} -> {res_target}m")
    
    good_indices = np.array([i for i, val in enumerate(BBL) if val == 1])
    factor = max(1, int(res_target // 5))
    
    try:
        with h5py.File(input_path, 'r') as src:
            cube = src['reflectance/reflectance']
            wavelengths = src['reflectance/wavelength'][()]
            fwhm = src['reflectance/fwhm'][()]
            cube_data = cube[()]
            
            if stop_event.is_set(): return
            cube_data = cube_data[good_indices]
            wavelengths = wavelengths[good_indices]
            fwhm = fwhm[good_indices]
            
            if factor > 1:
                b, h, w = cube_data.shape
                h_new = (h // factor) * factor
                w_new = (w // factor) * factor
                cube_data = cube_data[:, :h_new, :w_new]
                cube_data = cube_data.reshape(b, h_new // factor, factor, w_new // factor, factor)
                cube_data = cube_data.mean(axis=(2, 4))
            
            gt_str = src['projection'].attrs.get('GeoTransform', None)
            if gt_str is not None:
                if isinstance(gt_str, bytes): gt_str = gt_str.decode()
                gt = [float(x) for x in gt_str.split()]
                gt[1] *= factor
                gt[5] *= factor
                gt_out = " ".join(str(v) for v in gt)
            else:
                gt_out = None
            proj_wkt = src['projection'].attrs.get('spatial_ref', None)
            
        if stop_event.is_set(): return
        with h5py.File(output_path, 'w') as dst:
            proj_ds = dst.create_dataset('projection', data=np.uint8(0))
            if gt_out: proj_ds.attrs['GeoTransform'] = gt_out
            if proj_wkt is not None: proj_ds.attrs['spatial_ref'] = proj_wkt
            grp = dst.create_group('reflectance')
            grp.create_dataset('reflectance', data=cube_data.astype(np.float32), compression='gzip', compression_opts=4)
            grp.create_dataset('wavelength', data=wavelengths.astype(np.float32))
            grp.create_dataset('fwhm', data=fwhm.astype(np.float32))
            
        orig_mb = os.path.getsize(input_path) / 1024 / 1024
        new_mb = os.path.getsize(output_path) / 1024 / 1024
        safe_print(f"    {GREEN}[OK] Downsampled:{RESET} {filename} ({orig_mb:.0f}MB -> {new_mb:.0f}MB)")
    except Exception as e:
        if not stop_event.is_set():
            safe_print(f"  {RED}[!] Processing Error {filename}: {e}{RESET}")
        raise

def sync_worker(granule_info):
    """Worker function to handle download and processing for a single granule."""
    g, idx, total, session, s3_upload, local_dir, res = granule_info
    url = g.data_links()[0]
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
            short_name = filename.split('_L2B_')[0] if '_L2B_' in filename else filename[:15]
            
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
        if s3_upload:
            s3_path = f"s3://gaia-datasets/{res}m/{filename}"
            res_proc = subprocess.run(['python', S3CMD_PATH, '-c', CONFIG_PATH, 'put', processed_file, s3_path], 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_proc.returncode != 0:
                raise Exception(f"S3 Upload failed: {res_proc.stderr}")
        elif local_dir:
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

def smart_sync(inventory_path, limit=None, s3_upload=True, local_dir=None, res=30, workers=4):
    print_banner()
    start_time = time.time()
    
    if not os.path.exists(inventory_path):
        safe_print(f"{RED}[!] Error: {inventory_path} not found. Run generate_labeled_inventory.py first.{RESET}")
        return

    earthaccess.login(persist=True)
    
    # Establish local directories
    if local_dir == 'auto' or not local_dir:
        local_dir = f"data/bioscape/{res}m"
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ssl_dir = os.path.join(project_root, f"embedding_creation/data/{res}m")
    labeled_dir = os.path.join(project_root, local_dir)
    os.makedirs(ssl_dir, exist_ok=True)
    os.makedirs(labeled_dir, exist_ok=True)
    
    # Run Pre-flight Routing
    enforce_dataset_boundaries(labeled_dir, ssl_dir, inventory_path)
    
    if s3_upload:
        existing = get_existing_s3_files(res=res)
    else:
        existing = set()
        for d in [labeled_dir, ssl_dir]:
            if os.path.exists(d):
                existing.update({f.replace('.nc', '') for f in os.listdir(d) if f.endswith('.nc')})
    
    with open(inventory_path, 'r') as f:
        target_ids = [l.strip() for l in f if l.strip()]
    
    normalize = lambda nid: nid.replace("BioSCape_AVNG_L2B_BRDF_GCFR.", "").replace(".nc", "")
    new_ids = [nid for nid in target_ids if normalize(nid) not in existing]
    
    safe_print(f"{GREEN}[OK] Loaded {len(target_ids)} target granules. {len(new_ids)} are remaining.{RESET}")
    if limit is not None: new_ids = new_ids[:limit]
    if not new_ids:
        safe_print(f"{BOLD}{GREEN}All targeted granules are already in sync!{RESET}")
        return

    safe_print(f"{CYAN}[*] Discovering {len(new_ids)} granules on NASA CMR...{RESET}")
    granules = []
    chunk_size = 50
    for i in range(0, len(new_ids), chunk_size):
        if stop_event.is_set(): return
        chunk = new_ids[i:i + chunk_size]
        results = earthaccess.search_data(short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385", granule_ur=chunk)
        granules.extend(results)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="labeled_inventory.txt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local", type=str, nargs='?', const='auto')
    parser.add_argument("--res", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    local_dir = args.local
    if local_dir == 'auto': local_dir = f"data/bioscape/{args.res}m"
    
    smart_sync(
        args.inventory, 
        limit=args.limit, 
        s3_upload=(args.local is None and not os.path.exists('data/bioscape')), # simple heuristic
        local_dir=local_dir,
        res=args.res,
        workers=args.workers
    )
