import earthaccess
import os
import time
import sys
import subprocess
from datetime import datetime

# --- PATHS ---
S3CMD_PATH = r"C:\Users\silve\AppData\Roaming\Python\Python312\Scripts\s3cmd"
CONFIG_PATH = r"C:\Users\silve\AppData\Roaming\s3cmd.ini"

# --- ANSI COLORS ---
PURPLE_DARK = "\033[38;5;55m"
PURPLE_MED = "\033[38;5;99m"
PURPLE_LIGHT = "\033[38;5;141m"
BLUE_NEON = "\033[38;5;45m"
WHITE = "\033[37m"
RED = "\033[1;31m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_banner():
    print(f"\n{BOLD}{PURPLE_MED}" + "="*70)
    print(f"      GAIA PRODUCTION DATA PIPELINE: BIOSCAPE -> S3")
    print("="*70 + f"{RESET}\n")

def format_size(bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0

def get_existing_s3_files():
    print(f"{PURPLE_LIGHT}[*] Checking current S3 state (s3://gaia-datasets)...{RESET}")
    try:
        # Run s3cmd ls
        result = subprocess.run(['python', S3CMD_PATH, '-c', CONFIG_PATH, '--no-preserve', '--no-progress', '--quiet', 'ls', 's3://gaia-datasets/'], stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL)
        files = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                files.append(os.path.basename(parts[3]))
        print(f"{PURPLE_MED}[OK] Found {len(files)} existing granules in S3.{RESET}")
        return set(files)
    except Exception as e:
        print(f"{RED}[!] Could not check S3 state. Assuming empty. Error: {e}{RESET}")
        return set()

def production_sync(limit=None):
    print_banner()
    
    # 1. Login to Earthdata
    print(f"{PURPLE_LIGHT}[*] Phase 1: Authentication with NASA Earthdata...{RESET}")
    auth = earthaccess.login(persist=True)
    
    # 2. Get existing files in S3
    existing_files = get_existing_s3_files()
    
    # 3. Load Manifest
    print(f"{PURPLE_LIGHT}[*] Phase 2: Loading flightline inventory...{RESET}")
    inventory_path = 'full_inventory.txt'
    if not os.path.exists(inventory_path):
        print(f"{RED}[!] Error: {inventory_path} not found.{RESET}")
        return
        
    with open(inventory_path, 'r') as f:
        all_ids = [l.strip() for l in f if l.strip()]
    
    # Filter out what we ALREADY HAVE
    # Match by stripping the collection prefix: BioSCape_AVNG_L2B_BRDF_GCFR.
    def normalize(nid):
        return nid.replace("BioSCape_AVNG_L2B_BRDF_GCFR.", "")

    new_ids = [nid for nid in all_ids if normalize(nid) not in existing_files]
    
    if args.granule:
        new_ids = [nid for nid in all_ids if args.granule in nid]
        if not new_ids:
            print(f"{RED}[!] Error: Targeted granule {args.granule} not found in inventory.{RESET}")
            return
        print(f"{PURPLE_LIGHT}[!] TARGETED SYNC: {new_ids[0]}{RESET}")
    
    print(f"{PURPLE_MED}[OK] Loaded {len(all_ids)} granules. {len(new_ids)} are NEW.{RESET}")
    
    if limit:
        print(f"{PURPLE_LIGHT}[!] LIMIT ACTIVE: Syncing only first {limit} new granules.{RESET}")
        new_ids = new_ids[:limit]
    
    if not new_ids:
        print(f"{BOLD}{PURPLE_MED}Everything is already in sync!{RESET}")
        return

    # 4. Discovery (Only for the ones we need)
    print(f"{PURPLE_LIGHT}[*] Phase 3: Discovering {len(new_ids)} granules on NASA CMR...{RESET}")
    granules = []
    chunk_size = 50
    for i in range(0, len(new_ids), chunk_size):
        chunk = new_ids[i:i + chunk_size]
        sys.stdout.write(f"\r    > Querying NASA: {i + len(chunk)}/{len(new_ids)}...")
        sys.stdout.flush()
        try:
            results = earthaccess.search_data(
                short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385",
                granule_name=chunk
            )
            granules.extend(results)
        except Exception:
            pass
    
    print(f"\n{PURPLE_MED}[OK] Discovery Complete: {len(granules)} granules ready for sync.{RESET}")
    
    # 5. Sync Loop
    print(f"{PURPLE_LIGHT}[*] Phase 4: Streaming to S3...{RESET}")
    session = earthaccess.get_requests_https_session()
    
    total_synced_bytes = 0
    start_pipeline_time = time.time()
    
    for i, g in enumerate(granules):
        url = g.data_links()[0]
        filename = os.path.basename(url)
        s3_path = f"s3://gaia-datasets/{filename}"
        
        print(f"\n{BOLD}{PURPLE_MED}--- [{i+1}/{len(granules)}] SYNCING: {filename} ---{RESET}")
        
        try:
            with session.get(url, stream=True) as r:
                r.raise_for_status()
                file_size = int(r.headers.get('content-length', 0))
                
                print(f"  {PURPLE_LIGHT}Source:{RESET} {url}")
                print(f"  {PURPLE_LIGHT}Target:{RESET} {s3_path} ({format_size(file_size)})")
                
                cmd = ['python', S3CMD_PATH, '-c', CONFIG_PATH, '--no-preserve', '--no-progress', '--quiet', 'put', '-', s3_path]
                s3_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                
                downloaded = 0
                file_start_time = time.time()
                last_update = time.time()
                
                for chunk in r.iter_content(chunk_size=1024*1024*5):
                    if chunk:
                        s3_proc.stdin.write(chunk)
                        downloaded += len(chunk)
                        
                        if time.time() - last_update > 1.0:
                            elapsed = time.time() - file_start_time
                            speed = downloaded / elapsed if elapsed > 0 else 0
                            percent = (downloaded / file_size) * 100
                            file_eta = (file_size - downloaded) / speed if speed > 0 else 0
                            bar_len = 50
                            filled = int(percent / 100 * bar_len)
                            # Sleek blue bar in a purple frame
                            bar = f"{BLUE_NEON}" + "━" * filled + f"{RESET}" + " " * (bar_len - filled)
                            sys.stdout.write(f"\r  {PURPLE_DARK}[{bar}{PURPLE_DARK}]{RESET} {PURPLE_LIGHT}{percent:5.1f}%{RESET} | {PURPLE_LIGHT}{format_size(speed)}/s{RESET} | {PURPLE_LIGHT}ETA: {file_eta:.0f}s{RESET}    ")
                            sys.stdout.flush()
                            last_update = time.time()
                
                s3_proc.stdin.close()
                s3_proc.wait()
                total_synced_bytes += file_size
                print(f"\n{PURPLE_MED}[OK] SUCCESS: Transfer Finished.{RESET}")
                
        except Exception as e:
            print(f"\n{RED}[!] ERROR: {e}{RESET}")

    print(f"\n{BOLD}{PURPLE_MED}COMPLETED SYNC TASK.{RESET}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaia Production Data Pipeline")
    parser.add_argument("--limit", "-n", type=int, help="Number of new granules to sync")
    parser.add_argument("--granule", type=str, help="Specific granule name to sync")
    args = parser.parse_args()
    
    production_sync(limit=args.limit)
