"""
GAIA: Download ALL V2 Mosaic tiles covering the Southern Cape region.
Uses earthaccess bounding box search instead of a curated tile list.

Usage:
    python src/download_region.py                   # discover only (dry run)
    python src/download_region.py --download        # actually download
    python src/download_region.py --download --limit 5  # test with 5 tiles
    
Login:
    First run will prompt you for your NASA Earthdata username & password.
    After that, credentials are saved to ~/.netrc so you won't be asked again.
    
    If you want to pre-configure:
        python -c "import earthaccess; earthaccess.login(persist=True)"
"""
import os
import sys
import argparse
import time
import uuid
import signal
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import earthaccess

# ── Project root ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── ANSI Colors ──
GREEN  = "\033[32m"
PURPLE = "\033[35m"
CYAN   = "\033[36m"
RED    = "\033[31m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ── Southern Cape Bounding Box ──
# Covers the Greater Cape Floristic Region
BBOX_WEST  = 18.0
BBOX_SOUTH = -35.5
BBOX_EAST  = 25.0
BBOX_NORTH = -31.5

# ── BBL (Bad Band List) ──
BBL = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]

# ── Concurrency ──
print_lock = threading.Lock()
stop_event = threading.Event()
worker_seats = queue.Queue()
stats_lock = threading.Lock()
total_downloaded_bytes = 0

def add_stats(n):
    global total_downloaded_bytes
    with stats_lock:
        total_downloaded_bytes += n

def safe_print(msg):
    with print_lock:
        tqdm.write(msg)

def signal_handler(sig, frame):
    if not stop_event.is_set():
        safe_print(f"\n{RED}[!] Interrupt received. Cleaning up...{RESET}")
        stop_event.set()

signal.signal(signal.SIGINT, signal_handler)


def downsample_netcdf(input_path, output_path, res_target, log_prefix):
    """BBL filter + spatial downsample (same logic as smart_sync_v2.py)"""
    import h5py

    good_indices = np.array([i for i, val in enumerate(BBL) if val == 1])
    factor = max(1, int(res_target // 5))

    with h5py.File(input_path, 'r') as src:
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

        b, h, w = cube.shape
        h_new = (h // factor) * factor
        w_new = (w // factor) * factor
        cube = cube[:, :h_new, :w_new]
        cube = cube.reshape(b, h_new // factor, factor, w_new // factor, factor).mean(axis=(2, 4))

        easting = easting[:w_new].reshape(-1, factor).mean(axis=1)
        northing = northing[:h_new].reshape(-1, factor).mean(axis=1)

        with h5py.File(output_path, 'w') as dst:
            grp_out = dst.create_group('reflectance')
            grp_out.create_dataset('reflectance', data=cube.astype(np.float32), compression=None)
            grp_out.create_dataset('wavelength', data=wavelengths.astype(np.float32))
            grp_out.create_dataset('fwhm', data=fwhm.astype(np.float32))
            dst.create_dataset('easting', data=easting.astype(np.float32))
            dst.create_dataset('northing', data=northing.astype(np.float32))
            if proj_var is not None:
                tm = dst.create_dataset('transverse_mercator', data=proj_var)
                for k, v in proj_attrs.items():
                    tm.attrs[k] = v
            grp_out['reflectance'].attrs['grid_mapping'] = 'transverse_mercator'
            dst.attrs['resolution_factor'] = factor
            dst.attrs['source_file'] = os.path.basename(input_path)


def download_worker(work_item):
    """Download + process one granule."""
    g, idx, total, session, local_dir, res = work_item
    links = g.data_links()

    rfl_links = [l for l in links if l.endswith("_RFL.nc")]
    if not rfl_links:
        safe_print(f"{RED}[!] No RFL .nc for granule {idx}{RESET}")
        return

    url = rfl_links[0]
    filename = os.path.basename(url)

    # Skip if already downloaded
    target_path = os.path.join(local_dir, filename)
    if os.path.exists(target_path):
        safe_print(f"{GREEN}[SKIP] {filename} (already exists){RESET}")
        return

    if stop_event.is_set():
        return

    seat = worker_seats.get()
    log_prefix = f"{BOLD}{PURPLE}[{idx}/{total}]{RESET}"
    uid = uuid.uuid4().hex[:8]
    tmp_dir = os.path.join(PROJECT_ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    temp_file = os.path.join(tmp_dir, f"temp_{uid}_{filename}")
    proc_file = os.path.join(tmp_dir, f"proc_{uid}_{filename}")

    try:
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            short = filename[:25]

            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"DWN {short}", position=seat, leave=False) as pbar:
                with open(temp_file, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if stop_event.is_set():
                            return
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                    add_stats(downloaded)

                pbar.set_description(f"PRC {short}")
                downsample_netcdf(temp_file, proc_file, res, log_prefix)

        if stop_event.is_set():
            return

        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(target_path):
            os.remove(target_path)
        os.replace(proc_file, target_path)
        safe_print(f"{GREEN}{log_prefix} SUCCESS: {filename}{RESET}")

    except Exception as e:
        if not stop_event.is_set():
            safe_print(f"{RED}{log_prefix} ERROR {filename}: {e}{RESET}")
    finally:
        worker_seats.put(seat)
        for f in [temp_file, proc_file]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass


def main():
    parser = argparse.ArgumentParser(description="Download ALL V2 mosaic tiles in the Southern Cape region")
    parser.add_argument("--download", action="store_true", help="Actually download (default is dry-run discovery)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of granules to download")
    parser.add_argument("--res", type=int, default=30, help="Target resolution in meters (default: 30)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel download workers")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "data", "bioscape", f"{args.res}m_v2")

    print(f"\n{BOLD}{PURPLE}{'='*60}")
    print(f"  GAIA — Southern Cape V2 Mosaic Download")
    print(f"{'='*60}{RESET}\n")
    print(f"  Bounding Box: ({BBOX_WEST}°E, {BBOX_SOUTH}°S) → ({BBOX_EAST}°E, {BBOX_NORTH}°S)")
    print(f"  Resolution:   {args.res}m")
    print(f"  Output:       {out_dir}")
    print()

    # ── Login ──
    print(f"{CYAN}[*] Logging into NASA Earthdata...{RESET}")
    print(f"{CYAN}    (First time? You'll be prompted for username & password.){RESET}")
    print(f"{CYAN}    (Credentials are saved to ~/.netrc for future use.){RESET}\n")
    earthaccess.login(persist=True)
    print(f"{GREEN}[✔] Login successful!{RESET}\n")

    # ── Search ──
    print(f"{CYAN}[*] Searching NASA CMR for V2 mosaic tiles in bounding box...{RESET}")
    granules = earthaccess.search_data(
        short_name="BioSCape_ANG_V02_L3_RFL_Mosaic_2427",
        version="2",
        bounding_box=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH)
    )

    if not granules:
        # Fallback: search without bbox and filter manually
        print(f"{CYAN}[*] Bbox search returned 0. Trying full dataset search...{RESET}")
        granules = earthaccess.search_data(
            short_name="BioSCape_ANG_V02_L3_RFL_Mosaic_2427",
            version="2"
        )
        print(f"{GREEN}[✔] Found {len(granules)} total V2 mosaic tiles in the dataset{RESET}")
    else:
        print(f"{GREEN}[✔] Found {len(granules)} tiles in the Southern Cape bounding box{RESET}")

    # ── Check existing ──
    os.makedirs(out_dir, exist_ok=True)
    existing = {f for f in os.listdir(out_dir) if f.endswith('.nc')}
    print(f"{CYAN}[*] Already downloaded: {len(existing)} files in {out_dir}{RESET}")

    # ── Dry run: just list the tiles ──
    if not args.download:
        print(f"\n{BOLD}{'─'*60}{RESET}")
        print(f"{BOLD}  DRY RUN — Tiles that would be downloaded:{RESET}")
        print(f"{'─'*60}")
        for i, g in enumerate(granules[:20]):
            try:
                name = g["umm"]["GranuleUR"]
                links = g.data_links()
                rfl = [l for l in links if l.endswith("_RFL.nc")]
                fn = os.path.basename(rfl[0]) if rfl else "?"
                status = "SKIP (exists)" if fn in existing else "NEW"
                print(f"  [{i+1:3d}] {name}  →  {status}")
            except:
                pass
        if len(granules) > 20:
            print(f"  ... and {len(granules)-20} more")
        
        new_count = 0
        for g in granules:
            try:
                rfl = [l for l in g.data_links() if l.endswith("_RFL.nc")]
                if rfl and os.path.basename(rfl[0]) not in existing:
                    new_count += 1
            except:
                pass
        print(f"\n  {GREEN}Total: {len(granules)} tiles, {new_count} new to download{RESET}")
        print(f"\n  {BOLD}To download, re-run with --download{RESET}\n")
        return

    # ── Download ──
    if args.limit:
        granules = granules[:args.limit]
        print(f"{CYAN}[*] Limited to {args.limit} granules{RESET}")

    for i in range(args.workers):
        worker_seats.put(i)

    session = earthaccess.get_requests_https_session()
    work_items = [(g, i+1, len(granules), session, out_dir, args.res) for i, g in enumerate(granules)]

    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(download_worker, work_items))
    except (KeyboardInterrupt, SystemExit):
        stop_event.set()
        print(f"\n{RED}[!] Cancelled.{RESET}")
        sys.exit(1)

    if not stop_event.is_set():
        elapsed = time.time() - start_time
        total_mb = total_downloaded_bytes / 1024 / 1024
        print(f"\n{BOLD}{GREEN}{'='*60}")
        print(f"  DOWNLOAD COMPLETE")
        print(f"  Time: {elapsed:.0f}s  |  Data: {total_mb:.0f} MB  |  Speed: {total_mb/max(elapsed,1):.1f} MB/s")
        print(f"  Files saved to: {out_dir}")
        print(f"{'='*60}{RESET}\n")

        nc_count = len([f for f in os.listdir(out_dir) if f.endswith('.nc')])
        print(f"  {BOLD}Next step: Generate heatmap with:{RESET}")
        print(f"  python src/infer_heatmap.py --nc_dir {out_dir}\n")


if __name__ == "__main__":
    main()
