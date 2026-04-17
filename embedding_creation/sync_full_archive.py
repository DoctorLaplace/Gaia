"""
Sync Full AVIRIS-NG Archive for SSL Pre-training

Downloads ALL BioSCape AVIRIS-NG L2B granules (not just labeled ones),
applies BBL filtering + spatial downsampling, and saves locally.

This is a standalone script — it does NOT modify any existing Gaia code.

Usage:
    python -m embedding_creation.sync_full_archive                     # sync all
    python -m embedding_creation.sync_full_archive --limit 10          # first 10 new
    python -m embedding_creation.sync_full_archive --res 30            # 30m resolution
    python -m embedding_creation.sync_full_archive --inventory labeled_inventory.txt  # labeled only
"""
import os
import sys
import time
import argparse
import subprocess
import uuid
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import credentials (same as existing pipeline, not modifying it)
try:
    from src.credentials import EARTHDATA_USERNAME, EARTHDATA_PASSWORD
except ImportError:
    EARTHDATA_USERNAME = EARTHDATA_PASSWORD = None

# ── Bad Band List (BBL) for AVIRIS-NG 425 bands ──
# 0 = bad (atmospheric absorption), 1 = good
# After filtering: 373 good bands
BBL = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,
       0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,0,0]

GOOD_INDICES = np.array([i for i, val in enumerate(BBL) if val == 1])


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cleanup_tmp(tmp_dir):
    """Removes orphaned temp and proc files to prevent disk exhaustion."""
    print(f"{Colors.OKBLUE}[*] Cleaning up temporary directory...{Colors.ENDC}")
    freed_bytes = 0
    if os.path.exists(tmp_dir):
        for f in os.listdir(tmp_dir):
            if f.startswith("temp_") or f.startswith("proc_"):
                filepath = os.path.join(tmp_dir, f)
                try:
                    size = os.path.getsize(filepath)
                    os.remove(filepath)
                    freed_bytes += size
                    print(f"  {Colors.WARNING}[🧹] Removed orphan file: {f} ({size / 1024**2:.1f} MB){Colors.ENDC}")
                except Exception as e:
                    print(f"  {Colors.FAIL}[!] Failed to remove {f}: {e}{Colors.ENDC}")
    
    if freed_bytes > 0:
        print(f"{Colors.OKGREEN}[OK] Freed {freed_bytes / 1024**3:.2f} GB of disk space.{Colors.ENDC}")
    else:
        print(f"{Colors.OKGREEN}[OK] Temporary directory is clean.{Colors.ENDC}")


def downsample_netcdf(input_path, output_path, res_target, tile_size):
    """Apply BBL filtering and spatial downsampling with memory-efficient tiling."""
    import h5py
    import gc

    factor = max(1, int(res_target // 5))
    print(f"  {Colors.OKCYAN}[⚙️] Processing NetCDF (Resolution: {res_target}m, Downsample Factor: {factor}x){Colors.ENDC}")

    with h5py.File(input_path, 'r') as src:
        wavelengths = src['reflectance/wavelength'][()]
        fwhm = src['reflectance/fwhm'][()]
        
        # Determine source dataset
        if 'reflectance/reflectance' in src:
            src_cube = src['reflectance/reflectance']
        else:
            src_cube = src['reflectance']
            
        b_total, h_total, w_total = src_cube.shape

        # Filter BBL
        wavelengths = wavelengths[GOOD_INDICES]
        fwhm = fwhm[GOOD_INDICES]
        b_new = len(GOOD_INDICES)

        # Calculate new dimensions
        if factor > 1:
            h_new = (h_total // factor) * factor
            w_new = (w_total // factor) * factor
            out_h = h_new // factor
            out_w = w_new // factor
        else:
            out_h, out_w = h_total, w_total
            h_new, w_new = h_total, w_total

        # GeoTransform adjustments
        gt_str = None
        if 'projection' in src and 'GeoTransform' in src['projection'].attrs:
            gt_str = src['projection'].attrs['GeoTransform']
        elif 'reflectance' in src and 'GeoTransform' in src['reflectance'].attrs:
             gt_str = src['reflectance'].attrs['GeoTransform']

        if gt_str is not None:
            if isinstance(gt_str, bytes):
                gt_str = gt_str.decode()
            if isinstance(gt_str, str):
                gt = [float(x) for x in gt_str.split()]
            else:
                 gt = list(gt_str)
            gt[1] *= factor
            gt[5] *= factor
            gt_out = " ".join(str(v) for v in gt)
        else:
            gt_out = None
            
        proj_wkt = None
        if 'projection' in src and 'spatial_ref' in src['projection'].attrs:
             proj_wkt = src['projection'].attrs['spatial_ref']

        # Setup Tile Processing
        tile_h = max(factor, (tile_size // factor) * factor) # Ensure tile height is a multiple of factor
        print(f"  {Colors.OKBLUE}[ℹ️] Memory Config: Tiled processing (Size: {tile_h} lines/batch){Colors.ENDC}")

        with h5py.File(output_path, 'w') as dst:
            proj_ds = dst.create_dataset('projection', data=np.uint8(0))
            if gt_out:
                proj_ds.attrs['GeoTransform'] = gt_out
            if proj_wkt is not None:
                proj_ds.attrs['spatial_ref'] = proj_wkt
                
            grp = dst.create_group('reflectance')
            
            # Pre-allocate the output dataset chunked for efficient writing
            dst_cube = grp.create_dataset(
                'reflectance', 
                shape=(b_new, out_h, out_w),
                dtype=np.float32,
                chunks=(b_new, min(64, out_h), min(64, out_w)),
                compression='gzip',
                compression_opts=4
            )
            
            grp.create_dataset('wavelength', data=wavelengths.astype(np.float32))
            grp.create_dataset('fwhm', data=fwhm.astype(np.float32))

            # Process in tiles along Y axis
            print(f"  {Colors.OKCYAN}[⏳] Processing tiles (Total Height: {h_new})...{Colors.ENDC}")
            with tqdm(total=h_new, unit='lines', desc="Processing", leave=False) as pbar:
                for y in range(0, h_new, tile_h):
                    y_end = min(y + tile_h, h_new)
                    
                    # Read chunk (only Good Indices)
                    cube_chunk = src_cube[GOOD_INDICES, y:y_end, :w_new]
                    
                    if factor > 1:
                        # Downsample current chunk
                        chunk_h = y_end - y
                        cube_chunk = cube_chunk.reshape(b_new, chunk_h // factor, factor, w_new // factor, factor)
                        cube_chunk = cube_chunk.mean(axis=(2, 4))
                    
                    # Write to output
                    out_y = y // factor
                    out_y_end = out_y + cube_chunk.shape[1]
                    dst_cube[:, out_y:out_y_end, :] = cube_chunk.astype(np.float32)
                    
                    pbar.update(y_end - y)
                    
                    # Memory management
                    del cube_chunk
                    gc.collect()


def sync(args):
    import earthaccess
    import yaml

    # Load config to get directory paths
    config_path = os.path.join(PROJECT_ROOT, "embedding_creation", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    tile_size = cfg['data'].get('tile_size', 1000)

    inventory_path = os.path.join(PROJECT_ROOT, args.inventory)
    if not os.path.exists(inventory_path):
        print(f"{Colors.FAIL}[!] Inventory not found: {inventory_path}{Colors.ENDC}")
        return

    # Downloads go to the SSL-only directory (NOT the existing labeled data dir)
    res_str = str(args.res)
    output_dir = os.path.join(PROJECT_ROOT, cfg['data']['ssl_nc_dir'].replace('{res}', res_str))
    labeled_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'].replace('{res}', res_str))
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(PROJECT_ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"{Colors.HEADER}===================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}  AVIRIS-NG Native Pipeline Sync Tool{Colors.ENDC}")
    print(f"{Colors.HEADER}===================================================={Colors.ENDC}")
    
    # Pre-flight cleanup
    cleanup_tmp(tmp_dir)

    print(f"\n{Colors.OKBLUE}[*] Directories:{Colors.ENDC}")
    print(f"    - Output Folder:  {Colors.BOLD}{output_dir}{Colors.ENDC}")
    print(f"    - Labeled Folder: {labeled_dir} (Read-only)")

    # Auth
    if EARTHDATA_USERNAME and EARTHDATA_PASSWORD:
        os.environ['EARTHDATA_USERNAME'] = EARTHDATA_USERNAME
        os.environ['EARTHDATA_PASSWORD'] = EARTHDATA_PASSWORD
        earthaccess.login(strategy="environment", persist=True)
    else:
        earthaccess.login(persist=True)

    # Check existing in BOTH directories (don't re-download what's already in labeled)
    existing = set()
    for d in [output_dir, labeled_dir]:
        if os.path.isdir(d):
            existing.update(f for f in os.listdir(d) if f.endswith('.nc'))

    # Load targets
    with open(inventory_path, 'r') as f:
        target_ids = [l.strip() for l in f if l.strip()]

    normalize = lambda nid: nid.replace("BioSCape_AVNG_L2B_BRDF_GCFR.", "")
    new_ids = [nid for nid in target_ids if normalize(nid) not in existing]

    print(f"\n{Colors.OKBLUE}[*] Synchronization Queue:{Colors.ENDC}")
    print(f"    - Total Inventory:    {len(target_ids)}")
    print(f"    - Already Downloaded: {len(existing)} (combined dirs)")
    print(f"    - Remaining to Sync:  {Colors.BOLD}{len(new_ids)}{Colors.ENDC}")

    if args.limit:
        new_ids = new_ids[:args.limit]
        print(f"    - Limit Applied:      {Colors.WARNING}{len(new_ids)}{Colors.ENDC}")

    if not new_ids:
        print(f"\n{Colors.OKGREEN}[OK] All granules already synced!{Colors.ENDC}")
        return

    # Discover on NASA CMR
    print(f"\n{Colors.OKBLUE}[*] Discovering {len(new_ids)} granules on NASA CMR...{Colors.ENDC}")
    granules = []
    for i in range(0, len(new_ids), 50):
        chunk = new_ids[i:i + 50]
        results = earthaccess.search_data(
            short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385",
            granule_ur=chunk
        )
        granules.extend(results)
    print(f"{Colors.OKGREEN}[OK] Found and authenticated {len(granules)} actionable granules.{Colors.ENDC}")

    # Download and process
    session = earthaccess.get_requests_https_session()
    total_bytes = 0
    start = time.time()

    print(f"\n{Colors.HEADER}--- Starting Batch Processing ---{Colors.ENDC}")
    
    for i, g in enumerate(granules):
        url = g.data_links()[0]
        filename = os.path.basename(url)
        uid = uuid.uuid4().hex[:8]
        temp_file = os.path.join(tmp_dir, f"temp_{uid}_{filename}")
        processed_file = os.path.join(tmp_dir, f"proc_{uid}_{filename}")

        print(f"\n{Colors.BOLD}[{i+1}/{len(granules)}] {filename}{Colors.ENDC}")

        try:
            with session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True,
                         desc=f"  {Colors.OKCYAN}[↓] Downloading{Colors.ENDC}", leave=False, 
                         bar_format='{l_bar}{bar:20}{r_bar}') as pbar:
                    with open(temp_file, 'wb') as dest:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                dest.write(chunk)
                                pbar.update(len(chunk))
                                total_bytes += len(chunk)

            # Tiled Processing execution
            downsample_netcdf(temp_file, processed_file, args.res, tile_size)
            
            target = os.path.join(output_dir, filename)
            if os.path.exists(target):
                os.remove(target)
            os.replace(processed_file, target)
            print(f"  {Colors.OKGREEN}[✓] Successfully deployed to core directories.{Colors.ENDC}")

        except Exception as e:
            print(f"  {Colors.FAIL}[!] Critical Error encountered over granule: {e}{Colors.ENDC}")
        finally:
            for f in [temp_file, processed_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass

    elapsed = time.time() - start
    print(f"\n{Colors.HEADER}===================================================={Colors.ENDC}")
    print(f"{Colors.OKGREEN}[OK] Routine Completed.{Colors.ENDC}")
    print(f"     Total Data Transferred: {total_bytes/1024**3:.2f} GB")
    print(f"     Total Time Elapsed:     {elapsed/60:.1f} minutes")
    print(f"{Colors.HEADER}===================================================={Colors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync AVIRIS-NG archive for SSL")
    parser.add_argument("--inventory", default="full_inventory.txt",
                        help="Inventory file (default: full_inventory.txt for ALL granules)")
    parser.add_argument("--limit", type=int,
                        help="Only download first N new granules")
    parser.add_argument("--res", type=int, default=30,
                        help="Target resolution in meters (default: 30)")
    args = parser.parse_args()
    sync(args)
