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


def downsample_netcdf(input_path, output_path, res_target):
    """Apply BBL filtering and spatial downsampling."""
    import h5py

    factor = max(1, int(res_target // 5))

    with h5py.File(input_path, 'r') as src:
        cube = src['reflectance/reflectance'][()]
        wavelengths = src['reflectance/wavelength'][()]
        fwhm = src['reflectance/fwhm'][()]

        # BBL filter
        cube = cube[GOOD_INDICES]
        wavelengths = wavelengths[GOOD_INDICES]
        fwhm = fwhm[GOOD_INDICES]

        # Spatial downsampling
        if factor > 1:
            b, h, w = cube.shape
            h_new = (h // factor) * factor
            w_new = (w // factor) * factor
            cube = cube[:, :h_new, :w_new]
            cube = cube.reshape(b, h_new // factor, factor, w_new // factor, factor)
            cube = cube.mean(axis=(2, 4))

        # GeoTransform
        gt_str = src['projection'].attrs.get('GeoTransform', None)
        if gt_str is not None:
            if isinstance(gt_str, bytes):
                gt_str = gt_str.decode()
            gt = [float(x) for x in gt_str.split()]
            gt[1] *= factor
            gt[5] *= factor
            gt_out = " ".join(str(v) for v in gt)
        else:
            gt_out = None
        proj_wkt = src['projection'].attrs.get('spatial_ref', None)

    with h5py.File(output_path, 'w') as dst:
        proj_ds = dst.create_dataset('projection', data=np.uint8(0))
        if gt_out:
            proj_ds.attrs['GeoTransform'] = gt_out
        if proj_wkt is not None:
            proj_ds.attrs['spatial_ref'] = proj_wkt
        grp = dst.create_group('reflectance')
        grp.create_dataset('reflectance', data=cube.astype(np.float32),
                          compression='gzip', compression_opts=4)
        grp.create_dataset('wavelength', data=wavelengths.astype(np.float32))
        grp.create_dataset('fwhm', data=fwhm.astype(np.float32))


def sync(args):
    import earthaccess
    import yaml

    # Load config to get directory paths
    config_path = os.path.join(PROJECT_ROOT, "embedding_creation", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    inventory_path = os.path.join(PROJECT_ROOT, args.inventory)
    if not os.path.exists(inventory_path):
        print(f"[!] Inventory not found: {inventory_path}")
        return

    # Downloads go to the SSL-only directory (NOT the existing labeled data dir)
    res_str = str(args.res)
    output_dir = os.path.join(PROJECT_ROOT, cfg['data']['ssl_nc_dir'].replace('{res}', res_str))
    labeled_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'].replace('{res}', res_str))
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(PROJECT_ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"[*] Downloads go to: {output_dir}")
    print(f"    (existing labeled data in {labeled_dir} is NOT modified)")

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

    print(f"[*] Inventory: {len(target_ids)} total, {len(existing)} already downloaded "
          f"(across both dirs), {len(new_ids)} remaining")

    if args.limit:
        new_ids = new_ids[:args.limit]

    if not new_ids:
        print("[OK] All granules already synced.")
        return

    # Discover on NASA CMR
    print(f"[*] Discovering {len(new_ids)} granules on NASA CMR...")
    granules = []
    for i in range(0, len(new_ids), 50):
        chunk = new_ids[i:i + 50]
        results = earthaccess.search_data(
            short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385",
            granule_name=chunk
        )
        granules.extend(results)
    print(f"[OK] Found {len(granules)} granules.")

    # Download and process
    session = earthaccess.get_requests_https_session()
    total_bytes = 0
    start = time.time()

    for i, g in enumerate(granules):
        url = g.data_links()[0]
        filename = os.path.basename(url)
        uid = uuid.uuid4().hex[:8]
        temp_file = os.path.join(tmp_dir, f"temp_{uid}_{filename}")
        processed_file = os.path.join(tmp_dir, f"proc_{uid}_{filename}")

        print(f"\n[{i+1}/{len(granules)}] {filename}")

        try:
            with session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True,
                         desc="Download", leave=False) as pbar:
                    with open(temp_file, 'wb') as dest:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                dest.write(chunk)
                                pbar.update(len(chunk))
                                total_bytes += len(chunk)

            downsample_netcdf(temp_file, processed_file, args.res)
            target = os.path.join(output_dir, filename)
            if os.path.exists(target):
                os.remove(target)
            os.replace(processed_file, target)
            print(f"  [OK] Saved to {output_dir}")

        except Exception as e:
            print(f"  [!] Error: {e}")
        finally:
            for f in [temp_file, processed_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass

    elapsed = time.time() - start
    print(f"\n[OK] Sync complete. {total_bytes/1024**2:.0f} MB in {elapsed:.0f}s")


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
