"""
AVIRIS-NG Native Spectral Datasets

Two dataset classes for the foundation model pipeline:
  1. AvirisSSLDataset  — Unlabeled random patches for self-supervised pre-training
  2. AvirisRichnessDataset — Labeled patches for richness fine-tuning (no spectral resampling)

Both read 30m BBL-filtered NetCDFs (373 bands) and trim to 370 bands.
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import threading


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class AvirisSSLDataset(Dataset):
    """
    Self-supervised pre-training dataset.
    Extracts random spatial patches from AVIRIS-NG flightlines.
    No labels needed -- the reconstruction target IS the patch itself.

    Accepts one or more data directories so pre-training can read from
    both the existing labeled granules AND a separate SSL-only archive.
    """
    def __init__(self, nc_dirs, patch_size=16, num_bands=370, bands_to_trim=3,
                 patches_per_granule=200, max_granules=None, augment=True,
                 band_mean=None, band_std=None, nodata_threshold=-9000):
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.bands_to_trim = bands_to_trim
        self.patches_per_granule = patches_per_granule
        self.augment = augment
        self.band_mean = band_mean   # (num_bands, 1, 1) or None
        self.band_std = band_std
        self.nodata_threshold = nodata_threshold

        # Accept a single string or a list of directories
        if isinstance(nc_dirs, str):
            nc_dirs = [nc_dirs]

        # Discover all NetCDF files across all directories
        self.nc_files = []
        for nc_dir in nc_dirs:
            if os.path.isdir(nc_dir):
                self.nc_files.extend(sorted([
                    os.path.join(nc_dir, f)
                    for f in os.listdir(nc_dir) if f.endswith('.nc')
                ]))
        # Deduplicate by filename (in case same granule exists in both dirs)
        seen = set()
        unique = []
        for path in self.nc_files:
            name = os.path.basename(path)
            if name not in seen:
                unique.append(path)
                seen.add(name)
        self.nc_files = unique
        if max_granules:
            self.nc_files = self.nc_files[:max_granules]

        # Pre-scan: find valid (y, x) ranges for each granule
        self._scan_granules()

        # File handle cache (per-thread for safety)
        self._cache = {}
        self._cache_lock = threading.Lock()

    def _scan_granules(self, cache_path=None):
        """Scan granule dimensions to build the patch index with caching and parallelism."""
        if cache_path is None:
            cache_path = os.path.join("embedding_creation", "checkpoints", "dataset_index_cache.pth")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # 1. Try to load cache
        if os.path.exists(cache_path):
            print(f"{Colors.OKGREEN}[OK] Loading patch index from cache: {cache_path}{Colors.ENDC}")
            try:
                # Use weights_only=False because we trust this internal pth file
                cache = torch.load(cache_path)
                # Verify cache is for the SAME set of files and patches_per_granule setting
                if cache.get('nc_files') == self.nc_files and cache.get('patches_per_granule') == self.patches_per_granule:
                    self.index = cache['index']
                    self.granule_meta = cache['granule_meta']
                    print(f"  {Colors.OKCYAN}[OK] Found {len(self.index)} cached patches across {len(self.granule_meta)} granules.{Colors.ENDC}")
                    return
                else:
                    print(f"  {Colors.WARNING}[!] Cache mismatch (files or params changed). Re-scanning...{Colors.ENDC}")
            except Exception as e:
                print(f"  {Colors.WARNING}[!] Failed to load cache: {e}. Re-scanning...{Colors.ENDC}")

        self.index = []
        self.granule_meta = []

        # 2. Parallel Scan
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        index_lock = threading.Lock()
        
        def scan_single_granule(path):
            try:
                with h5py.File(path, 'r') as f:
                    if 'reflectance/reflectance' in f:
                        shape = f['reflectance/reflectance'].shape
                    else:
                        shape = f['reflectance'].shape
                total_bands, h, w = shape
            except Exception:
                return None

            usable_bands = total_bands - self.bands_to_trim
            min_required = self.num_bands // 2
            if usable_bands < min_required:
                return None

            p = self.patch_size
            if h < p or w < p:
                return None

            # Local meta for this granule
            meta = {'path': path, 'h': h, 'w': w, 'bands': total_bands}
            
            # Find patches
            local_patches = []
            with h5py.File(path, 'r') as f:
                if 'reflectance/reflectance' in f:
                    cube = f['reflectance/reflectance']
                else:
                    cube = f['reflectance']
                
                attempts = 0
                patches_found = 0
                max_attempts = self.patches_per_granule * 10
                
                while patches_found < self.patches_per_granule and attempts < max_attempts:
                    y = np.random.randint(0, h - p + 1)
                    x = np.random.randint(0, w - p + 1)
                    
                    coords = [
                        (y+p//2, x+p//2), (y, x), (y, x+p-1), 
                        (y+p-1, x), (y+p-1, x+p-1)
                    ]
                    
                    is_valid = True
                    for cy, cx in coords:
                        if cube[0, cy, cx] <= self.nodata_threshold:
                            is_valid = False
                            break
                    
                    if is_valid:
                        local_patches.append((y, x))
                        patches_found += 1
                    attempts += 1
            
            return meta, local_patches

        print(f"{Colors.OKBLUE}[*] Parallel Scanning {len(self.nc_files)} granules (using 8 threads)...{Colors.ENDC}")
        
        # Use 8 threads to match nproc and avoid OS contention
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Map the scan function across files
            results = list(tqdm(executor.map(scan_single_granule, self.nc_files), 
                               total=len(self.nc_files), desc="Scanning Granules"))

        # 3. Assemble Results
        for res in results:
            if res is not None:
                meta, patches = res
                self.granule_meta.append(meta)
                g_idx = len(self.granule_meta) - 1
                for y, x in patches:
                    self.index.append((g_idx, y, x))

        # 4. Save Cache
        print(f"{Colors.OKBLUE}[*] Saving patch index to cache...{Colors.ENDC}")
        torch.save({
            'nc_files': self.nc_files,
            'patches_per_granule': self.patches_per_granule,
            'index': self.index,
            'granule_meta': self.granule_meta
        }, cache_path)
        print(f"{Colors.OKGREEN}[OK] {len(self.index)} valid patches cached.{Colors.ENDC}")

    def _get_file(self, path):
        """Thread-safe cached file handle."""
        with self._cache_lock:
            if path not in self._cache:
                self._cache[path] = h5py.File(path, 'r',
                                              rdcc_nbytes=32*1024*1024,
                                              rdcc_nslots=11213)
            return self._cache[path]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        g_idx, y, x = self.index[idx]
        meta = self.granule_meta[g_idx]
        p = self.patch_size

        f = self._get_file(meta['path'])
        if 'reflectance/reflectance' in f:
            cube = f['reflectance/reflectance']
        else:
            cube = f['reflectance']

        # Extract patch: (bands, patch_size, patch_size)
        available_bands = min(cube.shape[0], self.num_bands)
        patch = cube[:available_bands, y:y+p, x:x+p]
        patch = np.array(patch, dtype=np.float32)
        
        # Replace NaN/Inf AND NoData (-10000) with 0.0
        # Reflectance should be positive. Anything below threshold is zeroed.
        mask = (patch <= self.nodata_threshold) | np.isnan(patch) | np.isinf(patch)
        patch[mask] = 0.0
        
        # Also clip high values (specular reflections/noise) to a sane max (e.g. 1.5 reflectance)
        # Note: Some AVIRIS data is scaled by 10000. We'll handle that in normalization.
        # But we'll clip to 15000 if it looks like scaled data, or 1.5 if raw.
        if patch.max() > 20: 
            patch = np.clip(patch, 0, 15000)
        else:
            patch = np.clip(patch, 0, 1.5)

        # Zero-pad if fewer bands than expected
        if available_bands < self.num_bands:
            pad = np.zeros((self.num_bands - available_bands, p, p), dtype=np.float32)
            patch = np.concatenate([patch, pad], axis=0)

        patch = torch.from_numpy(patch)

        # Per-band normalization
        if self.band_mean is not None and self.band_std is not None:
            patch = (patch - self.band_mean) / self.band_std

        # Augmentation: random rotation + flip
        if self.augment:
            k = np.random.randint(0, 4)
            patch = torch.rot90(patch, k=k, dims=(1, 2))
            if np.random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])

        return patch

    def compute_band_stats(self, max_samples=2000):
        """Compute per-band mean/std across random patches for normalization."""
        print(f"[*] Computing band statistics from {min(max_samples, len(self))} patches...")
        print(f"    (Ignoring NoData values <= {self.nodata_threshold})")
        
        running_sum = torch.zeros(self.num_bands)
        running_sq = torch.zeros(self.num_bands)
        n_pixels_per_band = torch.zeros(self.num_bands)

        indices = np.random.choice(len(self), min(max_samples, len(self)), replace=False)
        for i, idx in enumerate(indices):
            # Temporarily disable augmentation + normalization
            old_aug, old_mean = self.augment, self.band_mean
            self.augment, self.band_mean = False, None
            patch = self[idx] # (C, H, W)
            self.augment, self.band_mean = old_aug, old_mean

            # Only count non-zero pixels (since we zeroed the nodata)
            valid_mask = (patch > 0).float()
            
            running_sum += (patch * valid_mask).sum(dim=(1, 2))
            running_sq += ((patch ** 2) * valid_mask).sum(dim=(1, 2))
            n_pixels_per_band += valid_mask.sum(dim=(1, 2))

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(indices)}] scanned")

        # Avoid div by zero
        n_pixels_per_band = n_pixels_per_band.clamp(min=1)
        mean = running_sum / n_pixels_per_band
        std = ((running_sq / n_pixels_per_band) - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)

        self.band_mean = mean.reshape(self.num_bands, 1, 1)
        self.band_std = std.reshape(self.num_bands, 1, 1)
        print(f"[OK] Band stats computed units. Mean range: [{mean.min():.1f}, {mean.max():.1f}]")
        return self.band_mean, self.band_std

    def close(self):
        with self._cache_lock:
            for f in self._cache.values():
                try:
                    f.close()
                except:
                    pass
            self._cache.clear()


class AvirisRichnessDataset(Dataset):
    """
    Fine-tuning dataset for richness regression.
    Maps richness sites to AVIRIS-NG granules and extracts 370-band patches.
    No spectral resampling — uses native AVIRIS bands.
    """
    def __init__(self, nc_dir, richness_csv, patch_size=16, num_bands=370,
                 bands_to_trim=3, augment=False, max_granules=None,
                 band_mean=None, band_std=None):
        import pandas as pd
        from pyproj import Transformer

        self.patch_size = patch_size
        self.num_bands = num_bands
        self.bands_to_trim = bands_to_trim
        self.augment = augment
        self.band_mean = band_mean
        self.band_std = band_std

        self._cache = {}
        self._cache_lock = threading.Lock()

        # Load richness labels
        df = pd.read_csv(richness_csv)
        df = df.rename(columns={
            'Latitude': 'lat', 'Longitude': 'lon',
            'richness': 'richness', 'Richness': 'richness'
        })

        # Discover granules
        nc_files = sorted([
            os.path.join(nc_dir, f)
            for f in os.listdir(nc_dir) if f.endswith('.nc')
        ])
        if max_granules:
            nc_files = nc_files[:max_granules]

        # Map richness sites to granules
        self.mappings = []  # [(nc_path, lat, lon, richness)]
        crs_tx = Transformer.from_crs("epsg:4326", "epsg:32734", always_xy=True)

        print(f"[*] Mapping {len(df)} sites across {len(nc_files)} granules...")
        for nc_path in nc_files:
            try:
                with h5py.File(nc_path, 'r') as f:
                    if 'reflectance/reflectance' in f:
                        shape = f['reflectance/reflectance'].shape
                    else:
                        shape = f['reflectance'].shape
                    bands, h, w = shape

                    gt_str = None
                    if 'projection' in f and 'GeoTransform' in f['projection'].attrs:
                        gt_str = f['projection'].attrs['GeoTransform']
                    elif 'reflectance' in f and 'GeoTransform' in f['reflectance'].attrs:
                        gt_str = f['reflectance'].attrs['GeoTransform']

                    if gt_str is None:
                        continue

                    if isinstance(gt_str, bytes):
                        gt_str = gt_str.decode()
                    if isinstance(gt_str, str):
                        gt = [float(x) for x in gt_str.split()]
                    else:
                        gt = list(gt_str)

                    origin_x, pixel_w = gt[0], gt[1]
                    origin_y, pixel_h = gt[3], gt[5]

                p = patch_size // 2
                for _, row in df.iterrows():
                    easting, northing = crs_tx.transform(row['lon'], row['lat'])
                    x_idx = int((easting - origin_x) / pixel_w)
                    y_idx = int((northing - origin_y) / pixel_h)
                    if (p <= y_idx < h - p) and (p <= x_idx < w - p):
                        self.mappings.append((
                            nc_path, float(row['lat']), float(row['lon']),
                            float(row['richness'])
                        ))
            except Exception:
                continue

        # Deduplicate by coordinate
        seen = set()
        unique = []
        for m in self.mappings:
            key = (round(m[1], 5), round(m[2], 5))
            if key not in seen:
                unique.append(m)
                seen.add(key)
        self.mappings = unique
        print(f"[OK] Found {len(self.mappings)} unique richness sites.")

    def _get_file(self, path):
        with self._cache_lock:
            if path not in self._cache:
                self._cache[path] = h5py.File(path, 'r',
                                              rdcc_nbytes=32*1024*1024,
                                              rdcc_nslots=11213)
            return self._cache[path]

    def __len__(self):
        return len(self.mappings)

    def __getitem__(self, idx):
        from pyproj import Transformer
        nc_path, lat, lon, richness = self.mappings[idx]
        p = self.patch_size // 2

        f = self._get_file(nc_path)

        # Get geo info
        if 'projection' in f and 'GeoTransform' in f['projection'].attrs:
            gt_str = f['projection'].attrs['GeoTransform']
        else:
            gt_str = f['reflectance'].attrs['GeoTransform']
        if isinstance(gt_str, bytes):
            gt_str = gt_str.decode()
        if isinstance(gt_str, str):
            gt = [float(x) for x in gt_str.split()]
        else:
            gt = list(gt_str)
        origin_x, pixel_w = gt[0], gt[1]
        origin_y, pixel_h = gt[3], gt[5]

        crs_tx = Transformer.from_crs("epsg:4326", "epsg:32734", always_xy=True)
        easting, northing = crs_tx.transform(lon, lat)
        x_idx = int((easting - origin_x) / pixel_w)
        y_idx = int((northing - origin_y) / pixel_h)

        if 'reflectance/reflectance' in f:
            cube = f['reflectance/reflectance']
        else:
            cube = f['reflectance']

        # Extract patch: (bands, patch_size, patch_size)
        available_bands = min(cube.shape[0], self.num_bands)
        patch = cube[:available_bands, y_idx-p:y_idx+p, x_idx-p:x_idx+p]
        patch = np.array(patch, dtype=np.float32)
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle band mismatch (zero-pad if cube has fewer bands than num_bands)
        if patch.shape[0] < self.num_bands:
            full_patch = torch.zeros(self.num_bands, self.patch_size, self.patch_size)
            full_patch[:patch.shape[0]] = torch.from_numpy(patch)
            patch = full_patch
        else:
            patch = torch.from_numpy(patch)


        # Ensure correct spatial size (edge cases)
        if patch.shape[1] != self.patch_size or patch.shape[2] != self.patch_size:
            patch = torch.zeros(self.num_bands, self.patch_size, self.patch_size)

        if self.band_mean is not None and self.band_std is not None:
            patch = (patch - self.band_mean) / self.band_std

        if self.augment:
            k = np.random.randint(0, 4)
            patch = torch.rot90(patch, k=k, dims=(1, 2))
            if np.random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])

        return patch, torch.tensor([richness], dtype=torch.float32)

    def close(self):
        with self._cache_lock:
            for fh in self._cache.values():
                try:
                    fh.close()
                except:
                    pass
            self._cache.clear()
