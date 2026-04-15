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
                 band_mean=None, band_std=None):
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.bands_to_trim = bands_to_trim
        self.patches_per_granule = patches_per_granule
        self.augment = augment
        self.band_mean = band_mean   # (num_bands, 1, 1) or None
        self.band_std = band_std

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

    def _scan_granules(self):
        """Scan granule dimensions to build the patch index."""
        self.index = []   # list of (granule_idx, y_start, x_start) for each valid patch slot
        self.granule_meta = []

        for g_idx, path in enumerate(self.nc_files):
            try:
                with h5py.File(path, 'r') as f:
                    if 'reflectance/reflectance' in f:
                        shape = f['reflectance/reflectance'].shape
                    else:
                        shape = f['reflectance'].shape
                total_bands, h, w = shape
            except Exception:
                continue

            usable_bands = total_bands - self.bands_to_trim
            # Accept files with fewer bands -- we'll zero-pad in __getitem__
            min_required = self.num_bands // 2  # at least half the bands
            if usable_bands < min_required:
                print(f"  [skip] {os.path.basename(path)}: only {usable_bands} bands")
                continue

            p = self.patch_size
            if h < p or w < p:
                continue

            self.granule_meta.append({
                'path': path, 'h': h, 'w': w, 'bands': total_bands
            })

            # Generate random patch origins for this granule
            for _ in range(self.patches_per_granule):
                y = np.random.randint(0, h - p + 1)
                x = np.random.randint(0, w - p + 1)
                self.index.append((len(self.granule_meta) - 1, y, x))

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

        # Replace NaN/Inf with 0
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

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
        running_sum = torch.zeros(self.num_bands)
        running_sq = torch.zeros(self.num_bands)
        n_pixels = 0

        indices = np.random.choice(len(self), min(max_samples, len(self)), replace=False)
        for i, idx in enumerate(indices):
            # Temporarily disable augmentation + normalization
            old_aug, old_mean = self.augment, self.band_mean
            self.augment, self.band_mean = False, None
            patch = self[idx]
            self.augment, self.band_mean = old_aug, old_mean

            running_sum += patch.sum(dim=(1, 2))
            running_sq += (patch ** 2).sum(dim=(1, 2))
            n_pixels += patch.shape[1] * patch.shape[2]

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(indices)}] scanned")

        mean = running_sum / n_pixels
        std = ((running_sq / n_pixels) - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)

        self.band_mean = mean.reshape(self.num_bands, 1, 1)
        self.band_std = std.reshape(self.num_bands, 1, 1)
        print(f"[OK] Band stats computed. Mean range: [{mean.min():.1f}, {mean.max():.1f}]")
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

        patch = cube[:self.num_bands, y_idx-p:y_idx+p, x_idx-p:x_idx+p]
        patch = np.array(patch, dtype=np.float32)
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
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
