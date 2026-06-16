import os
import re
import threading
import numpy as np
import pandas as pd
import torch
import rasterio
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from pyproj import Transformer
from tqdm import tqdm

class SingleEagleTiffDataset:
    """
    Dataset to handle a single NASA BioSCape AVIRIS-NG Level-3 EAGLE 30m GeoTIFF tile.
    Handles coordinate projection and lazy spectral resampling to the 200 foundation bands.
    """
    def __init__(self, tif_path, patch_size=16, mask_water_vapor=True):
        self.tif_path = tif_path
        self.patch_size = patch_size
        self.mask_water_vapor = mask_water_vapor
        
        self.src = None
        self._load_metadata()

    def _ensure_open(self):
        if self.src is not None:
            return
        self.src = rasterio.open(self.tif_path)

    def _load_metadata(self):
        was_open = self.src is not None
        self._ensure_open()
        
        # Extent bounds
        self.bounds = self.src.bounds
        self.width = self.src.width
        self.height = self.src.height
        self.transform = self.src.transform
        self.count = self.src.count
        
        # Extract wavelengths from band descriptions
        self.wavelengths = []
        for d in self.src.descriptions:
            if not d:
                continue
            # Try to match 'reflectance_wavelength=X' (native)
            m_nat = re.search(r'wavelength=([\d.]+)', d)
            if m_nat:
                self.wavelengths.append(float(m_nat.group(1)))
                continue
            # Try to match 'EAGLE_Xnm' (10nm FWHM)
            m_fwhm = re.search(r'EAGLE_(\d+)nm', d)
            if m_fwhm:
                self.wavelengths.append(float(m_fwhm.group(1)))
                continue
                
        # If we failed to parse descriptions, fallback to linear spacing based on count
        if len(self.wavelengths) != self.count:
            if self.count == 425:
                self.wavelengths = np.linspace(377.2, 2500.8, 425)
            elif self.count == 180:
                self.wavelengths = np.linspace(400.0, 2480.0, 180)
            else:
                self.wavelengths = np.linspace(400.0, 2500.0, self.count)
        else:
            self.wavelengths = np.array(self.wavelengths)

        # CRS Projection Transformer
        crs = self.src.crs
        self.transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
        self.inverse_transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
        
        if not was_open:
            self.close()

    def get_bounds_wgs84(self):
        """Returns bounds in WGS84 coordinates: (min_lon, min_lat, max_lon, max_lat)"""
        lon1, lat1 = self.inverse_transformer.transform(self.bounds.left, self.bounds.bottom)
        lon2, lat2 = self.inverse_transformer.transform(self.bounds.right, self.bounds.top)
        return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)

    def mask_patch(self, patch, wavelengths):
        """Zero out water vapor bands (Atmospheric Cleaning)."""
        BAD_RANGES = [(1340, 1480), (1780, 1970)]
        for lo, hi in BAD_RANGES:
            mask = (wavelengths >= lo) & (wavelengths <= hi)
            patch[mask, :, :] = 0.0
        return patch

    def resample_to_foundation(self, patch_raw, current_wavs):
        """Resample spectral bands to match the 200-band EnMAP foundation model."""
        target_wavs = np.linspace(400, 2450, 200)
        
        # Replace NaNs, Infs, NoData fill values (less than 0), and boundary noise with 0.0 before interpolation
        patch_raw = np.where(np.isnan(patch_raw) | np.isinf(patch_raw) | (patch_raw < 0.0), 0.0, patch_raw)
        
        # Resample along spectral dimension (channel-last for interpolation)
        patch_hwc = np.transpose(patch_raw, (1, 2, 0))
        f = interp1d(current_wavs, patch_hwc, axis=-1, kind='linear',
                     fill_value=0.0, bounds_error=False)
        resampled_hwc = f(target_wavs)
        patch_tensor = torch.from_numpy(np.transpose(resampled_hwc, (2, 0, 1))).float()
        
        if self.mask_water_vapor:
            patch_tensor = self.mask_patch(patch_tensor, target_wavs)
            
        return patch_tensor

    def get_patch_at_latlon(self, lat, lon):
        self._ensure_open()
        easting, northing = self.transformer.transform(lon, lat)
        
        # Convert map coordinates to pixel offsets
        inv_transform = ~self.transform
        col_idx, row_idx = inv_transform * (easting, northing)
        col_idx, row_idx = int(col_idx), int(row_idx)
        
        p = self.patch_size // 2
        # Check boundary bounds
        if row_idx < p or row_idx >= self.height - p or col_idx < p or col_idx >= self.width - p:
            return None
            
        # Read the patch window (1-indexed band numbers in rasterio)
        window = ((row_idx - p, row_idx + p), (col_idx - p, col_idx + p))
        patch_raw = self.src.read(window=window)
        
        # Verify if the patch contains mostly valid data
        check_band = patch_raw[min(50, self.count - 1), :, :]
        nodata_ratio = (check_band < 0.0).mean()
        if nodata_ratio >= 0.5:
            return None
            
        # Resample to 200 bands
        patch = self.resample_to_foundation(patch_raw, self.wavelengths)
        return patch

    def close(self):
        if hasattr(self, 'src') and self.src is not None:
            try:
                self.src.close()
            except:
                pass
            self.src = None

    def __del__(self):
        self.close()

class MultiFlightEagleDataset(Dataset):
    def __init__(self, tif_paths, richness_csv, patch_size=16, augment=False, 
                 cache_path=None, mask_water_vapor=True):
        self.patch_size = patch_size
        self.augment = augment
        self.tif_paths = tif_paths
        self.richness_csv = richness_csv
        self.mappings = []
        self.dataset_cache = {}
        self.cache_lock = threading.Lock()
        self.mask_water_vapor = mask_water_vapor
        self.band_mean = None
        self.band_std = None

        if cache_path and os.path.exists(cache_path):
            import json
            print(f"[*] Loading EAGLE mapping from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                self.mappings = json.load(f)
            print(f"[OK] Loaded {len(self.mappings)} EAGLE sites from cache.")
            return

        self._initialize_mapping(richness_csv, tif_paths, patch_size, cache_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'cache_lock' in state:
            del state['cache_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache_lock = threading.Lock()

    def _initialize_mapping(self, richness_csv, tif_paths, patch_size, cache_path=None):
        self.richness_df = pd.read_csv(richness_csv)
        self.richness_df = self.richness_df.rename(columns={
            'Latitude': 'lat', 'Longitude': 'lon', 
            'richness': 'richness', 'Richness': 'richness'
        })
        
        # Load metadata to merge CLUSTER_ID
        metadata_csv = os.path.join(os.path.dirname(richness_csv), "biosoundscape_site_metadata.csv")
        if os.path.exists(metadata_csv):
            print(f"[*] Loading metadata cluster mappings from {metadata_csv}...")
            meta_df = pd.read_csv(metadata_csv)[['SiteID', 'CLUSTER_ID']]
            self.richness_df = self.richness_df.merge(meta_df, on='SiteID', how='left')
            self.richness_df['CLUSTER_ID'] = self.richness_df['CLUSTER_ID'].fillna(-1).astype(int)
        else:
            print(f"[!] Metadata file not found at {metadata_csv}. Defaulting all CLUSTER_ID to -1.")
            self.richness_df['CLUSTER_ID'] = -1
            
        print(f"[*] Mapping {len(self.richness_df)} potential richness sites across {len(tif_paths)} EAGLE tiles...")
        
        def process_one(tif_path):
            local_mappings = []
            try:
                ds = SingleEagleTiffDataset(tif_path, patch_size=patch_size, mask_water_vapor=self.mask_water_vapor)
                min_lon, min_lat, max_lon, max_lat = ds.get_bounds_wgs84()
                
                # Bounding box spatial filter
                mask = (self.richness_df['lat'] >= min_lat) & (self.richness_df['lat'] <= max_lat) & \
                       (self.richness_df['lon'] >= min_lon) & (self.richness_df['lon'] <= max_lon)
                relevant_sites = self.richness_df[mask]
                
                if len(relevant_sites) > 0:
                    for _, row in relevant_sites.iterrows():
                        patch = ds.get_patch_at_latlon(float(row['lat']), float(row['lon']))
                        if patch is not None:
                            local_mappings.append((
                                tif_path, 
                                float(row['lat']), 
                                float(row['lon']), 
                                float(row['richness']),
                                int(row.get('CLUSTER_ID', -1))
                            ))
                ds.close()
            except Exception as e:
                # print(f"Error mapping {tif_path}: {e}")
                pass
            return local_mappings

        all_results = []
        for tif_path in tqdm(tif_paths, desc="Mapping EAGLE Sites"):
            all_results.append(process_one(tif_path))

        seen_coords = set()
        for res_list in all_results:
            for item in res_list:
                coord_key = (round(item[1], 5), round(item[2], 5))
                if coord_key not in seen_coords:
                    self.mappings.append(item)
                    seen_coords.add(coord_key)

        if cache_path and len(tif_paths) > 2:
            import json
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(self.mappings, f)
            print(f"[OK] EAGLE mapping cached to {cache_path}")

        print(f"[OK] EAGLE Dataset Ready: {len(self.mappings)} unique sites.")

    def compute_band_stats(self, cache_path=None):
        """Compute global per-band mean/std across ALL patches. Results are cached to disk."""
        if cache_path and os.path.exists(cache_path):
            import json
            with open(cache_path, 'r') as f:
                stats = json.load(f)
            self.band_mean = torch.tensor(stats['mean']).reshape(200, 1, 1)
            self.band_std = torch.tensor(stats['std']).reshape(200, 1, 1)
            print(f"[*] Loaded EAGLE band stats from cache: {cache_path}")
            return
        
        print(f"[*] Computing global EAGLE band statistics across {len(self)} patches...")
        
        running_sum = torch.zeros(200)
        running_sq_sum = torch.zeros(200)
        n_pixels = 0
        
        def process_patch(i):
            tif_path, lat, lon, _, *rest = self.mappings[i]
            with self.cache_lock:
                if tif_path not in self.dataset_cache:
                    self.dataset_cache[tif_path] = SingleEagleTiffDataset(tif_path, patch_size=self.patch_size)
                ds = self.dataset_cache[tif_path]
            
            patch = ds.get_patch_at_latlon(lat, lon)
            if patch is None: return None
            return patch.sum(dim=(1, 2)), (patch**2).sum(dim=(1, 2)), patch.shape[1] * patch.shape[2]

        results = []
        for i in tqdm(range(len(self)), desc="EAGLE Band Stats"):
            results.append(process_patch(i))
            
        for res in results:
            if res:
                p_sum, p_sq_sum, p_pixels = res
                running_sum += p_sum
                running_sq_sum += p_sq_sum
                n_pixels += p_pixels
                
        if n_pixels == 0:
            print("[!] Could not compute band stats (no valid pixels found).")
            return
            
        global_mean = running_sum / n_pixels
        global_std = ((running_sq_sum / n_pixels) - global_mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
        
        self.band_mean = global_mean.reshape(200, 1, 1)
        self.band_std = global_std.reshape(200, 1, 1)
        
        if cache_path:
            import json
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({'mean': global_mean.tolist(), 'std': global_std.tolist()}, f)
            print(f"[OK] EAGLE band stats cached to {cache_path}")
            
    def set_band_stats(self, band_mean, band_std):
        self.band_mean = torch.tensor(band_mean).reshape(200, 1, 1) if not isinstance(band_mean, torch.Tensor) else band_mean.reshape(200, 1, 1)
        self.band_std = torch.tensor(band_std).reshape(200, 1, 1) if not isinstance(band_std, torch.Tensor) else band_std.reshape(200, 1, 1)

    def __len__(self):
        return len(self.mappings)

    def __getitem__(self, idx):
        tif_path, lat, lon, richness, *rest = self.mappings[idx]
        
        with self.cache_lock:
            if tif_path not in self.dataset_cache:
                self.dataset_cache[tif_path] = SingleEagleTiffDataset(
                    tif_path, patch_size=self.patch_size, mask_water_vapor=self.mask_water_vapor
                )
            ds = self.dataset_cache[tif_path]
        
        patch = ds.get_patch_at_latlon(lat, lon)
        if patch is None:
            patch = torch.zeros(200, self.patch_size, self.patch_size)
            
        if self.augment:
            patch = torch.rot90(patch, k=np.random.randint(0, 4), dims=(1, 2))
            if np.random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])
                
        return patch, torch.tensor([float(richness)])
