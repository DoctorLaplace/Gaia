import xarray as xr
import numpy as np
import pandas as pd
import torch
import os
from scipy.interpolate import interp1d
from pyproj import Transformer
from torch.utils.data import Dataset, DataLoader
import h5py
try:
    from .s3_utils import get_s3_fs, get_cached_s3_fs, InstrumentedS3File, set_worker_status
except ImportError:
    # Fallback for direct script execution
    from s3_utils import get_s3_fs, get_cached_s3_fs, InstrumentedS3File, set_worker_status

class BioScapeNetCDFDataset(Dataset):
    """
    Dataset to handle NASA BioSCape AVIRIS-NG Level 2B (NetCDF) flightlines.
    Handles the spectral resampling from ~430 bands to the 200 foundation bands lazily (per-patch).
    """
    def __init__(self, nc_path, richness_csv, patch_size=16, augment=False, 
                 quiet=False, use_cache=True, mask_water_vapor=True):
        self.nc_path = nc_path
        self.patch_size = patch_size
        self.augment = augment
        self.quiet = quiet
        self.richness_csv = richness_csv
        self.use_cache = use_cache
        self.mask_water_vapor = mask_water_vapor
        
        # Lazy initialization for worker safety
        self.f_obj = None
        self.h5_file = None
        self._load_metadata()

        # Load richness CSV (Expect: Latitude, Longitude, richness)
        if richness_csv and os.path.exists(richness_csv):
            if not quiet: print(f"[*] Loading richness labels from: {richness_csv}")
            self.richness_df = pd.read_csv(richness_csv)
            col_map = {'Latitude': 'lat', 'Longitude': 'lon', 'richness': 'richness'}
            self.richness_df = self.richness_df.rename(columns=col_map)
        else:
            self.richness_df = None

    def _ensure_open(self):
        """Ensures the file is open. Call this before any data access."""
        if self.h5_file is not None:
            return
            
        import h5py
        from src.s3_utils import get_cached_s3_fs, get_s3_fs
        
        if self.nc_path.startswith("s3://"):
            fs = get_cached_s3_fs() if self.use_cache else get_s3_fs()
            raw_f = fs.open(self.nc_path, mode='rb', cache_type='readahead')
            self.f_obj = InstrumentedS3File(raw_f)
            # Update status for UI
            worker_info = torch.utils.data.get_worker_info()
            if worker_info:
                fname = os.path.basename(self.nc_path)
                set_worker_status(worker_info.id, fname)
        else:
            self.f_obj = open(self.nc_path, 'rb')
            
        # rdcc_nbytes: 64MB chunk cache (Extreme), rdcc_nslots: 11213 (prime)
        self.h5_file = h5py.File(self.f_obj, 'r', rdcc_nbytes=64*1024*1024, rdcc_nslots=11213)
        self.h5_file = h5py.File(self.f_obj, 'r', rdcc_nbytes=64*1024*1024, rdcc_nslots=11213)

    def _load_metadata(self):
        """Loads spatial and spectral metadata. Opens file briefly if needed."""
        was_open = self.h5_file is not None
        self._ensure_open()
        f = self.h5_file
        
        # Access wavelengths (flexible for grouped or flat NetCDF)
        if 'reflectance/wavelength' in f:
            self.wavelengths = f['reflectance/wavelength'][()]
            cube = f['reflectance/reflectance']
        else:
            self.wavelengths = f['wavelength'][()]
            cube = f['reflectance']
            
        # Extract spatial metadata from GeoTransform
        gt_str = None
        if 'projection' in f and 'GeoTransform' in f['projection'].attrs:
            gt_str = f['projection'].attrs['GeoTransform']
        elif 'reflectance' in f and 'GeoTransform' in f['reflectance'].attrs:
            gt_str = f['reflectance'].attrs['GeoTransform']
        elif 'GeoTransform' in f.attrs:
            gt_str = f.attrs['GeoTransform']

        if gt_str is not None:
            if isinstance(gt_str, bytes): gt_str = gt_str.decode()
            if isinstance(gt_str, str): gt = [float(x) for x in gt_str.split()]
            else: gt = list(gt_str)
            
            self.origin_x, self.pixel_w = gt[0], gt[1]
            self.origin_y, self.pixel_h = gt[3], gt[5]
        else:
            # Fallback for Level 3 Mosaics which use easting/northing arrays
            if 'easting' in f and 'northing' in f:
                self.origin_x = f['easting'][0]
                self.origin_y = f['northing'][0]
                self.pixel_w = f['easting'][1] - f['easting'][0]
                self.pixel_h = f['northing'][1] - f['northing'][0]
            else:
                self.origin_x, self.pixel_w = 0, 1
                self.origin_y, self.pixel_h = 0, 1

        
        # Access the hyperspectral cube metadata
        self.bands, self.height, self.width = cube.shape
        
        # CRS Transformer (Standard BioSCape is EPSG:32734)
        target_crs = "epsg:32734"
        if 'projection' in f and 'spatial_ref' in f['projection'].attrs:
            try:
                import pyproj
                wkt = f['projection'].attrs['spatial_ref']
                if isinstance(wkt, bytes): wkt = wkt.decode()
                target_crs = pyproj.CRS.from_wkt(wkt)
            except: pass

        from pyproj import Transformer
        self.transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
        self.inverse_transformer = Transformer.from_crs(target_crs, "epsg:4326", always_xy=True)
        
        
        # If we weren't open before, close now to save resources
        if not was_open:
            self.close()

    def get_bounds(self):
        """Returns the geographic bounding box (min_lon, min_lat, max_lon, max_lat)."""
        # Transform corners from file CRS back to WGS84
        c1_x, c1_y = self.origin_x, self.origin_y
        c2_x, c2_y = self.origin_x + self.width * self.pixel_w, self.origin_y + self.height * self.pixel_h
        
        lon1, lat1 = self.inverse_transformer.transform(c1_x, c1_y)
        lon2, lat2 = self.inverse_transformer.transform(c2_x, c2_y)
        
        return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)

    def mask_patch(self, patch, wavelengths):
        """Zero out water vapor bands (Atmospheric Cleaning)."""
        # Known atmospheric absorption regions (BBL gaps)
        BAD_RANGES = [(1340, 1480), (1780, 1970)]
        
        for lo, hi in BAD_RANGES:
            mask = (wavelengths >= lo) & (wavelengths <= hi)
            patch[mask, :, :] = 0.0
        return patch

    def resample_to_foundation(self, patch_raw, current_wavs):
        """Resample spectral bands to match the 200-band EnMAP foundation model."""
        target_wavs = np.linspace(400, 2450, 200)
        
        # Replace NoData fill values and negative boundary/noise values with 0.0 before interpolation
        patch_raw = np.where(patch_raw < 0.0, 0.0, patch_raw)
        
        patch_hwc = np.transpose(patch_raw, (1, 2, 0))
        f = interp1d(current_wavs, patch_hwc, axis=-1, kind='linear',
                     fill_value=0.0, bounds_error=False)
        resampled_hwc = f(target_wavs)
        patch_tensor = torch.from_numpy(np.transpose(resampled_hwc, (2, 0, 1))).float()
        
        if self.mask_water_vapor:
            patch_tensor = self.mask_patch(patch_tensor, target_wavs)
            
        return patch_tensor

    def get_patch_at_latlon(self, lat, lon):
        easting, northing = self.transformer.transform(lon, lat)
        return self.get_patch_at_coord(easting, northing)

    def get_patch_at_coord(self, easting, northing):
        self._ensure_open()
        x_idx = int((easting - self.origin_x) / self.pixel_w)
        y_idx = int((northing - self.origin_y) / self.pixel_h)
        p = self.patch_size // 2
        if y_idx < p or y_idx >= self.height - p or x_idx < p or x_idx >= self.width - p:
            return None, None
        
        # Selective extraction (flexible for grouped or flat NetCDF)
        if 'reflectance/reflectance' in self.h5_file:
            cube = self.h5_file['reflectance/reflectance']
        else:
            cube = self.h5_file['reflectance']
            
        patch_raw = cube[:, y_idx-p : y_idx+p, x_idx-p : x_idx+p]
        
        # Calculate real geographic bounds of this exact 16x16 pixel region
        # Origin is top-left
        e_min = self.origin_x + (x_idx - p) * self.pixel_w
        n_max = self.origin_y + (y_idx - p) * self.pixel_h # pixel_h is negative
        e_max = self.origin_x + (x_idx + p) * self.pixel_w
        n_min = self.origin_y + (y_idx + p) * self.pixel_h
        
        ln_min, lt_max = self.inverse_transformer.transform(e_min, n_max)
        ln_max, lt_min = self.inverse_transformer.transform(e_max, n_min)
        bounds = [[lt_min, ln_min], [lt_max, ln_max]]

        patch = self.resample_to_foundation(patch_raw, self.wavelengths)
        return patch, bounds

    def __len__(self):
        return len(self.richness_df) if self.richness_df is not None else 0

    def __getitem__(self, idx):
        if self.richness_df is None: return torch.zeros(200, 16, 16), torch.tensor([0.0])
        row = self.richness_df.iloc[idx]
        patch, bounds = self.get_patch_at_latlon(row['lat'], row['lon'])
        if patch is None: return torch.zeros(200, 16, 16), torch.tensor([0.0])
        
        # Spatial Augmentation (8-way: 4 rotations * 2 flips)
        if self.augment:
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            patch = torch.rot90(patch, k=k, dims=(1, 2))
            # Random horizontal flip
            if np.random.random() > 0.5:
                patch = torch.flip(patch, dims=[2])
                
        return patch, torch.tensor([float(row['richness'])])

    def close(self):
        if self.h5_file is not None:
            try: self.h5_file.close()
            except: pass
            self.h5_file = None
        if self.f_obj is not None:
            try: self.f_obj.close()
            except: pass
            self.f_obj = None

    def __del__(self):
        self.close()

if __name__ == "__main__":
    print("BioSCape NetCDF Production Dataset Ready.")
