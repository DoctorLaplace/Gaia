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
    def __init__(self, nc_path, richness_csv, patch_size=16, augment=False, quiet=False, use_cache=True):
        self.nc_path = nc_path
        self.patch_size = patch_size
        self.augment = augment
        self.quiet = quiet
        self.richness_csv = richness_csv
        self.use_cache = use_cache
        
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
        if 'projection' in f and 'GeoTransform' in f['projection'].attrs:
            gt = f['projection'].attrs['GeoTransform']
            if isinstance(gt, bytes): gt = gt.decode()
            if isinstance(gt, str): gt = [float(x) for x in gt.split()]
            
            self.origin_x, self.pixel_w = gt[0], gt[1]
            self.origin_y, self.pixel_h = gt[3], gt[5]
        elif 'reflectance' in f and 'GeoTransform' in f['reflectance'].attrs:
            # Some NetCDFs store GT in the variable attrs
            gt = f['reflectance'].attrs['GeoTransform']
            if isinstance(gt, bytes): gt = gt.decode()
            if isinstance(gt, str): gt = [float(x) for x in gt.split()]
            self.origin_x, self.pixel_w = gt[0], gt[1]
            self.origin_y, self.pixel_h = gt[3], gt[5]
        else:
            self.origin_x, self.pixel_w = 0, 1
            self.origin_y, self.pixel_h = 0, 1

        # Access the hyperspectral cube metadata
        self.bands, self.height, self.width = cube.shape
        
        # CRS Transformer
        self.transformer = Transformer.from_crs("epsg:4326", "epsg:32734", always_xy=True)
        self.inverse_transformer = Transformer.from_crs("epsg:32734", "epsg:4326", always_xy=True)
        
        # If we weren't open before, close now to save resources
        if not was_open:
            self.close()

    def get_bounds(self):
        """Returns the geographic bounding box (min_lon, min_lat, max_lon, max_lat)."""
        # Transform corners from UTM back to WGS84
        rev_transformer = Transformer.from_crs("epsg:32734", "epsg:4326", always_xy=True)
        
        # corners in UTM
        c1_x, c1_y = self.origin_x, self.origin_y
        c2_x, c2_y = self.origin_x + self.width * self.pixel_w, self.origin_y + self.height * self.pixel_h
        
        lon1, lat1 = rev_transformer.transform(c1_x, c1_y)
        lon2, lat2 = rev_transformer.transform(c2_x, c2_y)
        
        return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)

    def resample_to_foundation(self, patch_raw, current_wavs):
        """Resample spectral bands to match the 200-band EnMAP foundation model.
        Masks wavelengths that fall in known atmospheric absorption gaps.
        Cleans L3 mosaic no-data fill values like -10000.
        """
        target_wavs = np.linspace(400, 2450, 200)
    
        BAD_RANGES = [(1340, 1480), (1780, 1970)]
    
        patch_raw = np.asarray(patch_raw, dtype=np.float32)
    
        # Critical fix for BioSCape L3 mosaics:
        # -10000 is no-data/fill, not reflectance.
        patch_raw = np.where(np.isfinite(patch_raw), patch_raw, 0.0)
        patch_raw = np.where(patch_raw <= -999.0, 0.0, patch_raw)
    
        # Reflectance should not be huge or negative.
        # Keep a little headroom above 1.0 in case scaled reflectance has bright pixels.
        patch_raw = np.clip(patch_raw, 0.0, 1.5)
    
        patch_hwc = np.transpose(patch_raw, (1, 2, 0))
    
        f = interp1d(
            current_wavs,
            patch_hwc,
            axis=-1,
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
        )
    
        resampled_hwc = f(target_wavs)
    
        # Clean again after interpolation.
        resampled_hwc = np.nan_to_num(resampled_hwc, nan=0.0, posinf=0.0, neginf=0.0)
        resampled_hwc = np.clip(resampled_hwc, 0.0, 1.5)
    
        for lo, hi in BAD_RANGES:
            mask = (target_wavs >= lo) & (target_wavs <= hi)
            resampled_hwc[:, :, mask] = 0.0
    
        patch_tensor = torch.from_numpy(np.transpose(resampled_hwc, (2, 0, 1))).float()
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
        if self.richness_df is None: return torch.zeros(200, self.patch_size, self.patch_size), torch.tensor([0.0])
        row = self.richness_df.iloc[idx]
        patch, bounds = self.get_patch_at_latlon(row['lat'], row['lon'])
        if patch is None: return torch.zeros(200, self.patch_size, self.patch_size), torch.tensor([0.0])
        
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


class BioScapeNetCDFDatasetV2(Dataset):
    """
    Dataset to handle NASA BioSCape AVIRIS-NG Level 2B (NetCDF) flightlines.
    Handles the spectral resampling from ~430 bands to the 200 foundation bands lazily (per-patch).
    """
    
    def resample_to_foundation(self, patch_raw, current_wavs):
        """Resample spectral bands to match the 200-band EnMAP foundation model."""
    
        patch_raw = np.asarray(patch_raw)
    
        # Fast path for L3 mosaic / already-resampled 200-band inputs.
        # Avoid expensive scipy interpolation on every 64x64 patch.
        if patch_raw.shape[0] == 200:
            return torch.from_numpy(patch_raw.copy()).float()
    
        target_wavs = np.linspace(400, 2450, 200)
    def __init__(self, nc_path, richness_csv, patch_size=16, augment=False, quiet=False, use_cache=True):
        self.nc_path = nc_path
        self.patch_size = patch_size
        self.augment = augment
        self.quiet = quiet
        self.richness_csv = richness_csv
        self.use_cache = use_cache

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
        self.h5_file = h5py.File(self.f_obj, 'r', rdcc_nbytes=64 * 1024 * 1024, rdcc_nslots=11213)

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

        # Use coordinate arrays directly
        self.easting = f['easting'][()]
        self.northing = f['northing'][()]

        self.origin_x = float(self.easting[0])
        self.origin_y = float(self.northing[0])
        self.pixel_w = float(self.easting[1] - self.easting[0])
        self.pixel_h = float(self.northing[1] - self.northing[0])
        # Ensure ascending order (important!)
        self.easting_ascending = self.easting[1] > self.easting[0]
        self.northing_ascending = self.northing[1] > self.northing[0]

        # Access the hyperspectral cube metadata
        self.bands, self.height, self.width = cube.shape

        from pyproj import Transformer

        from pyproj import CRS, Transformer

        tm_attrs = f['transverse_mercator'].attrs

        wkt = tm_attrs.get('crs_wkt') or tm_attrs.get('spatial_ref')

        if wkt is None:
            raise ValueError("No CRS WKT found in NetCDF")

        # FIX: convert numpy.bytes_ → str
        if isinstance(wkt, (bytes, np.bytes_)):
            wkt = wkt.decode("utf-8")

        crs = CRS.from_wkt(wkt)

        self.transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.inverse_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        # If we weren't open before, close now to save resources
        if not was_open:
            self.close()

    def get_bounds(self):
        # Corners from coordinate arrays
        x_min, x_max = self.easting[0], self.easting[-1]
        y_min, y_max = self.northing[-1], self.northing[0]

        lon1, lat1 = self.inverse_transformer.transform(x_min, y_min)
        lon2, lat2 = self.inverse_transformer.transform(x_max, y_max)

        return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)

    def resample_to_foundation(self, patch_raw, current_wavs):
        """Resample spectral bands to match the 200-band EnMAP foundation model.
        Masks wavelengths that fall in known atmospheric absorption gaps."""
        target_wavs = np.linspace(400, 2450, 200)

        # Known atmospheric absorption regions (BBL gaps) to mask
        BAD_RANGES = [(1340, 1480), (1780, 1970)]  # Water vapor bands

        patch_hwc = np.transpose(patch_raw, (1, 2, 0))
        f = interp1d(current_wavs, patch_hwc, axis=-1, kind='linear',
                     fill_value=0.0, bounds_error=False)
        resampled_hwc = f(target_wavs)

        # Zero out bands that fall inside atmospheric gaps
        for lo, hi in BAD_RANGES:
            mask = (target_wavs >= lo) & (target_wavs <= hi)
            resampled_hwc[:, :, mask] = 0.0

        patch_tensor = torch.from_numpy(np.transpose(resampled_hwc, (2, 0, 1))).float()
        return patch_tensor

    def get_patch_at_latlon(self, lat, lon):
        easting, northing = self.transformer.transform(lon, lat)
        return self.get_patch_at_coord(easting, northing)

    def get_patch_at_coord(self, easting, northing):
        self._ensure_open()
        # Find nearest indices from coordinate arrays
        x_idx = int(np.abs(self.easting - easting).argmin())
        y_idx = int(np.abs(self.northing - northing).argmin())
        p = self.patch_size // 2
        if y_idx < p or y_idx >= self.height - p or x_idx < p or x_idx >= self.width - p:
            return None, None

        # Selective extraction (flexible for grouped or flat NetCDF)
        if 'reflectance/reflectance' in self.h5_file:
            cube = self.h5_file['reflectance/reflectance']
        else:
            cube = self.h5_file['reflectance']

        patch_raw = cube[:, y_idx - p: y_idx + p, x_idx - p: x_idx + p]

        # Calculate real geographic bounds of this exact 16x16 pixel region
        # Origin is top-left
        e_min = self.easting[x_idx - p]
        e_max = self.easting[x_idx + p - 1]
        n_min = self.northing[y_idx + p - 1]
        n_max = self.northing[y_idx - p]

        ln_min, lt_max = self.inverse_transformer.transform(e_min, n_max)
        ln_max, lt_min = self.inverse_transformer.transform(e_max, n_min)
        bounds = [[lt_min, ln_min], [lt_max, ln_max]]

        patch = self.resample_to_foundation(patch_raw, self.wavelengths)
        return patch, bounds

    def __len__(self):
        return len(self.richness_df) if self.richness_df is not None else 0

    def __getitem__(self, idx):
        if self.richness_df is None: return torch.zeros(200, self.patch_size, self.patch_size), torch.tensor([0.0])
        row = self.richness_df.iloc[idx]
        patch, bounds = self.get_patch_at_latlon(row['lat'], row['lon'])
        if patch is None: return torch.zeros(200, self.patch_size, self.patch_size), torch.tensor([0.0])

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
            try:
                self.h5_file.close()
            except:
                pass
            self.h5_file = None
        if self.f_obj is not None:
            try:
                self.f_obj.close()
            except:
                pass
            self.f_obj = None

    def __del__(self):
        self.close()
BioScapeNetCDFDataset = BioScapeNetCDFDatasetV2

if __name__ == "__main__":
    print("BioSCape NetCDF Production Dataset Ready.")