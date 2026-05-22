"""
GAIA HEATMAP: Domain-Wide Richness Inference
=============================================

Tiles downloaded EnMAP L2A GeoTIFFs or BioSCape/AVIRIS NetCDF granules into
16x16 patches, runs the trained GaiaTransferModel on every valid patch, and
produces:

  1. GeoTIFF  - georeferenced raster for QGIS / ArcGIS
  2. PNG      - publication-quality static image
  3. HTML     - interactive Folium web map
  4. JSON     - cached raw patch predictions

Usage:
    python src/infer_heatmap.py
    python src/infer_heatmap.py --enmap_dir data/enmap/southern_cape
    python src/infer_heatmap.py --enmap_dir data/enmap/southern_cape --checkpoint checkpoints/gaia_bioscape_best.pth
    python src/infer_heatmap.py --from-json reports/heatmap_enmap/predictions.json

Notes:
    - EnMAP L2A GeoTIFFs may be stored as scaled integer reflectance. This script
      applies rasterio scale/offset metadata when present.
    - For EnMAP tiles, the script removes known invalid/water-vapor bands and
      resamples valid wavelengths to the 200-band model target grid.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vit_spatial_spectral import ViTSpatialSpectral


# ─── ANSI Colors ──────────────────────────────────────────────────────────────
class C:
    H = "\033[95m"
    B = "\033[94m"
    G = "\033[92m"
    W = "\033[93m"
    F = "\033[91m"
    E = "\033[0m"
    BOLD = "\033[1m"


# ══════════════════════════════════════════════════════════════════════════════
#  Model - architecture must match train_production.py
# ══════════════════════════════════════════════════════════════════════════════
class GaiaTransferModel(nn.Module):
    def __init__(self, num_targets: int = 1, patch_size: int = 16):
        super().__init__()
        self.encoder = ViTSpatialSpectral(
            image_size=patch_size,
            spatial_patch_size=1,
            spectral_patch_size=10,
            num_classes=num_targets,
            dim=96,
            depth=4,
            heads=8,
            mlp_dim=64,
            dropout=0.1,
            emb_dropout=0.1,
            channels=200,
            spectral_pos=torch.arange(20),
            spectral_pos_embed=True,
            blockwise_patch_embed=True,
            spectral_only=False,
            pixelwise=False,
        )
        self.encoder.mlp_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        # Keep compatibility with either patchwise output [B,H,W,C] / [B,H,W]
        # or a simpler [B,1] model head.
        if out.ndim >= 3:
            return out.mean(dim=tuple(range(1, out.ndim - 1))).squeeze(-1)
        return out.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Spectral resampling
# ══════════════════════════════════════════════════════════════════════════════
TARGET_WAVS = np.linspace(400, 2450, 200).astype(np.float32)
BAD_RANGES = [(1340, 1480), (1780, 1970)]

# 0-based EnMAP band indices to remove before interpolation.
ENMAP_INVALID_BANDS = [
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
    139, 140, 160, 161, 162, 163, 164, 165, 166,
]

ENMAP_WAVELENGTHS = np.array([
    418.24, 423.874, 429.294, 434.528, 439.603, 444.549, 449.391, 454.159,
    458.884, 463.584, 468.265, 472.934, 477.599, 482.265, 486.941, 491.633,
    496.349, 501.094, 505.87, 510.678, 515.519, 520.397, 525.313, 530.268,
    535.265, 540.305, 545.391, 550.525, 555.71, 560.947, 566.239, 571.587,
    576.995, 582.464, 587.997, 593.596, 599.267, 605.011, 610.833, 616.737,
    622.732, 628.797, 634.919, 641.1, 647.341, 653.643, 660.007, 666.435,
    672.927, 679.485, 686.11, 692.804, 699.567, 706.401, 713.307, 720.282,
    727.324, 734.431, 741.601, 748.833, 756.124, 763.472, 770.876, 778.333,
    785.843, 793.402, 801.01, 808.665, 816.367, 824.112, 831.901, 839.731,
    847.601, 855.509, 863.455, 871.433, 879.442, 887.478, 895.537, 902.257,
    903.617, 911.715, 911.872, 919.827, 921.624, 927.951, 931.512, 936.082,
    941.53, 944.217, 951.677, 952.355, 960.495, 961.948, 968.638, 972.341,
    976.783, 982.851, 984.932, 993.083, 993.475, 1004.21, 1015.05, 1026.0,
    1037.05, 1048.19, 1059.42, 1070.74, 1082.14, 1093.62, 1105.17, 1116.79,
    1128.47, 1140.2, 1151.98, 1163.81, 1175.67, 1187.56, 1199.48, 1211.42,
    1223.37, 1235.34, 1247.31, 1259.3, 1271.29, 1283.29, 1295.28, 1307.27,
    1319.25, 1331.22, 1343.18, 1355.13, 1367.06, 1378.96, 1390.84, 1461.46,
    1473.1, 1484.69, 1496.24, 1507.75, 1519.22, 1530.64, 1542.02, 1553.36,
    1564.65, 1575.9, 1587.1, 1598.26, 1609.36, 1620.43, 1631.44, 1642.41,
    1653.33, 1664.2, 1675.03, 1685.8, 1696.53, 1707.2, 1717.83, 1728.4,
    1738.93, 1749.4, 1759.83, 1939.44, 1948.98, 1958.49, 1967.95, 1977.37,
    1986.74, 1996.07, 2005.36, 2014.61, 2023.82, 2032.99, 2042.11, 2051.19,
    2060.24, 2069.24, 2078.21, 2087.13, 2096.01, 2104.86, 2113.67, 2122.44,
    2131.17, 2139.87, 2148.52, 2157.15, 2165.73, 2174.28, 2182.79, 2191.27,
    2199.71, 2208.12, 2216.5, 2224.84, 2233.14, 2241.42, 2249.66, 2257.86,
    2266.04, 2274.18, 2282.29, 2290.37, 2298.42, 2306.44, 2314.42, 2322.37,
    2330.29, 2338.19, 2346.05, 2353.88, 2361.68, 2369.45, 2377.19, 2384.9,
    2392.58, 2400.23, 2407.85, 2415.45, 2423.01, 2430.55, 2438.05, 2445.53,
], dtype=np.float32)

if len(ENMAP_WAVELENGTHS) != 224:
    raise RuntimeError(f"Expected 224 EnMAP wavelengths, got {len(ENMAP_WAVELENGTHS)}")

ENMAP_VALID_IDX = [i for i in range(224) if i not in set(ENMAP_INVALID_BANDS)]
ENMAP_VALID_WAVS = ENMAP_WAVELENGTHS[ENMAP_VALID_IDX]


def resample_patch(patch_raw: np.ndarray, current_wavs: np.ndarray) -> torch.Tensor:
    """Convert patch from (bands, H, W) to torch tensor (200, H, W)."""
    patch_raw = np.asarray(patch_raw, dtype=np.float32)
    current_wavs = np.asarray(current_wavs, dtype=np.float32)

    if patch_raw.ndim != 3:
        raise ValueError(f"Expected patch shape (bands,H,W), got {patch_raw.shape}")
    if patch_raw.shape[0] != len(current_wavs):
        raise ValueError(
            f"Band count/wavelength mismatch: patch has {patch_raw.shape[0]} bands, "
            f"wavelengths has {len(current_wavs)}"
        )

    patch_hwc = np.transpose(patch_raw, (1, 2, 0))
    interp = interp1d(
        current_wavs,
        patch_hwc,
        axis=-1,
        kind="linear",
        fill_value=0.0,
        bounds_error=False,
        assume_sorted=False,
    )
    resampled = interp(TARGET_WAVS).astype(np.float32)

    for lo, hi in BAD_RANGES:
        mask = (TARGET_WAVS >= lo) & (TARGET_WAVS <= hi)
        resampled[:, :, mask] = 0.0

    return torch.from_numpy(np.transpose(resampled, (2, 0, 1))).float()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def finite_patch(raw: np.ndarray, nodata: float | int | None = None, min_valid_fraction: float = 0.90) -> bool:
    """Reject nodata/NaN/empty patches."""
    arr = np.asarray(raw)
    finite = np.isfinite(arr)

    if nodata is not None:
        finite &= arr != nodata

    valid_fraction = float(finite.mean()) if finite.size else 0.0
    if valid_fraction < min_valid_fraction:
        return False

    valid_values = arr[finite]
    if valid_values.size == 0:
        return False

    if np.allclose(valid_values, 0.0):
        return False

    # Very negative sentinel values are common in remote sensing rasters.
    if np.nanmin(valid_values) < -9000:
        return False

    return True


def sanitize_patch(raw: np.ndarray, nodata: float | int | None = None) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    if nodata is not None:
        raw = np.where(raw == nodata, 0.0, raw)
    return raw



_LAND_GLOBE = None


def get_land_globe():
    """Lazy-load global land/ocean mask."""
    global _LAND_GLOBE
    if _LAND_GLOBE is None:
        try:
            from global_land_mask import globe
            _LAND_GLOBE = globe
        except ImportError as exc:
            raise RuntimeError(
                "Missing global-land-mask. Install with: pip install --user global-land-mask"
            ) from exc
    return _LAND_GLOBE


def is_land_latlon(lat: float, lon: float) -> bool:
    """Return True if patch center is on land."""
    if not np.isfinite(lat) or not np.isfinite(lon):
        return False
    globe = get_land_globe()
    return bool(globe.is_land(float(lat), float(lon)))


def filter_land_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ocean predictions from existing JSON results."""
    kept = []
    removed = 0

    for item in results:
        if is_land_latlon(float(item["lat"]), float(item["lon"])):
            kept.append(item)
        else:
            removed += 1

    print(f"{C.G}[OK] Land mask kept {len(kept)} patches; removed {removed} ocean/water patches.{C.E}")
    return kept
    





def window_latlon_bounds(src: Any, window: Any) -> tuple[list[list[float]], float, float]:
    """Return [[lat_min, lon_min], [lat_max, lon_max]], center_lat, center_lon."""
    import rasterio
    from rasterio.windows import bounds as window_bounds

    left, bottom, right, top = window_bounds(window, src.transform)

    crs = src.crs
    if crs is None:
        # Southern Cape is in the southern hemisphere; EPSG:32734 is UTM 34S.
        print(f"{C.W}[WARN] Source CRS missing. Assuming EPSG:32734 for coordinate output.{C.E}")
        crs = rasterio.crs.CRS.from_epsg(32734)

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Transform all four corners in case the raster has rotation/skew.
    corners_xy = [(left, bottom), (left, top), (right, bottom), (right, top)]
    corners_lonlat = [transformer.transform(x, y) for x, y in corners_xy]
    lons = [pt[0] for pt in corners_lonlat]
    lats = [pt[1] for pt in corners_lonlat]

    center_x = (left + right) / 2.0
    center_y = (bottom + top) / 2.0
    center_lon, center_lat = transformer.transform(center_x, center_y)

    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    return bounds, float(center_lat), float(center_lon)


# ══════════════════════════════════════════════════════════════════════════════
#  Tile NetCDF granule into patches
# ══════════════════════════════════════════════════════════════════════════════
def tile_granule(nc_path: str | Path, patch_size: int = 16) -> Iterable[dict[str, Any]]:
    """
    Yield dicts: {patch, row, col, lat, lon, bounds, source}
    for every valid 16x16 NetCDF tile.
    """
    nc_path = Path(nc_path)

    with h5py.File(nc_path, "r") as f:
        if "reflectance/wavelength" in f:
            wavelengths = f["reflectance/wavelength"][()]
            cube_ds = f["reflectance/reflectance"]
        else:
            wavelengths = f["wavelength"][()]
            cube_ds = f["reflectance"]

        bands, height, width = cube_ds.shape
        has_coord_arrays = "easting" in f and "northing" in f

        easting = f["easting"][()] if has_coord_arrays else None
        northing = f["northing"][()] if has_coord_arrays else None

        if "transverse_mercator" in f:
            tm_attrs = f["transverse_mercator"].attrs
            wkt = tm_attrs.get("crs_wkt") or tm_attrs.get("spatial_ref")
            if isinstance(wkt, (bytes, np.bytes_)):
                wkt = wkt.decode("utf-8")
            crs = CRS.from_wkt(wkt)
        else:
            crs = CRS.from_epsg(32734)

        origin_x, pixel_w = 0.0, 1.0
        origin_y, pixel_h = 0.0, -1.0

        if not has_coord_arrays:
            gt = None
            if "projection" in f and "GeoTransform" in f["projection"].attrs:
                gt = f["projection"].attrs["GeoTransform"]
            elif "reflectance" in f and isinstance(f["reflectance"], h5py.Group) and "GeoTransform" in f["reflectance"].attrs:
                gt = f["reflectance"].attrs["GeoTransform"]

            if gt is not None:
                if isinstance(gt, bytes):
                    gt = gt.decode()
                if isinstance(gt, str):
                    gt = [float(x) for x in gt.split()]
                origin_x, pixel_w = float(gt[0]), float(gt[1])
                origin_y, pixel_h = float(gt[3]), float(gt[5])

        inv_tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        p = patch_size
        n_rows = height // p
        n_cols = width // p

        for r in range(n_rows):
            for c in range(n_cols):
                y0, y1 = r * p, (r + 1) * p
                x0, x1 = c * p, (c + 1) * p

                sample = cube_ds[0, y0:y1, x0:x1]
                if not finite_patch(sample):
                    continue

                raw = cube_ds[:, y0:y1, x0:x1]
                if raw.shape[1] != p or raw.shape[2] != p or not finite_patch(raw):
                    continue

                raw = sanitize_patch(raw)
                patch_tensor = resample_patch(raw, wavelengths)

                if has_coord_arrays:
                    if easting is None or northing is None:
                        continue
                    if x1 > len(easting) or y1 > len(northing):
                        continue
                    e_center = float(easting[x0:x1].mean())
                    n_center = float(northing[y0:y1].mean())
                    e_min = float(easting[x0])
                    e_max = float(easting[x1 - 1])
                    n_min = float(northing[y1 - 1])
                    n_max = float(northing[y0])
                else:
                    e_min = origin_x + x0 * pixel_w
                    e_max = origin_x + x1 * pixel_w
                    n_max = origin_y + y0 * pixel_h
                    n_min = origin_y + y1 * pixel_h
                    e_center = (e_min + e_max) / 2.0
                    n_center = (n_min + n_max) / 2.0

                lon_min, lat_min = inv_tf.transform(e_min, n_min)
                lon_max, lat_max = inv_tf.transform(e_max, n_max)
                lon_c, lat_c = inv_tf.transform(e_center, n_center)

                yield {
                    "patch": patch_tensor,
                    "row": r,
                    "col": c,
                    "lat": float(lat_c),
                    "lon": float(lon_c),
                    "bounds": [[min(lat_min, lat_max), min(lon_min, lon_max)], [max(lat_min, lat_max), max(lon_min, lon_max)]],
                    "source": str(nc_path.name),
                }


# ══════════════════════════════════════════════════════════════════════════════
#  Tile EnMAP GeoTIFF into patches
# ══════════════════════════════════════════════════════════════════════════════
def get_raster_scales_offsets(src: Any, band_nums: list[int]) -> tuple[np.ndarray, np.ndarray]:
    scales = np.array([src.scales[i - 1] if src.scales else 1.0 for i in band_nums], dtype=np.float32)
    offsets = np.array([src.offsets[i - 1] if src.offsets else 0.0 for i in band_nums], dtype=np.float32)

    # Rasterio sometimes returns None-like values.
    scales = np.where(np.isfinite(scales) & (scales != 0), scales, 1.0).astype(np.float32)
    offsets = np.where(np.isfinite(offsets), offsets, 0.0).astype(np.float32)
    return scales[:, None, None], offsets[:, None, None]


def tile_enmap_geotiff(tif_path: str | Path, patch_size: int = 16) -> Iterable[dict[str, Any]]:
    """Yield non-overlapping 16x16 EnMAP GeoTIFF patches with WGS84 bounds."""
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        print(f"{C.F}[!] rasterio required for EnMAP GeoTIFFs. Install with: pip install rasterio{C.E}")
        return

    tif_path = Path(tif_path)

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        height = src.height
        width = src.width
        nodata = src.nodata

        if n_bands == 224:
            band_nums = [i + 1 for i in ENMAP_VALID_IDX]  # rasterio is 1-indexed
            current_wavs = ENMAP_VALID_WAVS
        elif n_bands == len(ENMAP_VALID_IDX):
            band_nums = list(range(1, n_bands + 1))
            current_wavs = ENMAP_VALID_WAVS
        elif n_bands == 200:
            band_nums = list(range(1, n_bands + 1))
            current_wavs = TARGET_WAVS
        else:
            print(
                f"{C.W}[WARN] {tif_path.name}: unexpected band count {n_bands}. "
                f"Assuming linear wavelengths across 400-2450 nm.{C.E}"
            )
            band_nums = list(range(1, n_bands + 1))
            current_wavs = np.linspace(400, 2450, n_bands).astype(np.float32)

        scales, offsets = get_raster_scales_offsets(src, band_nums)

        p = patch_size
        n_rows = height // p
        n_cols = width // p

        for r in range(n_rows):
            for c in range(n_cols):
                y0 = r * p
                x0 = c * p
                window = Window(x0, y0, p, p)

                sample = src.read(1, window=window, masked=True)
                if sample.mask.all():
                    continue
                sample_filled = sample.filled(nodata if nodata is not None else np.nan)
                if not finite_patch(sample_filled, nodata=nodata):
                    continue

                raw = src.read(band_nums, window=window, masked=True).astype(np.float32)
                raw = raw.filled(nodata if nodata is not None else np.nan)
                raw = raw * scales + offsets

                if not finite_patch(raw, nodata=nodata):
                    continue

                raw = sanitize_patch(raw, nodata=nodata)

                # Keep extreme sensor/sentinel values from destabilizing interpolation.
                raw = np.clip(raw, -1.0e4, 1.0e4)

                try:
                    patch_tensor = resample_patch(raw, current_wavs)
                except ValueError as exc:
                    print(f"{C.W}[SKIP] {tif_path.name} row={r} col={c}: {exc}{C.E}")
                    continue

                bounds, lat_c, lon_c = window_latlon_bounds(src, window)

                yield {
                    "patch": patch_tensor,
                    "row": r,
                    "col": c,
                    "lat": lat_c,
                    "lon": lon_c,
                    "bounds": bounds,
                    "source": tif_path.name,
                }


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════
def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[dict[str, Any], float, float, torch.Tensor | None, torch.Tensor | None]:
    checkpoint_path = Path(checkpoint_path)
    print(f"{C.B}[*] Loading checkpoint: {checkpoint_path}{C.E}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    richness_mean = float(ckpt.get("richness_mean", 0.0))
    richness_std = float(ckpt.get("richness_std", 1.0))

    band_mean = None
    band_std = None
    if "band_mean" in ckpt and "band_std" in ckpt:
        bm = ckpt["band_mean"]
        bs = ckpt["band_std"]
        if not torch.is_tensor(bm):
            bm = torch.tensor(bm, dtype=torch.float32)
        if not torch.is_tensor(bs):
            bs = torch.tensor(bs, dtype=torch.float32)
        band_mean = bm.float().view(-1, 1, 1)
        band_std = bs.float().view(-1, 1, 1)
        band_std = torch.where(band_std == 0, torch.ones_like(band_std), band_std)
        print(f"{C.G}[OK] Band normalization loaded ({bm.numel()} bands){C.E}")

    return ckpt, richness_mean, richness_std, band_mean, band_std


def discover_files(data_dir: str | Path, file_type: str) -> list[tuple[Path, str]]:
    data_dir = Path(data_dir)
    nc_files = sorted(data_dir.glob("*.nc"))

    # Only process combined EnMAP L2A image GeoTIFFs.
    # Ignore L0, VNIR/SWIR split files, quality masks, SWIM/SAR, etc.
    tif_files = sorted(data_dir.glob("ENMAP*L2A*_image.tif"))
    tif_files += sorted(data_dir.glob("ENMAP*L2A*_image.tiff"))
    tif_files += sorted(data_dir.glob("**/ENMAP*L2A*_image.tif"))
    tif_files += sorted(data_dir.glob("**/ENMAP*L2A*_image.tiff"))
    tif_files = sorted(set(tif_files))

    if file_type == "enmap":
        nc_files = []
    elif file_type == "nc":
        tif_files = []

    return [(f, "nc") for f in nc_files] + [(f, "enmap") for f in tif_files]


def infer_all(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    patch_size: int = 16,
    batch_size: int = 64,
    device: torch.device | None = None,
    file_type: str = "auto",
    max_patches: int | None = None,
    land_only: bool = False,
    existing_results: list[dict[str, Any]] | None = None,
    resume_save_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run model on every valid patch across all granules/tiles in data_dir."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt, richness_mean, richness_std, band_mean, band_std = load_checkpoint(checkpoint_path, device)

    model = GaiaTransferModel(num_targets=1, patch_size=patch_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"{C.G}[OK] Model loaded on {device}. Richness mean={richness_mean:.2f}, std={richness_std:.2f}{C.E}")

    all_files = discover_files(data_dir, file_type)
    if not all_files:
        print(f"{C.F}[!] No matching .nc/.tif files found in {data_dir}{C.E}")
        return []

    n_nc = sum(1 for _, kind in all_files if kind == "nc")
    n_tif = sum(1 for _, kind in all_files if kind == "enmap")
    print(f"{C.B}[*] Found {n_nc} NetCDF + {n_tif} GeoTIFF file(s) in {data_dir}{C.E}")

    results: list[dict[str, Any]] = list(existing_results or [])
    global_patch_count = 0
    
    completed_sources = {
        str(r.get("source"))
        for r in results
        if r.get("source")
    }
    
    if completed_sources:
        print(f"{C.G}[OK] Resume mode: {len(completed_sources)} source file(s) already completed.{C.E}")

    for fpath, kind in all_files:
        if fpath.name in completed_sources:
            print(f"{C.W}[SKIP] Already processed: {fpath.name}{C.E}")
            continue
        print(f"\n{C.BOLD}-- Processing ({kind.upper()}): {fpath.name}{C.E}")
        patches_buf: list[torch.Tensor] = []
        meta_buf: list[dict[str, Any]] = []
        file_patch_count = 0

        tile_fn = tile_granule if kind == "nc" else tile_enmap_geotiff

        try:
            iterator = tile_fn(fpath, patch_size)
            for tile in tqdm(iterator, desc="Patches", unit="patch"):
                if land_only and not is_land_latlon(tile["lat"], tile["lon"]):
                    continue
            
                patches_buf.append(tile["patch"])
                meta_buf.append(tile)
                file_patch_count += 1
                global_patch_count += 1

                if len(patches_buf) >= batch_size:
                    run_batch(
                        model,
                        patches_buf,
                        meta_buf,
                        results,
                        richness_mean,
                        richness_std,
                        device,
                        band_mean,
                        band_std,
                    )
                    patches_buf, meta_buf = [], []

                if max_patches is not None and global_patch_count >= max_patches:
                    break

            if patches_buf:
                run_batch(
                    model,
                    patches_buf,
                    meta_buf,
                    results,
                    richness_mean,
                    richness_std,
                    device,
                    band_mean,
                    band_std,
                )

        except Exception as exc:
            print(f"{C.F}[FAIL] {fpath.name}: {exc}{C.E}")
            continue

        print(f"   {C.G}-> {file_patch_count} patches extracted, {len(results)} total predictions{C.E}")
        if resume_save_path is not None:
            save_predictions_json(results, resume_save_path)
            print(f"{C.G}[OK] Resume checkpoint saved after {fpath.name}{C.E}")
        if max_patches is not None and global_patch_count >= max_patches:
            print(f"{C.W}[INFO] Stopped early at --max-patches={max_patches}{C.E}")
            break

    return results


def run_batch(
    model: nn.Module,
    patches: list[torch.Tensor],
    metas: list[dict[str, Any]],
    results: list[dict[str, Any]],
    richness_mean: float,
    richness_std: float,
    device: torch.device,
    band_mean: torch.Tensor | None = None,
    band_std: torch.Tensor | None = None,
) -> None:
    batch = torch.stack(patches).float()

    if band_mean is not None and band_std is not None:
        if band_mean.shape[0] != batch.shape[1]:
            raise ValueError(
                f"Checkpoint band stats have {band_mean.shape[0]} bands, "
                f"but input batch has {batch.shape[1]} bands."
            )
        batch = (batch - band_mean) / band_std

    batch = batch.to(device, non_blocking=True)

    with torch.no_grad():
        preds_norm = model(batch).detach().cpu().numpy().reshape(-1)

    preds_real = preds_norm * richness_std + richness_mean

    for pred, meta in zip(preds_real, metas):
        results.append({
            "lat": float(meta["lat"]),
            "lon": float(meta["lon"]),
            "richness": float(pred),
            "bounds": meta["bounds"],
            "source": meta.get("source"),
            "row": int(meta.get("row", -1)),
            "col": int(meta.get("col", -1)),
        })


# ══════════════════════════════════════════════════════════════════════════════
#  Outputs
# ══════════════════════════════════════════════════════════════════════════════
def results_extent(results: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    lats = [float(r["lat"]) for r in results]
    lons = [float(r["lon"]) for r in results]
    return min(lons), min(lats), max(lons), max(lats)

def rasterize_results(results: list[dict[str, Any]], resolution: float) -> tuple[np.ndarray, Any, tuple[float, float, float, float]]:
    try:
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise RuntimeError("rasterio is required for raster outputs") from exc

    # Result lat/lon values are patch CENTERS, not raster edges.
    # Expand bounds by half a cell so the PNG/GeoTIFF aligns correctly.
    lon_min_c, lat_min_c, lon_max_c, lat_max_c = results_extent(results)

    half = resolution / 2.0
    lon_min = lon_min_c - half
    lon_max = lon_max_c + half
    lat_min = lat_min_c - half
    lat_max = lat_max_c + half

    if lon_min == lon_max:
        lon_max += resolution
    if lat_min == lat_max:
        lat_max += resolution

    n_cols = max(1, int(np.ceil((lon_max - lon_min) / resolution)))
    n_rows = max(1, int(np.ceil((lat_max - lat_min) / resolution)))

    grid_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
    counts = np.zeros((n_rows, n_cols), dtype=np.float32)

    for item in results:
        lat = float(item["lat"])
        lon = float(item["lon"])
        val = float(item["richness"])

        if not np.isfinite(val):
            continue

        col = int((lon - lon_min) / resolution)
        row = int((lat_max - lat) / resolution)

        row = min(max(row, 0), n_rows - 1)
        col = min(max(col, 0), n_cols - 1)

        grid_sum[row, col] += val
        counts[row, col] += 1

    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    valid = counts > 0
    grid[valid] = (grid_sum[valid] / counts[valid]).astype(np.float32)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, n_cols, n_rows)
    return grid, transform, (lon_min, lat_min, lon_max, lat_max)

def save_predictions_json(results: list[dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    tmp_path.write_text(
        json.dumps(results, separators=(",", ":")),
        encoding="utf-8",
    )

    tmp_path.replace(out_path)

    print(f"{C.G}[OK] Predictions JSON saved: {out_path} ({len(results)} points){C.E}")


def save_geotiff(results: list[dict[str, Any]], out_path: str | Path, resolution: float = 0.005) -> None:
    try:
        import rasterio
    except ImportError:
        print(f"{C.W}[!] rasterio not installed - skipping GeoTIFF. Install with: pip install rasterio{C.E}")
        return

    grid, transform, _ = rasterize_results(results, resolution)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(grid, 1)
        dst.update_tags(
            description="GAIA predicted species richness from EnMAP/BioSCape patches",
            resolution_degrees=str(resolution),
        )

    print(f"{C.G}[OK] GeoTIFF saved: {out_path} ({grid.shape[1]} x {grid.shape[0]}){C.E}")


def save_png(results: list[dict[str, Any]], out_path: str | Path, resolution: float = 0.005) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{C.W}[!] matplotlib not installed - skipping PNG.{C.E}")
        return

    grid, _, extent = rasterize_results(results, resolution)
    lon_min, lat_min, lon_max, lat_max = extent
    valid = np.isfinite(grid)
    if not valid.any():
        print(f"{C.W}[!] No finite values for PNG output.{C.E}")
        return

    vmin = float(np.nanpercentile(grid[valid], 2))
    vmax = float(np.nanpercentile(grid[valid], 98))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(
        grid,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Predicted Species Richness", fontsize=12)

    ax.set_title("GAIA - Species Richness Heatmap", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{C.G}[OK] PNG saved: {out_path}{C.E}")
    

def add_site_markers_to_map(
    m,
    sites_csv: str | Path | None,
    sites_season: str = "all",
    richness_cmap=None,
    richness_vmin: float | None = None,
    richness_vmax: float | None = None,
) -> None:
    """
    Add observed BioSoundscape/BioSCape field richness sites.

    Behavior:
      - filters sites by dry/wet/all
      - colors site circles by observed richness using the same colormap as heatmap
      - shows observed richness in popup/tooltip only
      - forces markers above PNG heatmap overlay
    """
    if sites_csv is None:
        print(f"{C.W}[INFO] No --sites-csv provided; skipping observed site markers.{C.E}")
        return

    try:
        import folium
        import pandas as pd
    except ImportError:
        print(f"{C.W}[!] pandas/folium missing - skipping site markers.{C.E}")
        return

    sites_csv = Path(sites_csv)
    if not sites_csv.exists():
        print(f"{C.W}[!] Sites CSV not found: {sites_csv}{C.E}")
        return

    df = pd.read_csv(sites_csv)
    print(f"{C.B}[*] Loaded sites CSV: {sites_csv} ({len(df)} rows){C.E}")
    print(f"{C.B}[*] Site CSV columns: {list(df.columns)}{C.E}")

    # Flexible column matching so capitalization does not break the map.
    colmap = {c.lower().strip(): c for c in df.columns}

    def find_col(*names):
        for name in names:
            key = name.lower().strip()
            if key in colmap:
                return colmap[key]
        return None

    lat_col = find_col("Latitude", "lat", "decimalLatitude")
    lon_col = find_col("Longitude", "lon", "long", "decimalLongitude")
    site_col = find_col("SiteID", "site_id", "site", "id")
    richness_col = find_col("richness", "observed_richness", "species_richness")
    campaign_col = find_col("Campaign", "campaign", "season")
    month_col = find_col("Month", "month")
    year_col = find_col("Year", "year")
    day_col = find_col("Day", "day")
    recnum_col = find_col("recnum", "record", "record_id")

    missing = []
    if lat_col is None:
        missing.append("Latitude")
    if lon_col is None:
        missing.append("Longitude")
    if site_col is None:
        missing.append("SiteID")
    if richness_col is None:
        missing.append("richness")

    if missing:
        print(f"{C.W}[!] Sites CSV missing required columns: {missing}{C.E}")
        print(f"{C.W}    Found columns: {list(df.columns)}{C.E}")
        return

    df = df.dropna(subset=[lat_col, lon_col, richness_col]).copy()

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[richness_col] = pd.to_numeric(df[richness_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col, richness_col]).copy()

    print(f"{C.B}[*] Sites after numeric/dropna cleanup: {len(df)}{C.E}")

    # Keep only Southern Cape-ish coordinates, so random bad rows do not appear.
    before_bbox = len(df)
    df = df[
        (df[lon_col] >= 18.0)
        & (df[lon_col] <= 25.0)
        & (df[lat_col] >= -35.5)
        & (df[lat_col] <= -31.5)
    ].copy()
    print(f"{C.B}[*] Sites after Southern Cape bbox filter: {len(df)} / {before_bbox}{C.E}")

    # Season filter.
    before_season = len(df)

    if sites_season != "all":
        if month_col is not None:
            months = pd.to_numeric(df[month_col], errors="coerce")

            if sites_season == "dry":
                # Dry/BioSCape match: Oct-Nov, plus dry-ish Dec-Mar if present.
                df = df[months.isin([10, 11, 12, 1, 2, 3])].copy()
            elif sites_season == "wet":
                df = df[months.isin([5, 6, 7, 8, 9])].copy()

            print(
                f"{C.B}[*] Sites after {sites_season} Month filter: "
                f"{len(df)} / {before_season}{C.E}"
            )

        elif campaign_col is not None:
            campaign = df[campaign_col].astype(str).str.lower().str.strip()

            if sites_season == "dry":
                df = df[campaign.str.contains("dry", na=False)].copy()
            elif sites_season == "wet":
                df = df[campaign.str.contains("wet", na=False)].copy()

            print(
                f"{C.B}[*] Sites after {sites_season} Campaign filter: "
                f"{len(df)} / {before_season}{C.E}"
            )

        else:
            print(
                f"{C.W}[WARN] --sites-season {sites_season} requested, "
                f"but CSV has no Month or Campaign column. Showing all sites.{C.E}"
            )

    if len(df) == 0:
        print(f"{C.W}[!] No site markers left after filtering. Try --sites-season all.{C.E}")
        return

    # Force site circles above the image overlay.
    try:
        folium.map.CustomPane("siteMarkers", z_index=650).add_to(m)
        marker_pane = "siteMarkers"
    except Exception:
        marker_pane = None

    site_layer = folium.FeatureGroup(
        name=f"Observed field richness sites ({sites_season})",
        show=True,
    )

    added = 0

    for _, row in df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        observed = float(row[richness_col])

        if not np.isfinite(lat) or not np.isfinite(lon) or not np.isfinite(observed):
            continue

        site_id = str(row.get(site_col, "unknown"))
        recnum = row.get(recnum_col, "") if recnum_col else ""
        campaign = row.get(campaign_col, "") if campaign_col else ""
        year = row.get(year_col, "") if year_col else ""
        month = row.get(month_col, "") if month_col else ""
        day = row.get(day_col, "") if day_col else ""

        # Use same color scale as heatmap.
        if richness_cmap is not None and richness_vmin is not None and richness_vmax is not None:
            clipped = max(richness_vmin, min(richness_vmax, observed))
            fill_color = richness_cmap(clipped)
        else:
            fill_color = "#FF8C00"

        popup = (
            f"<b>Site:</b> {site_id}<br>"
            f"<b>Observed richness:</b> {observed:.2f}<br>"
            f"<b>Campaign:</b> {campaign}<br>"
            f"<b>Date:</b> {year}-{month}-{day}<br>"
            f"<b>Record:</b> {recnum}<br>"
            f"<b>Lat/Lon:</b> {lat:.5f}, {lon:.5f}"
        )

        marker_kwargs = dict(
            location=[lat, lon],
            radius=7,
            color="black",
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=1.0,
            popup=popup,
            tooltip=f"{site_id} | observed richness: {observed:.1f}",
        )

        if marker_pane is not None:
            marker_kwargs["pane"] = marker_pane

        folium.CircleMarker(**marker_kwargs).add_to(site_layer)
        added += 1

    site_layer.add_to(m)

    print(
        f"{C.G}[OK] Added {added} observed richness site markers "
        f"from {sites_csv} using sites-season={sites_season}{C.E}"
    )

def save_html(
    results: list[dict[str, Any]],
    out_path: str | Path,
    resolution: float = 0.005,
    sites_csv: str | Path | None = None,
    sites_season: str = "all",
    ) -> None:
    """
    Fast interactive map.

    Instead of drawing tens of thousands of Folium rectangles, this creates one
    transparent PNG overlay and places it on top of a basemap. This is much
    faster in the browser.
    """
    try:
        import branca.colormap as bcm
        import folium
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            f"{C.W}[!] Missing packages - skipping HTML. "
            f"Install with: pip install --user folium branca matplotlib{C.E}"
        )
        return

    grid, _, extent = rasterize_results(results, resolution)
    lon_min, lat_min, lon_max, lat_max = extent

    valid = np.isfinite(grid)
    if not valid.any():
        print(f"{C.W}[!] No finite values for HTML output.{C.E}")
        return

    vmin = float(np.nanpercentile(grid[valid], 2))
    vmax = float(np.nanpercentile(grid[valid], 98))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    # Build transparent heatmap PNG.
    cmap_mpl = plt.get_cmap("RdYlBu_r")
    norm = np.clip((grid - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap_mpl(norm)

    # Transparent where no prediction exists.
    rgba[..., 3] = np.where(valid, 0.72, 0.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path = out_path.parent / "richness_overlay.png"
    plt.imsave(overlay_path, rgba)

    center_lat = float(np.nanmean([r["lat"] for r in results]))
    center_lon = float(np.nanmean([r["lon"] for r in results]))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None,
        prefer_canvas=True,
    )
    
    # Clean light basemap
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Light basemap",
        control=True,
        show=True,
    ).add_to(m)
    
    # Standard street/rivers layer
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap roads/rivers",
        control=True,
        show=False,
    ).add_to(m)
    
    # Terrain / mountains / rivers
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data © OpenStreetMap contributors, SRTM | Map style © OpenTopoMap",
        name="Terrain: mountains/rivers",
        max_zoom=17,
        control=True,
        show=False,
    ).add_to(m)
    
    # Satellite imagery background
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Satellite imagery",
        control=True,
        show=False,
    ).add_to(m)
    
    # Topographic map background
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Topographic map",
        control=True,
        show=False,
    ).add_to(m)

    # Small visual alignment correction.
    # Positive latitude shift moves the heatmap north.
    overlay_lat_shift_pixels = 1.0
    overlay_lon_shift_pixels = 0.0
    
    overlay_lat_shift = overlay_lat_shift_pixels * resolution
    overlay_lon_shift = overlay_lon_shift_pixels * resolution
    
    folium.raster_layers.ImageOverlay(
        name="Predicted Richness PNG Overlay",
        image=str(overlay_path),
        bounds=[
            [lat_min + overlay_lat_shift, lon_min + overlay_lon_shift],
            [lat_max + overlay_lat_shift, lon_max + overlay_lon_shift],
        ],
        opacity=0.85,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    colors = ["#313695", "#4575b4", "#abd9e9", "#ffffbf", "#fdae61", "#f46d43", "#a50026"]
    cmap = bcm.LinearColormap(
        colors=colors,
        index=np.linspace(vmin, vmax, len(colors)).tolist(),
        vmin=vmin,
        vmax=vmax,
        caption="Predicted Species Richness",
    )
    cmap.add_to(m)

    # Make the color scale/legend easier to read on all basemaps.
    legend_css = """
    <style>
      .legend, .leaflet-control, .branca-colormap {
          background: rgba(255, 255, 255, 0.95) !important;
          padding: 8px !important;
          border-radius: 8px !important;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.30) !important;
      }
      .legend text {
          fill: black !important;
          font-weight: 600 !important;
      }
    </style>
    """
    m.get_root().html.add_child(folium.Element(legend_css))

    # Optional observed BioScape ground sites.
    add_site_markers_to_map(
        m,
        sites_csv=sites_csv,
        sites_season=sites_season,
        richness_cmap=cmap,
        richness_vmin=vmin,
        richness_vmax=vmax,
    )
    
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    m.save(out_path)

    print(f"{C.G}[OK] Fast HTML map saved: {out_path}{C.E}")
    print(f"{C.G}[OK] PNG overlay saved: {overlay_path}{C.E}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAIA heatmap: domain-wide richness inference")
    parser.add_argument("--nc_dir", type=str, default=None, help="Directory containing NetCDF .nc files")
    parser.add_argument("--enmap_dir", type=str, default=None, help="Directory containing EnMAP .tif/.tiff files")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--resolution", type=float, default=0.005, help="Output grid resolution in degrees")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--skip-geotiff", action="store_true", help="Skip GeoTIFF output")
    parser.add_argument("--skip-png", action="store_true", help="Skip PNG output")
    parser.add_argument("--skip-html", action="store_true", help="Skip HTML output")
    parser.add_argument("--from-json", type=str, default=None, help="Load predictions from previous JSON instead of running inference")
    parser.add_argument("--max-patches", type=int, default=None, help="Debug: stop after this many valid patches")
    parser.add_argument(
        "--land-only",
        action="store_true",
        help="Drop ocean/water patches using global-land-mask.",
    )
    parser.add_argument(
    "--sites-csv",
    type=str,
    default=None,
    help="Optional CSV of site markers to show on the HTML map.",
    )
    parser.add_argument(
        "--sites-season",
        type=str,
        default="all",
        choices=["all", "wet", "dry"],
        help="Filter site markers by Campaign column if available.",
    )
    parser.add_argument(
        "--resume-json",
        type=str,
        default=None,
        help="Resume from an existing predictions.json and skip source files already processed.",
    )
    return parser.parse_args()


def autodetect_data() -> tuple[Path | None, str, Path]:
    enmap_dir = PROJECT_ROOT / "data" / "enmap" / "southern_cape"
    nc_v2_dir = PROJECT_ROOT / "data" / "bioscape" / "30m_v2"
    nc_v1_dir = PROJECT_ROOT / "data" / "bioscape" / "30m"

    if enmap_dir.exists() and list(enmap_dir.glob("**/*.tif")):
        return enmap_dir, "enmap", PROJECT_ROOT / "reports" / "heatmap_enmap"
    if nc_v2_dir.exists() and list(nc_v2_dir.glob("*.nc")):
        return nc_v2_dir, "nc", PROJECT_ROOT / "reports" / "heatmap"
    if nc_v1_dir.exists() and list(nc_v1_dir.glob("*.nc")):
        return nc_v1_dir, "nc", PROJECT_ROOT / "reports" / "heatmap"
    return None, "auto", PROJECT_ROOT / "reports" / "heatmap"


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else PROJECT_ROOT / "checkpoints" / "gaia_bioscape_best.pth"

    if args.enmap_dir:
        data_dir = Path(args.enmap_dir)
        file_type = "enmap"
        default_out = PROJECT_ROOT / "reports" / "heatmap_enmap"
    elif args.nc_dir:
        data_dir = Path(args.nc_dir)
        file_type = "nc"
        default_out = PROJECT_ROOT / "reports" / "heatmap"
    else:
        data_dir, file_type, default_out = autodetect_data()
        if data_dir is None:
            print(f"{C.F}[!] No data directory found. Download EnMAP or AVIRIS data first.{C.E}")
            print("    EnMAP:  python src/download_enmap.py --download --limit 1")
            print("    AVIRIS: python src/download_region.py --download")
            return

    resume_results = None
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    resume_save_path = out_dir / "predictions.json"
    
    if args.resume_json:
        resume_path = Path(args.resume_json)
        if resume_path.exists():
            print(f"{C.B}[*] Resuming from: {resume_path}{C.E}")
            resume_results = json.loads(resume_path.read_text(encoding="utf-8"))
            if args.land_only:
                resume_results = filter_land_results(resume_results)
            print(f"{C.G}[OK] Loaded {len(resume_results)} existing predictions{C.E}")
        else:
            print(f"{C.W}[WARN] Resume JSON not found, starting fresh: {resume_path}{C.E}")

    print(f"\n{C.BOLD}{C.H}{'=' * 60}")
    print("  GAIA - Domain-Wide Richness Heatmap Generator")
    print(f"{'=' * 60}{C.E}\n")
    print(f"  Data dir:    {data_dir}")
    print(f"  Data type:   {file_type.upper()}")
    print(f"  Checkpoint:  {checkpoint}")
    print(f"  Patch size:  {args.patch_size} x {args.patch_size}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Resolution:  {args.resolution} degrees")
    print(f"  Output dir:  {out_dir}\n")

    if args.from_json:
        json_path = Path(args.from_json)
        print(f"{C.B}[*] Loading predictions from {json_path}{C.E}")
        results = json.loads(json_path.read_text(encoding="utf-8"))

        if args.land_only:
            results = filter_land_results(results)

        print(f"{C.G}[OK] Loaded {len(results)} predictions{C.E}")

    else:
        if not checkpoint.exists():
            print(f"{C.F}[!] Checkpoint not found: {checkpoint}{C.E}")
            print("    Train the model first with: python src/train_production.py")
            return

        results = infer_all(
            data_dir=data_dir,
            checkpoint_path=checkpoint,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            file_type=file_type,
            max_patches=args.max_patches,
            land_only=args.land_only,
            existing_results=resume_results,
            resume_save_path=resume_save_path,
        )

    if not results:
        print(f"{C.F}[!] No valid predictions produced. Check data, checkpoint, and no-data filtering.{C.E}")
        return

    print(f"\n{C.BOLD}-- Generating outputs --{C.E}")

    json_path = out_dir / "predictions.json"
    save_predictions_json(results, json_path)

    if not args.skip_geotiff:
        save_geotiff(results, out_dir / "richness_heatmap.tif", resolution=args.resolution)
    if not args.skip_png:
        save_png(results, out_dir / "richness_heatmap.png", resolution=args.resolution)
    if not args.skip_html:
        save_html(
            results,
            out_dir / "richness_heatmap.html",
            resolution=args.resolution,
            sites_csv=args.sites_csv,
            sites_season=args.sites_season,
        )

    print(f"\n{C.BOLD}{C.G}{'=' * 60}")
    print("  HEATMAP GENERATION COMPLETE")
    print(f"  {len(results)} patches predicted")
    print(f"  Outputs saved to: {out_dir}")
    print(f"{'=' * 60}{C.E}\n")


if __name__ == "__main__":
    main()
