"""
Generate a species-richness heatmap by sliding the trained EAGLE model over
every tile in a directory and exporting predictions as a point layer for QGIS.

Usage:
    python -m src.generate_heatmap --checkpoint checkpoints/gaia_eagle_best_strata.pth --mode native
    python -m src.generate_heatmap --checkpoint checkpoints/gaia_eagle_best_strata.pth --mode 10nm --format gpkg
"""
import os
import sys
import json
import argparse
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eagle_dataset import SingleEagleTiffDataset
from src.model_transfer import GaiaTransferModel, Colors


def patch_generator(tif_paths, patch_size, stride, mask_water_vapor):
    """Sequential, single-process sweep over every tile (no DataLoader/workers —
    see the GDAL fork-safety issue that caused training stalls)."""
    for tif_path in tqdm(tif_paths, desc="Tiles"):
        try:
            ds = SingleEagleTiffDataset(tif_path, patch_size=patch_size, mask_water_vapor=mask_water_vapor)
        except Exception as e:
            print(f"{Colors.WARNING}[!] Skipping unreadable tile {os.path.basename(tif_path)}: {e}{Colors.ENDC}")
            continue

        try:
            for patch, lat, lon, row_idx, col_idx in ds.iter_grid_patches(stride=stride):
                yield patch, lat, lon, os.path.basename(tif_path), row_idx, col_idx
        except Exception as e:
            print(f"{Colors.WARNING}[!] Error reading patches from {os.path.basename(tif_path)}: {e}{Colors.ENDC}")
        finally:
            ds.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a richness heatmap from a trained EAGLE model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--mode", type=str, choices=["native", "10nm"], default="native")
    parser.add_argument("--tif_dir", type=str, help="Override data directory")
    parser.add_argument("--stride", type=int, default=None,
                         help="Pixel stride between patch centers (default: patch_size, i.e. non-overlapping)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out", type=str, default=None, help="Output path (default: reports/heatmap_<mode>.<ext>)")
    parser.add_argument("--format", type=str, choices=["geojson", "gpkg"], default="geojson")
    parser.add_argument("--no-mask", dest="mask", action="store_false", help="Disable water vapor masking")
    parser.set_defaults(mask=True)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    patch_size = config['bioscape'].get('patch_size', 16)

    tif_dir = args.tif_dir
    if not tif_dir:
        tif_dir = os.path.join(project_root, "data", "eagle",
                                "30m_native" if args.mode == "native" else "30m_10nm")
    if not os.path.exists(tif_dir):
        print(f"{Colors.FAIL}[!] Directory not found: {tif_dir}{Colors.ENDC}")
        return

    tif_paths = sorted([os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')])

    # Skip anything already known to be broken, if a list exists
    corrupted_txt = os.path.join(project_root, "data", "eagle", "corrupted_files.txt")
    if os.path.exists(corrupted_txt):
        with open(corrupted_txt) as f:
            bad_names = {line.strip() for line in f if line.strip()}
        before = len(tif_paths)
        tif_paths = [p for p in tif_paths if os.path.basename(p) not in bad_names]
        if before != len(tif_paths):
            print(f"{Colors.WARNING}[!] Skipping {before - len(tif_paths)} known-corrupted tif(s){Colors.ENDC}")

    if not tif_paths:
        print(f"{Colors.FAIL}[!] No .tif files found in {tif_dir}{Colors.ENDC}")
        return

    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"{Colors.HEADER}[*] Loading checkpoint: {args.checkpoint} on {device}{Colors.ENDC}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    richness_mean = ckpt['richness_mean']
    richness_std = ckpt['richness_std']

    model = GaiaTransferModel(num_targets=1, patch_size=patch_size).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    out_path = args.out or os.path.join(
        project_root, "reports", f"heatmap_{args.mode}.{'geojson' if args.format == 'geojson' else 'gpkg'}"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    records = []  # only used for --format gpkg
    geojson_f = None
    n_written = 0

    if args.format == "geojson":
        geojson_f = open(out_path, 'w')
        geojson_f.write('{"type": "FeatureCollection", "features": [\n')

    def flush_geojson(batch_lats, batch_lons, batch_preds, batch_tiles, batch_rows, batch_cols):
        nonlocal n_written
        for lat, lon, pred, tile, row, col in zip(batch_lats, batch_lons, batch_preds, batch_tiles, batch_rows, batch_cols):
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                "properties": {"richness": round(float(pred), 3), "tile": tile, "row": row, "col": col},
            }
            prefix = ",\n" if n_written > 0 else ""
            geojson_f.write(prefix + json.dumps(feature))
            n_written += 1

    batch_patches, batch_lats, batch_lons, batch_tiles, batch_rows, batch_cols = [], [], [], [], [], []

    def run_batch():
        nonlocal batch_patches, batch_lats, batch_lons, batch_tiles, batch_rows, batch_cols
        if not batch_patches:
            return
        x = torch.stack(batch_patches).to(device)
        with torch.no_grad():
            preds_norm = model(x).cpu().numpy().flatten()
        preds_real = preds_norm * richness_std + richness_mean

        if args.format == "geojson":
            flush_geojson(batch_lats, batch_lons, preds_real, batch_tiles, batch_rows, batch_cols)
        else:
            for lat, lon, pred, tile, row, col in zip(batch_lats, batch_lons, preds_real, batch_tiles, batch_rows, batch_cols):
                records.append({"lat": lat, "lon": lon, "richness": float(pred), "tile": tile, "row": row, "col": col})

        batch_patches, batch_lats, batch_lons, batch_tiles, batch_rows, batch_cols = [], [], [], [], [], []

    for patch, lat, lon, tile, row, col in patch_generator(tif_paths, patch_size, args.stride, args.mask):
        batch_patches.append(patch)
        batch_lats.append(lat)
        batch_lons.append(lon)
        batch_tiles.append(tile)
        batch_rows.append(row)
        batch_cols.append(col)

        if len(batch_patches) >= args.batch_size:
            run_batch()

    run_batch()  # flush remainder

    if args.format == "geojson":
        geojson_f.write('\n]}\n')
        geojson_f.close()
        print(f"{Colors.OKGREEN}[OK] Wrote {n_written} points to {out_path}{Colors.ENDC}")
    else:
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            print(f"{Colors.FAIL}[!] --format gpkg requires geopandas + shapely "
                  f"(pip install geopandas shapely). Rerun with --format geojson instead.{Colors.ENDC}")
            return
        gdf = gpd.GeoDataFrame(
            records,
            geometry=[Point(r["lon"], r["lat"]) for r in records],
            crs="EPSG:4326",
        )
        gdf.to_file(out_path, driver="GPKG")
        print(f"{Colors.OKGREEN}[OK] Wrote {len(records)} points to {out_path}{Colors.ENDC}")


if __name__ == "__main__":
    main()
