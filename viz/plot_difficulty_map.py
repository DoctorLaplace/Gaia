import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
import math
from tqdm import tqdm
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor

def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_bounds(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_min = xtile / n * 360.0 - 180.0
    lon_max = (xtile + 1) / n * 360.0 - 180.0
    
    lat_rad_top = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_top = math.degrees(lat_rad_top)
    
    lat_rad_bottom = math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n)))
    lat_bottom = math.degrees(lat_rad_bottom)
    
    return lon_min, lat_bottom, lon_max, lat_top

def get_basemap(lat_min, lat_max, lon_min, lon_max, zoom=11):
    """Downloads ESRI World Imagery tiles in parallel and stitches them together."""
    x_min, y_min = latlon_to_tile(lat_max, lon_min, zoom)
    x_max, y_max = latlon_to_tile(lat_min, lon_max, zoom)
    
    # Coordinate validation
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)
    
    num_tiles_x = x_end - x_start + 1
    num_tiles_y = y_end - y_start + 1
    total_tiles = num_tiles_x * num_tiles_y
    
    print(f"[*] Map bounds: Lat [{lat_min:.4f}, {lat_max:.4f}], Lon [{lon_min:.4f}, {lon_max:.4f}]")
    print(f"[*] Zoom level: {zoom} (Tiles: {num_tiles_x}x{num_tiles_y} = {total_tiles} total)")
    
    # Adaptive zoom fallback to prevent downloading massive amounts of tiles
    if total_tiles > 1000:
        print("[!] Too many tiles! Reducing zoom level to avoid rate limits.")
        return get_basemap(lat_min, lat_max, lon_min, lon_max, zoom - 1)
        
    cache_dir = os.path.join("viz", "data", "tile_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # ESRI World Imagery XYZ Tile URL
    tile_url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    # Identify missing tiles
    download_queue = []
    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            tile_file = os.path.join(cache_dir, f"{zoom}_{x}_{y}.jpg")
            if not os.path.exists(tile_file):
                url = tile_url_template.format(z=zoom, x=x, y=y)
                download_queue.append((url, tile_file))
                
    # Define tile downloader
    def download_single(item):
        url, tile_file = item
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                with open(tile_file, "wb") as f:
                    f.write(response.read())
        except Exception:
            # Retry once
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    with open(tile_file, "wb") as f:
                        f.write(response.read())
            except Exception:
                # Save a placeholder black tile on failure
                placeholder = Image.new("RGB", (256, 256), (10, 10, 15))
                placeholder.save(tile_file)

    # Fetch tiles concurrently using ThreadPool
    if download_queue:
        print(f"[*] Downloading {len(download_queue)} missing tiles in parallel...")
        with ThreadPoolExecutor(max_workers=24) as executor:
            list(tqdm(executor.map(download_single, download_queue), total=len(download_queue), desc="Downloading Tiles"))
            
    # Stitch tiles
    print("[*] Stitching tiles into a single high-resolution image...")
    stitched_image = Image.new("RGB", (num_tiles_x * 256, num_tiles_y * 256))
    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            tile_file = os.path.join(cache_dir, f"{zoom}_{x}_{y}.jpg")
            try:
                tile_img = Image.open(tile_file)
                px_x = (x - x_start) * 256
                px_y = (y - y_start) * 256
                stitched_image.paste(tile_img, (px_x, px_y))
            except Exception as e:
                print(f"[!] Warning: Failed to paste tile {x},{y}: {e}")
                
    # Calculate exact bounding coordinates of the stitched image
    lon_left = tile_bounds(x_start, y_start, zoom)[0]
    lat_top = tile_bounds(x_start, y_start, zoom)[3]
    lon_right = tile_bounds(x_end, y_end, zoom)[2]
    lat_bottom = tile_bounds(x_end, y_end, zoom)[1]
    
    return stitched_image, (lon_left, lon_right, lat_bottom, lat_top)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load metadata
    metadata_csv = os.path.join(project_root, "data", "bioscape", "biosoundscape_site_metadata.csv")
    if not os.path.exists(metadata_csv):
        print(f"[!] Metadata file not found at {metadata_csv}")
        return
    df_meta = pd.read_csv(metadata_csv)
    
    # Load rankings CSV
    downloads_dir = os.path.expanduser("~/Downloads")
    stats_path = os.path.join(downloads_dir, "cluster_influence_stats(2).csv")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(downloads_dir, "cluster_influence_stats.csv")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(project_root, "reports", "cluster_influence_stats.csv")
        
    if not os.path.exists(stats_path):
        print(f"[!] Cluster influence stats CSV not found. Please verify the path.")
        return
        
    print(f"[*] Loading performance stats from: {stats_path}")
    df_stats = pd.read_csv(stats_path)
    
    # Merge datasets
    df_merged = pd.merge(df_meta, df_stats, left_on="CLUSTER_ID", right_on="cluster_id")
    
    # Filter noise clusters
    df_valid = df_merged[df_merged["CLUSTER_ID"] != -1].copy()
    
    lat_min, lat_max = df_valid["Latitude"].min() - 0.1, df_valid["Latitude"].max() + 0.1
    lon_min, lon_max = df_valid["Longitude"].min() - 0.1, df_valid["Longitude"].max() + 0.1
    
    # Initialize projection transformer to Web Mercator (EPSG:3857)
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    
    # Get stitched satellite imagery background at zoom 11 (high detail)
    basemap, extent_wgs84 = get_basemap(lat_min, lat_max, lon_min, lon_max, zoom=11)
    
    # Convert basemap extent to Web Mercator to resolve vertical squishing
    lon_left, lon_right, lat_bottom, lat_top = extent_wgs84
    x_left, y_top = transformer.transform(lon_left, lat_top)
    x_right, y_bottom = transformer.transform(lon_right, lat_bottom)
    extent_merc = (x_left, x_right, y_bottom, y_top)
    
    # Convert coordinates of all sites to Mercator space
    x_points, y_points = transformer.transform(df_valid["Longitude"].values, df_valid["Latitude"].values)
    
    # Initialize high-resolution plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(40, 32), dpi=250)
    
    # Display satellite background
    ax.imshow(basemap, extent=extent_merc)
    
    # Plot sites in Mercator space
    scatter = ax.scatter(
        x_points, 
        y_points, 
        c=df_valid["mean_r2_when_val"],
        cmap="RdYlGn", 
        s=70, 
        edgecolors='white', 
        linewidths=0.4, 
        alpha=0.9,
        zorder=3
    )
    
    # Add custom colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.03, shrink=0.5)
    cbar.set_label("Validation R² when Held Out (Higher = Easiest to Predict)", fontsize=22, fontweight='bold', labelpad=12)
    cbar.ax.tick_params(labelsize=18)
    
    # Label cluster centroids using fine annotation offsets
    centroids = df_valid.groupby("CLUSTER_ID")[["Longitude", "Latitude"]].mean().reset_index()
    
    np.random.seed(42)
    
    for _, row in centroids.iterrows():
        cluster_id = int(row["CLUSTER_ID"])
        # Transform centroid to Mercator
        x_c, y_c = transformer.transform(row["Longitude"], row["Latitude"])
        
        # Disperse labels clean of the center points
        angle = np.random.uniform(0, 2 * math.pi)
        dist = 22.0  # Screen points offset distance
        ox = dist * math.cos(angle)
        oy = dist * math.sin(angle)
        
        ax.annotate(
            f"Cluster {cluster_id}",
            xy=(x_c, y_c),
            xytext=(ox, oy),
            textcoords='offset points',
            color='white', 
            fontsize=9, 
            fontweight='bold',
            ha='center', 
            va='center',
            zorder=5,
            arrowprops=dict(
                arrowstyle="-", 
                color='#aaaaaa', 
                lw=0.8, 
                alpha=0.8,
                shrinkA=0, 
                shrinkB=4
            ),
            bbox=dict(
                boxstyle="round,pad=0.3", 
                facecolor='#0d1117', 
                edgecolor='#30363d', 
                alpha=0.85,
                lw=0.8
            )
        )
        
    ax.set_title("GAIA CLUSTER INDEPENDENT GENERALIZATION DIFFICULTY MAP\n(NASA BioSCape - Western Cape, South Africa)", fontsize=32, fontweight='bold', pad=25)
    
    # Hide raw meter-scale axes values to look clean and premium
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Apply Mercator bounding box limits
    x_lim_min, y_lim_min = transformer.transform(lon_min, lat_min)
    x_lim_max, y_lim_max = transformer.transform(lon_max, lat_max)
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)
    
    # Add key insights as text block
    hardest_5 = df_stats.sort_values("mean_r2_when_val").head(5)["cluster_id"].tolist()
    easiest_5 = df_stats.sort_values("mean_r2_when_val", ascending=False).head(5)["cluster_id"].tolist()
    
    text_info = (
        f"Hardest Clusters (Lowest R²):\n"
        f"  {', '.join(map(str, hardest_5))}\n\n"
        f"Easiest Clusters (Highest R²):\n"
        f"  {', '.join(map(str, easiest_5))}"
    )
    ax.text(
        0.02, 0.02, 
        text_info, 
        transform=ax.transAxes, 
        fontsize=18, 
        fontfamily='monospace',
        verticalalignment='bottom', 
        bbox=dict(boxstyle="round,pad=0.8", facecolor='black', edgecolor='white', alpha=0.8, lw=1.0)
    )
    
    # Save visualization
    out_path = os.path.join(project_root, "viz", "cluster_difficulty_map.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Successfully generated and saved difficulty map to {out_path}")

if __name__ == "__main__":
    main()
