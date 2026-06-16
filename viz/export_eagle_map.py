import os
import re
import json
import pandas as pd
import numpy as np
import rasterio
from pyproj import Transformer
from tqdm import tqdm

def main():
    project_root = r"E:\Git Repositories\Laboratory\Models\Gaia"
    tif_dir = os.path.join(project_root, "data", "eagle", "30m_10nm")
    richness_csv = os.path.join(project_root, "data", "bioscape", "western_cape_site_embeddings.csv")
    metadata_csv = os.path.join(project_root, "data", "bioscape", "biosoundscape_site_metadata.csv")
    output_html = os.path.join(project_root, "viz", "eagle_map.html")
    
    if not os.path.exists(tif_dir):
        print(f"[!] Directory not found: {tif_dir}")
        return
    if not os.path.exists(richness_csv):
        print(f"[!] CSV not found: {richness_csv}")
        return
        
    df = pd.read_csv(richness_csv)
    df = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'richness': 'richness', 'Richness': 'richness'})
    
    # Load cluster IDs
    if os.path.exists(metadata_csv):
        meta_df = pd.read_csv(metadata_csv)[['SiteID', 'CLUSTER_ID']]
        df = df.merge(meta_df, on='SiteID', how='left')
        df['CLUSTER_ID'] = df['CLUSTER_ID'].fillna(-1).astype(int)
    else:
        df['CLUSTER_ID'] = -1
        
    tif_files = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')]
    print(f"[*] Mapping sites across all {len(tif_files)} tiles...")
    
    mapped_sites = []
    seen_coords = set()
    
    for tif_path in tqdm(tif_files, desc="Processing tiles"):
        try:
            with rasterio.open(tif_path) as src:
                bounds = src.bounds
                crs = src.crs
                inv_transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
                transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
                
                lon1, lat1 = inv_transformer.transform(bounds.left, bounds.bottom)
                lon2, lat2 = inv_transformer.transform(bounds.right, bounds.top)
                min_lon, min_lat = min(lon1, lon2), min(lat1, lat2)
                max_lon, max_lat = max(lon1, lon2), max(lat1, lat2)
                
                mask = (df['lat'] >= min_lat) & (df['lat'] <= max_lat) & \
                       (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
                relevant_sites = df[mask]
                
                if len(relevant_sites) > 0:
                    for idx, row in relevant_sites.iterrows():
                        easting, northing = transformer.transform(row['lon'], row['lat'])
                        col_idx, row_idx = ~src.transform * (easting, northing)
                        col_idx, row_idx = int(col_idx), int(row_idx)
                        
                        p = 8
                        is_inside = (row_idx >= p and row_idx < src.height - p and
                                     col_idx >= p and col_idx < src.width - p)
                        
                        if is_inside:
                            coord_key = (round(row['lat'], 5), round(row['lon'], 5))
                            if coord_key not in seen_coords:
                                # Quick nodata verification
                                val = src.read(1, window=((row_idx-1, row_idx+2), (col_idx-1, col_idx+2)))
                                if not np.isnan(val).all():
                                    mapped_sites.append({
                                        'lat': float(row['lat']),
                                        'lon': float(row['lon']),
                                        'richness': float(row['richness']),
                                        'cluster_id': int(row['CLUSTER_ID']),
                                        'site_id': str(row.get('SiteID', f"Site_{idx}"))
                                    })
                                    seen_coords.add(coord_key)
        except Exception as e:
            pass
            
    print(f"[OK] Mapped {len(mapped_sites)} unique valid richness sites.")
    
    # Generate Leaflet HTML
    sites_json = json.dumps([[s['lat'], s['lon'], s['richness'], s['cluster_id'], s['site_id']] for s in mapped_sites])
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>EAGLE 10nm FWHM Richness Sites Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body, #map {{
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
        }}
        .info-legend {{
            background: rgba(30, 30, 40, 0.95);
            color: #fff;
            padding: 12px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            line-height: 1.6;
            box-shadow: 0 0 15px rgba(0,0,0,0.5);
            border: 1px solid #30363d;
        }}
        .info-legend h4 {{
            margin: 0 0 8px 0;
            font-size: 14px;
            font-weight: bold;
            border-bottom: 1px solid #30363d;
            padding-bottom: 4px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        // Calculate center of all mapped sites
        var sites = {sites_json};
        
        var avg_lat = 0, avg_lon = 0;
        if (sites.length > 0) {{
            var sum_lat = 0, sum_lon = 0;
            for (var i = 0; i < sites.length; i++) {{
                sum_lat += sites[i][0];
                sum_lon += sites[i][1];
            }}
            avg_lat = sum_lat / sites.length;
            avg_lon = sum_lon / sites.length;
        }} else {{
            avg_lat = -33.9;
            avg_lon = 19.5;
        }}
        
        var map = L.map('map').setView([avg_lat, avg_lon], 9);
        
        var dark = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO'
        }});
        
        var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, USDA, USGS'
        }}).addTo(map);
        
        L.control.layers({{
            "Satellite": satellite,
            "Dark Mode": dark
        }}).addTo(map);
        
        // Helper to get color based on richness
        function getColor(r) {{
            return r > 25 ? '#d73027' :
                   r > 20 ? '#fc8d59' :
                   r > 15 ? '#fee08b' :
                   r > 10 ? '#d9ef8b' :
                   r > 5  ? '#91cf60' :
                            '#1a9850';
        }}
        
        // Add markers
        sites.forEach(function(site) {{
            var marker = L.circleMarker([site[0], site[1]], {{
                radius: 6,
                fillColor: getColor(site[2]),
                color: "#ffffff",
                weight: 1,
                opacity: 1,
                fillOpacity: 0.85
            }}).addTo(map);
            
            marker.bindPopup(
                "<b>Site ID:</b> " + site[4] + "<br>" +
                "<b>Species Richness:</b> " + site[2] + "<br>" +
                "<b>Cluster ID:</b> " + (site[3] === -1 ? 'Noise (-1)' : site[3]) + "<br>" +
                "<b>Latitude:</b> " + site[0].toFixed(5) + "<br>" +
                "<b>Longitude:</b> " + site[1].toFixed(5)
            );
        }});
        
        // Add legend
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'info-legend');
            div.innerHTML = '<h4>Richness Gradient</h4>' +
                '<i style="background:#1a9850; width:12px; height:12px; display:inline-block; margin-right:6px; vertical-align:middle;"></i> &lt;= 5<br>' +
                '<i style="background:#91cf60; width:12px; height:12px; display:inline-block; margin-right:6px; vertical-align:middle;"></i> 6 - 10<br>' +
                '<i style="background:#d9ef8b; width:12px; height:12px; display:inline-block; margin-right:6px; vertical-align:middle;"></i> 11 - 15<br>' +
                '<i style="background:#fee08b; width:12px; height:12px; display:inline-block; margin-right:6px; vertical-align:middle;"></i> 16 - 20<br>' +
                '<i style="background:#fc8d59; width:12px; height:12px; display:inline-block; margin-right:6px; vertical-align:middle;"></i> 21 - 25<br>' +
                '<i style="background:#d73027; width:12px; height:12px; display:inline-block; margin-right:6px; vertical-align:middle;"></i> &gt; 25<br><br>' +
                '<b>Total Sites:</b> ' + sites.length;
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Generated interactive Leaflet map at: {output_html}")

if __name__ == "__main__":
    main()
