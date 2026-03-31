import shapefile
import json
import os
from pyproj import Transformer

def convert_footprints(shp_path, output_json):
    print(f"[*] Reading shapefile: {shp_path}")
    if not os.path.exists(shp_path):
        print(f"[!] Shapefile not found: {shp_path}")
        return

    # UTM Zone 34S (EPSG:32734) to WGS84 (EPSG:4326)
    # The .prj confirms UTM Zone 34S
    transformer = Transformer.from_crs("epsg:32734", "epsg:4326", always_xy=True)

    sf = shapefile.Reader(shp_path)
    fields = [f[0] for f in sf.fields[1:]] # Skip DeletionFlag
    
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for shape_rec in sf.shapeRecords():
        # Get geometry
        points = shape_rec.shape.points
        
        # Transform UTM to Lat/Lon
        transformed_points = []
        for x, y in points:
            lon, lat = transformer.transform(x, y)
            transformed_points.append([round(lon, 6), round(lat, 6)])
        
        # properties
        props = dict(zip(fields, shape_rec.record))
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [transformed_points]
            },
            "properties": props
        }
        geojson["features"].append(feature)
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(geojson, f)
        
    print(f"[✔] Successfully converted {len(geojson['features'])} footprints to {output_json}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shp = os.path.join(base_dir, "data", "footprint", "ANG_L3_v2_footprints", "ANG_L3_v2_footprints.shp")
    out = os.path.join(base_dir, "viz", "data", "footprints.json")
    convert_footprints(shp, out)
