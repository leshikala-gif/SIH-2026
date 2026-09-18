import os
import requests
import geopandas as gpd
from shapely.geometry import LineString

CITIES = {
    "bhopal_aiims": {"lat": 23.206, "lon": 77.460},
    "delhi_cp": {"lat": 28.632, "lon": 77.217},
    "delhi_iit": {"lat": 28.545, "lon": 77.193},
    "mumbai_dadar": {"lat": 19.018, "lon": 72.843},
    "mumbai_hindmata": {"lat": 18.999, "lon": 72.843},
    "chennai_central": {"lat": 13.083, "lon": 80.271},
    "bengaluru_central": {"lat": 12.972, "lon": 77.595}
}

os.makedirs("data/processed/geojson_cache", exist_ok=True)
headers = {"User-Agent": "JalMarg-Pipeline/1.0"}

for name, coords in CITIES.items():
    lat, lon = coords["lat"], coords["lon"]
    cache_key = f"roads_{round(lat, 3)}_{round(lon, 3)}.geojson"
    out_path = os.path.join("data/processed/geojson_cache", cache_key)
    
    if os.path.exists(out_path):
        print(f"Skipping {name} (already cached at {out_path})")
        continue

    delta_deg = 1000 / 111320.0
    south, north = round(lat - delta_deg, 5), round(lat + delta_deg, 5)
    west, east = round(lon - delta_deg, 5), round(lon + delta_deg, 5)

    query = f"""[out:json][timeout:30];
    (way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]({south},{west},{north},{east}););
    out geom qt;"""

    print(f"Fetching real street networks for {name}...")
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, headers=headers, timeout=30)
        elements = r.json().get("elements", [])
        features = []
        for el in elements:
            if el.get("type") == "way" and "geometry" in el:
                pts = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
                if len(pts) >= 2:
                    features.append({
                        "geometry": LineString(pts),
                        "name": el.get("tags", {}).get("name", "Corridor Link"),
                        "highway": el.get("tags", {}).get("highway", "residential"),
                        "slope_pct": 0.6
                    })
        if features:
            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            gdf.to_file(out_path, driver="GeoJSON")
            print(f"✓ Successfully cached {len(features)} roads for {name}")
        else:
            print(f"⚠ No roads found for {name}")
    except Exception as e:
        print(f"❌ Failed to fetch {name}: {e}")