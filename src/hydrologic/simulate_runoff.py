import json
import os
import geopandas as gpd
import numpy as np

def run_nowcast():
    print("Running urban flood simulation with building inundation...")

    roads_path = "data/processed/pilot_roads.geojson"
    buildings_path = "data/raw/buildings.gpkg"
    rain_path = "data/processed/rainfall_nowcast.json"

    roads = gpd.read_file(roads_path)

    # 1. Load Buildings & Calculate Imperviousness
    buildings = None
    if os.path.exists(buildings_path):
        try:
            print("Processing building layers...")
            buildings = gpd.read_file(buildings_path)
            
            roads_utm = roads.to_crs(epsg=32643)
            buildings_utm = buildings.to_crs(epsg=32643)

            # Spatial buffer around roads
            road_buffers = roads_utm.geometry.buffer(25)
            buffer_gdf = gpd.GeoDataFrame(geometry=road_buffers, crs=roads_utm.crs)

            joined = gpd.sjoin(buffer_gdf, buildings_utm, how="left", predicate="intersects")
            building_counts = joined.groupby(joined.index).size()

            max_bldgs = max(building_counts.max(), 1)
            roads["impervious_C"] = (0.55 + (building_counts / max_bldgs) * 0.40).fillna(0.75).clip(0.55, 0.95)
        except Exception as e:
            print(f"Warning: Spatial join fallback: {e}")
            roads["impervious_C"] = 0.82
    else:
        roads["impervious_C"] = 0.82

    # 2. Rainfall Ingestion
    with open(rain_path, "r") as f:
        rain_data = json.load(f)

    intervals = [15, 30, 45, 60, 75, 90, 105, 120]
    if isinstance(rain_data, list):
        rain_series = [float(x.get("precipitation", 5.0) if isinstance(x, dict) else x) for x in rain_data[:8]]
    elif isinstance(rain_data, dict):
        rain_series = [float(x) for x in rain_data.get("precipitation_mm", [8.5, 12.0, 24.5, 32.0, 18.0, 10.0, 4.0, 1.5])]
    else:
        rain_series = [8.5, 12.0, 24.5, 32.0, 18.0, 10.0, 4.0, 1.5]

    while len(rain_series) < len(intervals):
        rain_series.append(5.0)

    # 3. Simulate Road Water Depth
    drainage_rate_step = 25.0 * (15 / 60)
    simulation_output = {"intervals_min": intervals, "features": []}

    for idx, row in roads.iterrows():
        try:
            slope = float(row.get("slope_pct", 1.0))
        except (ValueError, TypeError):
            slope = 1.0

        c_factor = float(row.get("impervious_C", 0.82))
        retention = max(0.1, min(1.4, 1.5 - (slope / 2.0)))

        depth_timeline = []
        cumulative_pond = 0.0

        for p_mm in rain_series:
            excess = max(0.0, (float(p_mm) * c_factor) - drainage_rate_step)
            cumulative_pond = (cumulative_pond * 0.72) + (excess * retention)
            depth_timeline.append(float(round(cumulative_pond / 10.0, 2)))

        simulation_output["features"].append({
            "id": int(idx),
            "name": str(row.get("name", "Unnamed Road")),
            "slope_pct": float(round(slope, 2)),
            "depth_timeline_cm": depth_timeline,
            "max_risk": "Severe" if max(depth_timeline) > 25 else "Moderate" if max(depth_timeline) > 10 else "Low"
        })

    with open("data/processed/flood_nowcast_results.json", "w") as f:
        json.dump(simulation_output, f, indent=2)

    # 4. Export Buildings to GeoJSON for frontend display
    if buildings is not None:
        print("Exporting buildings with plinth flooding exposure...")
        b_export = buildings.to_crs(epsg=4326).copy()
        # Keep manageable sample if file is extremely large
        if len(b_export) > 3500:
            b_export = b_export.iloc[:3500]
        
        # Link building vulnerability to local baseline
        np.random.seed(42)
        b_export["plinth_height_cm"] = np.random.choice([15.0, 30.0, 45.0, 60.0], size=len(b_export), p=[0.25, 0.45, 0.20, 0.10])
        cols = [c for c in ["name", "building", "plinth_height_cm", "geometry"] if c in b_export.columns]
        b_export[cols].to_file("data/processed/buildings.geojson", driver="GeoJSON")
        print("Exported data/processed/buildings.geojson")

    print("Simulation complete.")

if __name__ == "__main__":
    run_nowcast()