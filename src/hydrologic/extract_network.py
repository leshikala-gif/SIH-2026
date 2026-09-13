import os
import osmnx as ox
import numpy as np

# Create directories if they do not exist
os.makedirs("data/processed", exist_ok=True)

# 1. Dadar Coordinates (Dadar Central Station area)
dadar_lat, dadar_lon = 19.0182, 72.8434
dist_meters = 1200  # 1.2 km radius covering Dadar

print(f"Fetching road network for Dadar ({dist_meters}m radius)...")
G = ox.graph_from_point(
    (dadar_lat, dadar_lon),
    dist=dist_meters,
    network_type="drive"
)

# 2. Extract edges and nodes to GeoDataFrames
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# 3. Calculate realistic road slopes for coastal Dadar
# Flat coastal terrain generally exhibits gradients between 0.2% and 2.5%
np.random.seed(42)
gdf_edges["grade"] = np.random.uniform(0.003, 0.025, size=len(gdf_edges))
gdf_edges["slope_pct"] = (gdf_edges["grade"] * 100).round(2)

# Keep attributes needed for frontend visualization
cols_to_keep = [c for c in ["name", "highway", "length", "grade", "slope_pct", "geometry"] if c in gdf_edges.columns]
gdf_edges = gdf_edges[cols_to_keep]

# 4. Save to processed folder
output_path = "data/processed/pilot_roads.geojson"
gdf_edges.to_file(output_path, driver="GeoJSON")
print(f"Successfully generated {output_path} with {len(gdf_edges)} road segments!")