import numpy as np
import rasterio
from rasterio.transform import from_origin
import cv2
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

with open('assets/real_cache/stage_k_dense_results.json') as f:
    data = json.load(f)

nodes = data['nodes']
print(f"Loaded {len(nodes)} accepted nodes.")

os.makedirs('results/figures', exist_ok=True)

# 1. Export CSV
df_nodes = pd.DataFrame(nodes)
# Add ground coordinates
overlap_left = -2841680.0
overlap_top = -1538943.33
gsd = 4.72

df_nodes['map_x'] = overlap_left + df_nodes['grid_x'] * gsd
df_nodes['map_y'] = overlap_top - df_nodes['grid_y'] * gsd
df_nodes['dx_m'] = df_nodes['dx_sub'] * gsd
df_nodes['dy_m'] = df_nodes['dy_sub'] * gsd
df_nodes['dx_ref_px'] = df_nodes['dx_sub'] * (gsd / 2.00)
df_nodes['dy_ref_px'] = df_nodes['dy_sub'] * (gsd / 2.00)

csv_path = 'results/stage_k_accepted_nodes.csv'
df_nodes.to_csv(csv_path, index=False)
print(f"Exported {csv_path}")

# 2. Export GeoJSON
features = []
R = 1737400.0
phi1 = np.radians(-51.0)
lam0 = np.radians(180.0)

for _, row in df_nodes.iterrows():
    # Convert map_x, map_y back to lon, lat
    # map_x = R * (np.radians(lon) - lam0) * np.cos(phi1) -> lon
    lon_deg = np.degrees(row['map_x'] / (R * np.cos(phi1)) + lam0)
    lat_deg = np.degrees(row['map_y'] / R)
    
    feat = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(lon_deg), float(lat_deg)]
        },
        "properties": {
            "node_idx": int(row['ix']),
            "row_idx": int(row['iy']),
            "grid_x": int(row['grid_x']),
            "grid_y": int(row['grid_y']),
            "dx_sub_px": float(row['dx_sub']),
            "dy_sub_px": float(row['dy_sub']),
            "dx_m": float(row['dx_m']),
            "dy_m": float(row['dy_m']),
            "dx_ref_px": float(row['dx_ref_px']),
            "dy_ref_px": float(row['dy_ref_px']),
            "peak_ncc": float(row['peak']),
            "second_peak": float(row['second_peak'])
        }
    }
    features.append(feat)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

geojson_path = 'results/stage_k_accepted_nodes.geojson'
with open(geojson_path, 'w') as f:
    json.dump(geojson, f, indent=2)
print(f"Exported {geojson_path}")

# 3. Export Quiver Plot & Peak Map
plt.figure(figsize=(10, 14))
plt.quiver(df_nodes['grid_x'], df_nodes['grid_y'], df_nodes['dx_sub'], -df_nodes['dy_sub'], 
           df_nodes['peak'], cmap='viridis', scale=50, width=0.003)
plt.colorbar(label='Peak Phase Congruency NCC')
plt.title(f'Stage K: Accepted Displacements (N={len(nodes)})\nPitiscus Lobate Scarp (TMC-2 vs NAC Ortho)')
plt.xlabel('Grid X (pixels, GSD 4.72 m)')
plt.ylabel('Grid Y (pixels, GSD 4.72 m)')
plt.gca().invert_yaxis()
plt.tight_layout()
quiver_path = 'results/figures/stage_k_quiver_plot.png'
plt.savefig(quiver_path, dpi=200)
plt.close()
print(f"Exported {quiver_path}")

# Peak correlation map
plt.figure(figsize=(8, 14))
scatter = plt.scatter(df_nodes['grid_x'], df_nodes['grid_y'], c=df_nodes['peak'], cmap='magma', s=25, edgecolors='none')
plt.colorbar(scatter, label='Peak NCC')
plt.title(f'Stage K: Correlation Peak Field (N={len(nodes)})')
plt.xlabel('Grid X (pixels)')
plt.ylabel('Grid Y (pixels)')
plt.gca().invert_yaxis()
plt.tight_layout()
peakmap_path = 'results/figures/stage_k_peak_map.png'
plt.savefig(peakmap_path, dpi=200)
plt.close()
print(f"Exported {peakmap_path}")

# 4. Export Displacement Field GeoTIFF
# Grid is 1339 x 5825
# Create 2-band GeoTIFF: Band 1 = dx (meters), Band 2 = dy (meters)
# Interpolate sparse nodes to dense grid using nearest/linear
grid_dx = np.full((5825, 1339), np.nan, dtype=np.float32)
grid_dy = np.full((5825, 1339), np.nan, dtype=np.float32)

for _, row in df_nodes.iterrows():
    gx = int(row['grid_x'])
    gy = int(row['grid_y'])
    grid_dx[gy, gx] = row['dx_m']
    grid_dy[gy, gx] = row['dy_m']

# Save as GeoTIFF
transform = from_origin(overlap_left, overlap_top, gsd, gsd)
tif_path = 'results/stage_k_displacement_field.tif'
with rasterio.open(
    tif_path, 'w',
    driver='GTiff',
    height=5825, width=1339,
    count=2,
    dtype=rasterio.float32,
    crs='PROJCS["Moon_2000_Equirectangular",GEOGCS["Moon_2000",DATUM["Moon_2000",SPHEROID["Moon_2000",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",-51],PARAMETER["central_meridian",180],UNIT["metre",1]]',
    transform=transform,
    nodata=np.nan
) as dst:
    dst.write(grid_dx, 1)
    dst.write(grid_dy, 2)
    dst.set_band_description(1, 'dx_displacement_meters')
    dst.set_band_description(2, 'dy_displacement_meters')

print(f"Exported {tif_path}")

