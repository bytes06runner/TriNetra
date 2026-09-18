import numpy as np
import rasterio
from rasterio.transform import from_origin
import cv2
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

with open('assets/real_cache/stage_l_geometric_filter_results.json') as f:
    data = json.load(f)

# L1 nodes and L2 10m inliers
inl_10m = data['l2_ransac']['threshold_sweeps']['10m']['inliers']
print(f"Loaded {len(inl_10m)} RANSAC 10m inliers.")

os.makedirs('results/figures', exist_ok=True)
gsd = 4.72
overlap_left = -2841680.0
overlap_top = -1538943.33

df_inl = pd.DataFrame(inl_10m)
df_inl['map_x'] = overlap_left + df_inl['grid_x'] * gsd
df_inl['map_y'] = overlap_top - df_inl['grid_y'] * gsd
df_inl['dx_m'] = df_inl['dx_sub'] * gsd
df_inl['dy_m'] = df_inl['dy_sub'] * gsd
df_inl['dx_ref_px'] = df_inl['dx_sub'] * (gsd / 2.00)
df_inl['dy_ref_px'] = df_inl['dy_sub'] * (gsd / 2.00)

csv_path = 'results/stage_l_inliers_10m.csv'
df_inl.to_csv(csv_path, index=False)
print(f"Exported {csv_path}")

# GeoJSON
features = []
R = 1737400.0
phi1 = np.radians(-51.0)
lam0 = np.radians(180.0)

for _, row in df_inl.iterrows():
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

geojson_path = 'results/stage_l_inliers_10m.geojson'
with open(geojson_path, 'w') as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
print(f"Exported {geojson_path}")

# Quiver plot for 10m inliers
plt.figure(figsize=(10, 14))
plt.quiver(df_inl['grid_x'], df_inl['grid_y'], df_inl['dx_sub'], -df_inl['dy_sub'], 
           df_inl['peak'], cmap='viridis', scale=20, width=0.005)
plt.colorbar(label='Peak Phase Congruency NCC')
plt.title(f'Stage L: Filtered RANSAC 10m Inliers (N={len(inl_10m)})\nPitiscus Lobate Scarp (TMC-2 vs NAC Ortho)')
plt.xlabel('Grid X (pixels, GSD 4.72 m)')
plt.ylabel('Grid Y (pixels, GSD 4.72 m)')
plt.gca().invert_yaxis()
plt.tight_layout()
quiver_path = 'results/figures/stage_l_quiver_plot_filtered.png'
plt.savefig(quiver_path, dpi=200)
plt.close()
print(f"Exported {quiver_path}")

# Peak map
plt.figure(figsize=(8, 14))
scatter = plt.scatter(df_inl['grid_x'], df_inl['grid_y'], c=df_inl['peak'], cmap='magma', s=60, edgecolors='black')
plt.colorbar(scatter, label='Peak NCC')
plt.title(f'Stage L: Filtered Peak Field (N={len(inl_10m)})')
plt.xlabel('Grid X (pixels)')
plt.ylabel('Grid Y (pixels)')
plt.gca().invert_yaxis()
plt.tight_layout()
peakmap_path = 'results/figures/stage_l_peak_map_filtered.png'
plt.savefig(peakmap_path, dpi=200)
plt.close()
print(f"Exported {peakmap_path}")

# Filtered Displacement Field GeoTIFF
grid_dx = np.full((5825, 1339), np.nan, dtype=np.float32)
grid_dy = np.full((5825, 1339), np.nan, dtype=np.float32)
for _, row in df_inl.iterrows():
    gx = int(row['grid_x'])
    gy = int(row['grid_y'])
    grid_dx[gy, gx] = row['dx_m']
    grid_dy[gy, gx] = row['dy_m']

transform = from_origin(overlap_left, overlap_top, gsd, gsd)
tif_path = 'results/stage_l_displacement_field_filtered.tif'
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
    dst.set_band_description(1, 'dx_displacement_meters_filtered')
    dst.set_band_description(2, 'dy_displacement_meters_filtered')
print(f"Exported {tif_path}")

