#!/usr/bin/env python3
"""inspect_5m_dem.py — Inspect 5m LOLA DEM and verify topographic relief."""

import sys
from pathlib import Path
import numpy as np
from osgeo import gdal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEM_FILE = PROJECT_ROOT / 'data' / 'dem' / 'Site04_final_adj_5mpp_surf.tif'

def main():
    if not DEM_FILE.exists():
        print(f'❌ DEM file not found: {DEM_FILE}')
        sys.exit(1)

    ds = gdal.Open(str(DEM_FILE))
    if ds is None:
        print(f'❌ Failed to open: {DEM_FILE}')
        sys.exit(1)

    nx, ny = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    data = band.ReadAsArray()
    if nodata is not None:
        valid = data[data != nodata]
    else:
        valid = data[np.isfinite(data)]

    elev_min = float(np.min(valid))
    elev_max = float(np.max(valid))
    elev_mean = float(np.mean(valid))
    elev_std = float(np.std(valid))
    relief = elev_max - elev_min

    pix_x = abs(gt[1])
    pix_y = abs(gt[5])

    print('=' * 70)
    print('🌙 5 m/pixel LOLA DEM Analysis (Barker et al. 2021 PGDA Product 78)')
    print('=' * 70)
    print(f'   Source File:       {DEM_FILE.name}')
    print(f'   Grid Dimensions:   {nx} cols × {ny} rows ({nx * ny:,} total pixels)')
    print(f'   Pixel Resolution:  {pix_x:.2f} m × {pix_y:.2f} m')
    print(f'   Elevation Min:     {elev_min:+.1f} m')
    print(f'   Elevation Max:     {elev_max:+.1f} m')
    print(f'   Elevation Mean:    {elev_mean:+.1f} m (Std: {elev_std:.1f} m)')
    print(f'   Topographic Relief:{relief:.1f} m')
    print(f'   Valid Pixels:      {len(valid):,} / {nx * ny:,} ({len(valid)/(nx*ny)*100:.1f}%)')
    print('=' * 70)

    if relief < 100.0:
        print(f'❌ SANITY CHECK FAILED: Relief ({relief:.1f} m) is < 100 m!')
        sys.exit(1)
    else:
        print(f'✅ SANITY CHECK PASSED: Authentic lunar relief confirmed ({relief:.1f} m > 100 m threshold).')

if __name__ == '__main__':
    main()
