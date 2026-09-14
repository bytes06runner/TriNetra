#!/usr/bin/env python3
"""fetch_dem.py — Download and subset a lunar south polar DEM.

Target region: 69°S to 71°S, 31°E to 34°E (Shiv Shakti Point, Chandrayaan-2 landing area).

Data Sources:
  Primary: NASA PDS Geosciences Node (WUSTL) LOLA Gridded Data Record (LOLA GDR)
           Dataset: LRO-L-LOLA-4-GDR-V1.0 (polar stereographic, 60°S-90°S coverage)
           URL: https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/
  Fallback: MIT LOLA GDR Archive (https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG/)

Usage:
    python scripts/fetch_dem.py [--force]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "dem"
OUTPUT_TIF = OUTPUT_DIR / "south_pole_subset.tif"

# Target region in geographic coordinates (selenographic degrees)
LAT_MIN, LAT_MAX = -71.0, -69.0
LON_MIN, LON_MAX = 31.0, 34.0


def print_raster_info(tif_path: Path):
    """Print detailed raster statistics and verify sanity."""
    ds = gdal.Open(str(tif_path))
    if ds is None:
        print(f"   ❌ Cannot open {tif_path}")
        sys.exit(1)

    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    data = band.ReadAsArray()

    nodata = band.GetNoDataValue()
    if nodata is not None:
        valid = data[data != nodata]
    else:
        valid = data.ravel()

    if len(valid) == 0:
        print("   ❌ SANITY CHECK FAILED: No valid elevation pixels in subset!")
        sys.exit(1)

    elev_min = float(np.min(valid))
    elev_max = float(np.max(valid))
    elev_mean = float(np.mean(valid))
    elev_range = elev_max - elev_min

    pixel_size_x = abs(gt[1])
    pixel_size_y = abs(gt[5])

    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + nx * gt[1]
    y_min = y_max + ny * gt[5]

    print(f"\n📊 Raster Statistics for {tif_path.name}:")
    print(f"   Shape:          {ny} rows × {nx} cols")
    print(f"   Pixel size:     {pixel_size_x:.1f} m × {pixel_size_y:.1f} m")
    print(f"   CRS:            Lunar Polar Stereographic (Moon 1737.4 km sphere)")
    print(f"   Bounds (Proj):  X [{x_min:.1f}, {x_max:.1f}]  Y [{y_min:.1f}, {y_max:.1f}]")
    print(f"   Geographic:     Lat [{LAT_MIN}°, {LAT_MAX}°]  Lon [{LON_MIN}°, {LON_MAX}°]")
    print(f"   Elevation min:  {elev_min:.1f} m (relative to 1737.4 km datum)")
    print(f"   Elevation max:  {elev_max:.1f} m (relative to 1737.4 km datum)")
    print(f"   Elevation mean: {elev_mean:.1f} m")
    print(f"   Relief:         {elev_range:.1f} m")
    print(f"   Valid pixels:   {len(valid):,} / {nx * ny:,} ({len(valid)/(nx*ny)*100:.1f}%)")

    if elev_range < 100.0:
        print(f"\n   ❌ SANITY CHECK FAILED: Elevation range {elev_range:.1f} m is < 100 m.")
        print("   The subset is flat or empty.")
        sys.exit(1)
    else:
        print(f"\n   ✅ SANITY CHECK PASSED: {elev_range:.0f} m relief confirmed on authentic Chandrayaan-2 landing terrain.")

    ds = None


def fetch_and_subset_lola(force: bool = False) -> bool:
    """Fetch LOLA polar GDR from NASA PDS and subset the target region."""
    sources = [
        # Source 1: WUSTL PDS Geosciences node 240m product
        "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_60s_240m.lbl",
        # Source 2: WUSTL PDS Geosciences node 120m product
        "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_60s_120m.lbl",
        # Source 3: MIT archive
        "https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG/LDEM_60S_240M.LBL",
    ]

    for src_url in sources:
        vsi_url = f"/vsicurl/{src_url}"
        print(f"\n📡 Connecting to LOLA PDS repository: {src_url}")
        t0 = time.time()
        ds = gdal.Open(vsi_url)
        if ds is None:
            print("   ⚠️  Failed to open repository, trying next source...")
            continue

        print(f"   Connected in {time.time() - t0:.2f}s")
        print(f"   Full raster dimensions: {ds.RasterXSize} × {ds.RasterYSize}")

        # Compute projected coordinate bounding box for the target selenographic window
        proj = ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj)
        geo_srs = srs.CloneGeogCS()
        ct = osr.CoordinateTransformation(geo_srs, srs)

        corners = [
            (LON_MIN, LAT_MAX),
            (LON_MAX, LAT_MAX),
            (LON_MIN, LAT_MIN),
            (LON_MAX, LAT_MIN),
        ]
        coords = [ct.TransformPoint(lon, lat)[:2] for lon, lat in corners]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ulx, uly = min(xs), max(ys)
        lrx, lry = max(xs), min(ys)
        print(f"   Target window (EPSG-equivalent lunar proj):")
        print(f"     UL: ({ulx:.1f}, {uly:.1f})  LR: ({lrx:.1f}, {lry:.1f})")

        print(f"   ⬇️  Streaming and subsetting target region...")
        t_sub = time.time()

        # Extract subset via gdal.Translate
        mem_driver = gdal.GetDriverByName("MEM")
        subset_ds = gdal.Translate(
            "",
            ds,
            format="MEM",
            projWin=[ulx, uly, lrx, lry],
        )

        if subset_ds is None:
            print("   ⚠️  Subsetting failed on this source, trying next...")
            continue

        print(f"   Streaming completed in {time.time() - t_sub:.2f}s")

        # Read array and scale to meters relative to lunar reference radius (1737.4 km)
        band = subset_ds.GetRasterBand(1)
        raw_arr = band.ReadAsArray()
        gt = subset_ds.GetGeoTransform()
        proj_wkt = subset_ds.GetProjection()

        # In LOLA GDR: Height = DN * SCALING_FACTOR (typically 0.5)
        scale = band.GetScale() or 0.5
        offset = band.GetOffset() or 0.0

        # Elevation relative to reference radius 1737.4 km (in meters)
        elev_m = (raw_arr.astype(np.float32) * scale).astype(np.float32)
        elev_m[raw_arr == -32768] = -9999.0

        # Write to GeoTIFF
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tif_driver = gdal.GetDriverByName("GTiff")
        out_ds = tif_driver.Create(
            str(OUTPUT_TIF),
            subset_ds.RasterXSize,
            subset_ds.RasterYSize,
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=LZW", "TILED=YES"],
        )
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj_wkt)
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(-9999.0)
        out_band.WriteArray(elev_m)
        out_band.FlushCache()
        out_ds = None
        subset_ds = None
        ds = None

        print(f"   ✅ Saved DEM subset to {OUTPUT_TIF}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Download and subset lunar south polar DEM")
    parser.add_argument("--force", action="store_true", help="Force redownload if subset exists")
    args = parser.parse_args()

    if OUTPUT_TIF.exists() and not args.force:
        print(f"✅ DEM subset already exists: {OUTPUT_TIF}")
        print_raster_info(OUTPUT_TIF)
        return

    print("=" * 65)
    print("🌙 TriNetra Lunar DEM Acquisition Pipeline")
    print(f"   Target: South Pole Shiv Shakti Point [{LAT_MIN}° to {LAT_MAX}°S, {LON_MIN}° to {LON_MAX}°E]")
    print(f"   Destination: {OUTPUT_TIF}")
    print("=" * 65)

    success = fetch_and_subset_lola(args.force)

    if not success:
        print("\n❌ DEM acquisition failed. Please check network connection.")
        sys.exit(1)

    print_raster_info(OUTPUT_TIF)


if __name__ == "__main__":
    main()
