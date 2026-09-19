"""
Common map grid for TMC-2 (ch2_tmc_ncn_20230130T1900132182_d_img_d32) against the
LROC NAC DTM orthophoto NAC_DTM_PITISCUS_M1149280834_2M.TIF.

Grid convention (fixed here, differs from Stages J-M):
    grid pixel (col i, row j) is the map point
        X = X0 + i * GSD,  Y = Y0 - j * GSD
    and the NAC orthophoto is sampled at that exact point. In rasterio's
    convention NAC pixel c has its centre at left + (c + 0.5) * 2 m, so the
    fractional NAC index is (X - left) / 2 - 0.5. Stages J-M omitted the -0.5,
    sampling the orthophoto 1 m east and 1 m south of the stated map point.

TMC-2 is sampled through the ISRO geometry CSV (Scan, Pixel -> Longitude,
Latitude, both 0-based), with no DEM: any terrain relief displacement in the
geometry product is carried into the grid and must be absorbed by the model.
"""

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

R_MOON = 1737400.0
STD_PARALLEL_DEG = -51.0
CENTRAL_MERIDIAN_DEG = 180.0
GSD = 4.72
X0, X1 = -2841680.0, -2835360.0
Y0, Y1 = -1538943.33, -1566436.0

NAC_REL = "data/lroc_nac/NAC_DTM_PITISCUS_M1149280834_2M.TIF"
TMC_DIR = "data/ch2_tmc_ncn_20230130T1900132182_d_img_d32"
TMC_IMG_REL = f"{TMC_DIR}/data/calibrated/20230130/ch2_tmc_ncn_20230130T1900132182_d_img_d32.img"
TMC_GEO_REL = f"{TMC_DIR}/geometry/calibrated/20230130/ch2_tmc_ncn_20230130T1900132182_g_grd_d32.csv"
TMC_SHAPE = (270516, 4000)
SCAN_RANGE = (239000, 247000)


def map_to_lonlat(X: np.ndarray, Y: np.ndarray):
    """Inverse equirectangular (sphere R_MOON, standard parallel -51, CM 180)."""
    lon = np.degrees(np.asarray(X) / (R_MOON * np.cos(np.radians(STD_PARALLEL_DEG)))) + CENTRAL_MERIDIAN_DEG
    lat = np.degrees(np.asarray(Y) / R_MOON)
    return lon, lat


def lonlat_to_map(lon: np.ndarray, lat: np.ndarray):
    X = R_MOON * (np.radians(np.asarray(lon) - CENTRAL_MERIDIAN_DEG)) * np.cos(np.radians(STD_PARALLEL_DEG))
    Y = R_MOON * np.radians(np.asarray(lat))
    return X, Y


def grid_to_map(col, row):
    return X0 + np.asarray(col) * GSD, Y0 - np.asarray(row) * GSD


def build(repo_root: Path) -> Dict[str, Any]:
    import pandas as pd
    import rasterio
    from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

    repo_root = Path(repo_root)
    with rasterio.open(repo_root / NAC_REL) as src:
        b = src.bounds
        nac = src.read(1)
        nac_crs_wkt = src.crs.to_wkt()
        nac_res = src.res
    xs = np.arange(X0, X1, GSD)
    ys = np.arange(Y0, Y1, -GSD)
    mx, my = np.meshgrid(xs, ys)
    nac_col = ((mx - b.left) / nac_res[0] - 0.5).astype(np.float32)
    nac_row = ((b.top - my) / nac_res[1] - 0.5).astype(np.float32)
    ortho = cv2.remap(nac.astype(np.float32), nac_col, nac_row, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # zero is NAC nodata; exclude anything touching it
    valid_nac = cv2.remap((nac > 0).astype(np.float32), nac_col, nac_row, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999

    df = pd.read_csv(repo_root / TMC_GEO_REL)
    sub = df[(df["Scan"] >= SCAN_RANGE[0]) & (df["Scan"] <= SCAN_RANGE[1])]
    gx, gy = lonlat_to_map(sub["Longitude"].values, sub["Latitude"].values)
    P = np.column_stack([gx, gy])
    i_scan = LinearNDInterpolator(P, sub["Scan"].values)
    i_pix = LinearNDInterpolator(P, sub["Pixel"].values)
    xc = np.linspace(X0 - 100, X1 + 100, 100)
    yc = np.linspace(Y1 - 100, Y0 + 100, 400)
    cx, cy = np.meshgrid(xc, yc)
    scan_map = RegularGridInterpolator((yc, xc), i_scan(cx, cy))((my, mx)).astype(np.float64)
    pix_map = RegularGridInterpolator((yc, xc), i_pix(cx, cy))((my, mx)).astype(np.float64)
    s0 = int(np.floor(np.nanmin(scan_map))) - 10
    s1 = int(np.ceil(np.nanmax(scan_map))) + 10
    mm = np.memmap(repo_root / TMC_IMG_REL, dtype="<u2", mode="r", shape=TMC_SHAPE)
    tmc = cv2.remap(np.asarray(mm[s0:s1, :], dtype=np.float32), pix_map.astype(np.float32),
                    (scan_map - s0).astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return {
        "ortho": ortho, "valid_nac": valid_nac, "tmc": tmc,
        "tmc_scan_map": scan_map, "tmc_pixel_map": pix_map,
        "nac_crs_wkt": nac_crs_wkt, "nac_bounds": tuple(b), "nac_res": tuple(nac_res),
        "shape": ortho.shape,
    }
