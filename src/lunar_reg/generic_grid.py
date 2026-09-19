"""
Generic common grid: any Chandrayaan-2 product (raw .img + ISRO geometry CSV)
resampled onto an axis-aligned grid in a reference GeoTIFF's own CRS.

Grid convention: grid pixel (col i, row j) is the map point
    X = X0 + i * gsd,  Y = Y0 - j * gsd
The reference is sampled at that exact point (rasterio pixel-centre convention).
When gsd is coarser than the reference resolution, the reference is first
low-passed with a Gaussian of sigma = 0.5 * gsd / res (reference pixels).

The source is sampled through its geometry CSV (Scan, Pixel -> Longitude,
Latitude; 0-based), transformed to the reference CRS with rasterio/PROJ on the
IAU Moon sphere R = 1737400 m. No DEM is applied. An optional k x k block mean
decimates the source first (used for OHRC); decimated index = (orig - (k-1)/2) / k.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

MOON_LONLAT = "+proj=longlat +R=1737400 +no_defs"


def reference_grid(ref_tif: Path, gsd: float, bounds: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
    """bounds = (left, bottom, right, top) in the reference CRS; default = reference bounds."""
    import rasterio
    with rasterio.open(ref_tif) as src:
        b = src.bounds if bounds is None else bounds
        res = src.res[0]
        full_transform = src.transform
        # read only the needed window (+ margin for the blur)
        pad = 4 * max(gsd / res, 1.0) * res
        win = rasterio.windows.from_bounds(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad, src.transform)
        win = win.round_offsets().round_lengths().intersection(
            rasterio.windows.Window(0, 0, src.width, src.height))
        arr = src.read(1, window=win).astype(np.float32)
        wt = src.window_transform(win)
        crs_wkt = src.crs.to_wkt()
        nodata = src.nodata if src.nodata is not None else 0
    valid_src = arr != nodata
    if gsd > res * 1.01:
        sig = 0.5 * gsd / res
        num = cv2.GaussianBlur(np.where(valid_src, arr, 0), (0, 0), sig)
        den = cv2.GaussianBlur(valid_src.astype(np.float32), (0, 0), sig)
        arr = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    X0 = b[0] + gsd / 2.0
    Y0 = b[3] - gsd / 2.0
    ncol = int(np.floor((b[2] - b[0]) / gsd))
    nrow = int(np.floor((b[3] - b[1]) / gsd))
    xs = X0 + np.arange(ncol) * gsd
    ys = Y0 - np.arange(nrow) * gsd
    mx, my = np.meshgrid(xs, ys)
    col = ((mx - wt.c) / wt.a - 0.5).astype(np.float32)
    row = ((my - wt.f) / wt.e - 0.5).astype(np.float32)
    ref = cv2.remap(arr, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap(valid_src.astype(np.float32), col, row, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    return {"ref": ref, "valid": valid, "X0": X0, "Y0": Y0, "gsd": gsd, "crs_wkt": crs_wkt,
            "shape": ref.shape, "bounds": tuple(b), "ref_res": res,
            "ref_transform": tuple(full_transform)[:6]}


def map_to_ref_pixel(grid: Dict, X, Y):
    """0-based pixel-centre (col, row) in the full reference GeoTIFF."""
    a, b_, c, d, e, f = grid["ref_transform"]
    col = (np.asarray(X) - c) / a - 0.5
    row = (np.asarray(Y) - f) / e - 0.5
    return col, row


def grid_to_map(grid: Dict, col, row):
    return grid["X0"] + np.asarray(col) * grid["gsd"], grid["Y0"] - np.asarray(row) * grid["gsd"]


def map_to_lonlat(grid: Dict, X, Y):
    from rasterio.warp import transform
    lon, lat = transform(grid["crs_wkt"], MOON_LONLAT, list(np.ravel(X)), list(np.ravel(Y)))
    return np.asarray(lon).reshape(np.shape(X)), np.asarray(lat).reshape(np.shape(Y))


def source_on_grid(grid: Dict, img_path: Path, img_shape: Tuple[int, int], dtype: str,
                   geo_csv: Path, decimate: int = 1, lattice: int = 8) -> Dict[str, Any]:
    import pandas as pd
    from rasterio.warp import transform
    from scipy.interpolate import LinearNDInterpolator

    df = pd.read_csv(geo_csv)
    gx, gy = transform(MOON_LONLAT, grid["crs_wkt"], list(df["Longitude"].values), list(df["Latitude"].values))
    gx, gy = np.asarray(gx), np.asarray(gy)
    b, m = grid["bounds"], 2000.0
    near = (gx > b[0] - m) & (gx < b[2] + m) & (gy > b[1] - m) & (gy < b[3] + m)
    if near.sum() < 10:
        raise ValueError("source geometry does not reach the reference grid")
    P = np.column_stack([gx[near], gy[near]])
    i_scan = LinearNDInterpolator(P, df["Scan"].values[near].astype(np.float64))
    i_pix = LinearNDInterpolator(P, df["Pixel"].values[near].astype(np.float64))
    h, w = grid["shape"]
    lc = np.arange(0, w + lattice, lattice)
    lr = np.arange(0, h + lattice, lattice)
    LX, LY = grid_to_map(grid, *np.meshgrid(lc, lr))
    ls, lp = i_scan(LX, LY), i_pix(LX, LY)
    # bilinear upsampling of the lattice to every grid pixel (exact at lattice nodes)
    fx = (np.arange(w) / lattice).astype(np.float32)
    fy = (np.arange(h) / lattice).astype(np.float32)
    FX, FY = np.meshgrid(fx, fy)
    nanmask = np.isnan(ls) | np.isnan(lp)
    scan_map = cv2.remap(np.nan_to_num(ls).astype(np.float32), FX, FY, cv2.INTER_LINEAR)
    pix_map = cv2.remap(np.nan_to_num(lp).astype(np.float32), FX, FY, cv2.INTER_LINEAR)
    bad = cv2.remap(nanmask.astype(np.float32), FX, FY, cv2.INTER_LINEAR) > 0
    inside = (~bad) & (scan_map >= 0) & (scan_map <= img_shape[0] - 1) & (pix_map >= 0) & (pix_map <= img_shape[1] - 1)
    if not inside.any():
        raise ValueError("no grid pixel falls inside the source image")
    s0 = int(max(np.floor(scan_map[inside].min()) - 2 * decimate, 0))
    s1 = int(min(np.ceil(scan_map[inside].max()) + 2 * decimate + 1, img_shape[0]))
    mm = np.memmap(img_path, dtype=dtype, mode="r", shape=img_shape)
    raw = np.asarray(mm[s0:s1, :], dtype=np.float32)
    k = decimate
    if k > 1:
        hh, ww = (raw.shape[0] // k) * k, (raw.shape[1] // k) * k
        raw = raw[:hh, :ww].reshape(hh // k, k, ww // k, k).mean(axis=(1, 3))
    map_c = ((pix_map - (k - 1) / 2.0) / k).astype(np.float32)
    map_r = ((scan_map - s0 - (k - 1) / 2.0) / k).astype(np.float32)
    src = cv2.remap(raw, map_c, map_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = inside & (src > 0)
    src[~valid] = 0
    return {"src": src, "valid": valid, "scan_map": scan_map.astype(np.float64),
            "pixel_map": pix_map.astype(np.float64), "scan_range": (s0, s1), "decimate": k}


def native_gsd_from_geometry(geo_csv: Path, near_lonlat: Tuple[float, float], crs_wkt: str) -> Dict[str, float]:
    """Median ground spacing per source pixel along sample and line near a point."""
    import pandas as pd
    from rasterio.warp import transform
    df = pd.read_csv(geo_csv)
    x, y = transform(MOON_LONLAT, crs_wkt, list(df["Longitude"]), list(df["Latitude"]))
    df["x"], df["y"] = x, y
    cx, cy = transform(MOON_LONLAT, crs_wkt, [near_lonlat[0]], [near_lonlat[1]])
    d = np.hypot(df["x"] - cx[0], df["y"] - cy[0])
    sub = df[d < 5000].sort_values(["Scan", "Pixel"])
    samp, line = [], []
    for _, g in sub.groupby("Scan"):
        g = g.sort_values("Pixel")
        dp = np.diff(g["Pixel"].values)
        ds = np.hypot(np.diff(g["x"].values), np.diff(g["y"].values))
        samp.extend(ds[dp > 0] / dp[dp > 0])
    for _, g in sub.groupby("Pixel"):
        g = g.sort_values("Scan")
        dl = np.diff(g["Scan"].values)
        ds = np.hypot(np.diff(g["x"].values), np.diff(g["y"].values))
        line.extend(ds[dl > 0] / dl[dl > 0])
    return {"sample_m": float(np.median(samp)), "line_m": float(np.median(line))}


def source_frame_pair(ref_tif: Path, img_path: Path, img_shape: Tuple[int, int], dtype: str, geo_csv: Path,
                      scan_range: Tuple[int, int], decimate: int, dtm_tif: Optional[Path] = None) -> Dict[str, Any]:
    """Work in the source image's own (line, sample) frame, decimated by k.

    Source grid pixel (i, j) is the source position line = s0 + i*k + (k-1)/2,
    sample = j*k + (k-1)/2, i.e. the centre of a k x k block. The reference is
    sampled at the map point the ISRO geometry assigns to that position
    (bilinear over the geometry lattice in (scan, pixel), which is regular), after
    a Gaussian prefilter matched to the block size. Displacements found in this
    frame are therefore offsets of the reference relative to the geometry
    prediction, in decimated source pixels.
    """
    import pandas as pd
    import rasterio
    from rasterio.warp import transform
    from scipy.interpolate import RegularGridInterpolator

    df = pd.read_csv(geo_csv)
    s0, s1 = scan_range
    k = decimate
    sub = df[(df["Scan"] >= s0 - 200) & (df["Scan"] <= s1 + 200)]
    with rasterio.open(ref_tif) as src:
        crs_wkt = src.crs.to_wkt()
        rt = src.transform
        res = src.res[0]
        nodata = src.nodata if src.nodata is not None else 0
        x, y = transform(MOON_LONLAT, crs_wkt, list(sub["Longitude"]), list(sub["Latitude"]))
        sub = sub.assign(x=x, y=y)
        scans = np.sort(sub["Scan"].unique())
        pixels = np.sort(sub["Pixel"].unique())
        X = sub.pivot(index="Scan", columns="Pixel", values="x").reindex(index=scans, columns=pixels).values
        Y = sub.pivot(index="Scan", columns="Pixel", values="y").reindex(index=scans, columns=pixels).values
        if np.isnan(X).any():
            raise ValueError("geometry lattice is not complete in the requested scan range")
        ix = RegularGridInterpolator((scans, pixels), X)
        iy = RegularGridInterpolator((scans, pixels), Y)
        nrow = (s1 - s0) // k
        ncol = img_shape[1] // k
        lines = s0 + np.arange(nrow) * k + (k - 1) / 2.0
        samples = np.arange(ncol) * k + (k - 1) / 2.0
        L, S = np.meshgrid(lines, samples, indexing="ij")
        pts = np.column_stack([L.ravel(), np.clip(S.ravel(), pixels[0], pixels[-1])])
        MX = ix(pts).reshape(L.shape)
        MY = iy(pts).reshape(L.shape)
        # reference window covering the footprint
        pad = 50 * res
        win = rasterio.windows.from_bounds(MX.min() - pad, MY.min() - pad, MX.max() + pad, MY.max() + pad, rt)
        win = win.round_offsets().round_lengths().intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        ref_arr = src.read(1, window=win).astype(np.float32)
        wt = src.window_transform(win)
    valid_src = ref_arr != nodata
    native = float(np.median(np.hypot(np.diff(MX, axis=1), np.diff(MY, axis=1))))  # ground metres per decimated px
    if native > res * 1.01:
        sig = 0.5 * native / res
        num = cv2.GaussianBlur(np.where(valid_src, ref_arr, 0), (0, 0), sig)
        den = cv2.GaussianBlur(valid_src.astype(np.float32), (0, 0), sig)
        ref_arr = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
    col = ((MX - wt.c) / wt.a - 0.5).astype(np.float32)
    row = ((MY - wt.f) / wt.e - 0.5).astype(np.float32)
    ref = cv2.remap(ref_arr, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ref_valid = cv2.remap(valid_src.astype(np.float32), col, row, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    mm = np.memmap(img_path, dtype=dtype, mode="r", shape=img_shape)
    raw = np.asarray(mm[s0:s0 + nrow * k, :ncol * k], dtype=np.float32)
    srcimg = raw.reshape(nrow, k, ncol, k).mean(axis=(1, 3)) if k > 1 else raw
    dtm = None
    if dtm_tif is not None:
        # heights at the same map points as `ref` (bilinear, NaN outside / nodata)
        with rasterio.open(dtm_tif) as dsrc:
            dwin = rasterio.windows.from_bounds(MX.min() - 100, MY.min() - 100, MX.max() + 100, MY.max() + 100, dsrc.transform)
            dwin = dwin.round_offsets().round_lengths().intersection(rasterio.windows.Window(0, 0, dsrc.width, dsrc.height))
            darr = dsrc.read(1, window=dwin).astype(np.float32)
            dwt = dsrc.window_transform(dwin)
            dnod = dsrc.nodata
        bad = ~np.isfinite(darr) | ((darr == dnod) if dnod is not None else False) | (darr < -1e6)
        darr[bad] = np.nan
        dcol = ((MX - dwt.c) / dwt.a - 0.5).astype(np.float32)
        drow = ((MY - dwt.f) / dwt.e - 0.5).astype(np.float32)
        from scipy.ndimage import map_coordinates
        dtm = map_coordinates(darr, [drow, dcol], order=1, mode="constant", cval=np.nan).astype(np.float32)
    return {"src": srcimg.astype(np.float32), "src_valid": srcimg > 0, "ref": ref, "ref_valid": ref_valid, "dtm": dtm,
            "map_x": MX, "map_y": MY, "lines": lines, "samples": samples, "decimate": k,
            "ground_m_per_px": native, "crs_wkt": crs_wkt, "ref_transform": tuple(rt)[:6], "scan_range": (s0, s1)}
