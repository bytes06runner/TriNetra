import numpy as np
import rasterio
import cv2
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy import stats
import json
import time
import os
import matplotlib.pyplot as plt
from rasterio.transform import from_origin

from src.phase_congruency import compute_phase_congruency

print("=== STAGE L: GEOMETRIC CONSISTENCY FILTERING ===")
print("Step 1: Reprojecting to Common Grid (4.72 m/px)...")

with rasterio.open('data/lroc_nac/NAC_DTM_PITISCUS_M1149280834_2M.TIF') as src:
    ortho_bounds = src.bounds
    ortho_img = src.read(1)

overlap_left = -2841680.0
overlap_right = -2835360.0
overlap_bottom = -1566436.0
overlap_top = -1538943.33
gsd = 4.72

xs_dense = np.arange(overlap_left, overlap_right, gsd)
ys_dense = np.arange(overlap_top, overlap_bottom, -gsd)

mesh_x, mesh_y = np.meshgrid(xs_dense, ys_dense)
ortho_cols = ((mesh_x - ortho_bounds.left) / 2.0).astype(np.float32)
ortho_rows = ((ortho_bounds.top - mesh_y) / 2.0).astype(np.float32)
ortho_reproj = cv2.remap(ortho_img.astype(np.float32), ortho_cols, ortho_rows, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

df = pd.read_csv('data/ch2_tmc_ncn_20230130T1900132182_d_img_d32/geometry/calibrated/20230130/ch2_tmc_ncn_20230130T1900132182_g_grd_d32.csv')
R = 1737400.0
phi1 = np.radians(-51.0)
lam0 = np.radians(180.0)

sub = df[(df['Scan'] >= 239000) & (df['Scan'] <= 247000)].copy()
sub['X'] = R * (np.radians(sub['Longitude']) - lam0) * np.cos(phi1)
sub['Y'] = R * np.radians(sub['Latitude'])
pts_xy = np.column_stack([sub['X'], sub['Y']])
interp_scan = LinearNDInterpolator(pts_xy, sub['Scan'])
interp_pixel = LinearNDInterpolator(pts_xy, sub['Pixel'])

xs_coarse = np.linspace(overlap_left - 100, overlap_right + 100, 100)
ys_coarse = np.linspace(overlap_bottom - 100, overlap_top + 100, 400)
cx, cy = np.meshgrid(xs_coarse, ys_coarse)
scans_c = interp_scan(cx, cy)
pixels_c = interp_pixel(cx, cy)
rgi_s = RegularGridInterpolator((ys_coarse, xs_coarse), scans_c, method='linear')
rgi_p = RegularGridInterpolator((ys_coarse, xs_coarse), pixels_c, method='linear')
dense_scans = rgi_s((mesh_y, mesh_x)).astype(np.float32)
dense_pixels = rgi_p((mesh_y, mesh_x)).astype(np.float32)

scan_min = int(np.floor(dense_scans.min())) - 10
scan_max = int(np.ceil(dense_scans.max())) + 10
mm = np.memmap('data/ch2_tmc_ncn_20230130T1900132182_d_img_d32/data/calibrated/20230130/ch2_tmc_ncn_20230130T1900132182_d_img_d32.img',
               dtype='<u2', mode='r', shape=(270516, 4000))
tmc_slice = mm[scan_min:scan_max, :].astype(np.float32)
map_x = dense_pixels
map_y = (dense_scans - scan_min).astype(np.float32)
tmc_reproj = cv2.remap(tmc_slice, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

ortho_blur = cv2.GaussianBlur(ortho_reproj, (7, 7), 1.2)

# Global offset application (K1)
dx_bulk = -82.12
dy_bulk = 224.04
M_shift = np.float32([[1, 0, dx_bulk], [0, 1, dy_bulk]])
tmc_shifted = cv2.warpAffine(tmc_reproj, M_shift, (tmc_reproj.shape[1], tmc_reproj.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

valid_mask = (ortho_reproj > 0) & (tmc_shifted > 0)

print("Step 2: Computing Phase Congruency (K2a)...")
t0 = time.time()
pc_ortho_u8, pc_ortho_f = compute_phase_congruency(ortho_blur, nscale=4, norient=6)
pc_tmc_u8, pc_tmc_f = compute_phase_congruency(tmc_shifted, nscale=4, norient=6)
print(f"Phase congruency completed in {time.time() - t0:.1f}s")

# Correlator function returning accepted node records and peaks
def run_phase_correlation(ref_img, src_img, valid_m, n_x=16, n_y=70, srch_sz=128, tpl_sz=64, peak_thresh=0.40):
    half_t = tpl_sz // 2
    half_s = srch_sz // 2
    radius = half_s - half_t
    H, W = ref_img.shape[:2]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*half_s + 1, 2*half_s + 1))
    safe_mask = cv2.erode(valid_m.astype(np.uint8), kernel) > 0
    valid_rows = np.where(safe_mask.any(axis=1))[0]
    ys = np.linspace(valid_rows.min(), valid_rows.max(), n_y, dtype=int)
    nodes = []
    for iy, y in enumerate(ys):
        row_cols = np.where(safe_mask[y, :])[0]
        if len(row_cols) >= n_x:
            xs = np.linspace(row_cols.min(), row_cols.max(), n_x, dtype=int)
        else:
            xs = np.linspace(half_s + 10, W - half_s - 10, n_x, dtype=int)
        for ix, x in enumerate(xs):
            nodes.append((x, y, ix, iy))
            
    tot = len(nodes)
    accepted = []
    rej_p, rej_u, rej_r, rej_nd = 0, 0, 0, 0
    all_peaks = []
    
    for x, y, ix, iy in nodes:
        if (y - half_s < 0 or y + half_s > H or x - half_s < 0 or x + half_s > W):
            rej_nd += 1
            all_peaks.append(0.0)
            continue
            
        tpl_mask = valid_m[y - half_t : y + half_t, x - half_t : x + half_t]
        srch_mask = valid_m[y - half_s : y + half_s, x - half_s : x + half_s]
        if np.any(~tpl_mask) or np.any(~srch_mask):
            rej_nd += 1
            all_peaks.append(0.0)
            continue
            
        tpl = ref_img[y - half_t : y + half_t, x - half_t : x + half_t].astype(np.float32)
        srch = src_img[y - half_s : y + half_s, x - half_s : x + half_s].astype(np.float32)
        
        if tpl.std() < 1e-4 or srch.std() < 1e-4:
            rej_nd += 1
            all_peaks.append(0.0)
            continue
            
        c = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(c)
        all_peaks.append(float(mx))
        
        if mx < peak_thresh:
            rej_p += 1
            continue
            
        c_supp = c.copy()
        py, px = loc[1], loc[0]
        c_supp[max(0, py-2):min(c.shape[0], py+3), max(0, px-2):min(c.shape[1], px+3)] = -1.0
        second_mx = cv2.minMaxLoc(c_supp)[1]
        if second_mx >= 0.90 * mx:
            rej_u += 1
            continue
            
        if px <= 0 or px >= c.shape[1] - 1 or py <= 0 or py >= c.shape[0] - 1:
            rej_r += 1
            continue
            
        patch = c[py-1:py+2, px-1:px+2]
        denom_x = 2 * (patch[1, 0] - 2 * patch[1, 1] + patch[1, 2])
        denom_y = 2 * (patch[0, 1] - 2 * patch[1, 1] + patch[2, 1])
        
        delta_x = (patch[1, 0] - patch[1, 2]) / denom_x if abs(denom_x) > 1e-6 else 0.0
        delta_y = (patch[0, 1] - patch[2, 1]) / denom_y if abs(denom_y) > 1e-6 else 0.0
        if abs(delta_x) > 1.0 or abs(delta_y) > 1.0:
            delta_x, delta_y = 0.0, 0.0
            
        dx_sub = (loc[0] + half_t) - half_s + delta_x
        dy_sub = (loc[1] + half_t) - half_s + delta_y
        
        if np.hypot(dx_sub, dy_sub) > radius:
            rej_r += 1
            continue
            
        accepted.append({
            'ix': int(ix), 'iy': int(iy),
            'grid_x': int(x), 'grid_y': int(y),
            'dx_sub': float(dx_sub), 'dy_sub': float(dy_sub),
            'peak': float(mx), 'second_peak': float(second_mx)
        })
        
    return accepted, tot, (rej_p, rej_u, rej_r, rej_nd), float(np.mean(all_peaks)), np.array(all_peaks)

print("\nRunning genuine correlation on conforming 16x70 grid...")
gen_acc, gen_tot, gen_rejs, gen_mp, gen_pks = run_phase_correlation(pc_ortho_u8, pc_tmc_u8, valid_mask)
print(f"Genuine accepted: {len(gen_acc)} / {gen_tot} ({len(gen_acc)/gen_tot*100:.2f}%)")

# Geometry evaluation helper for any set of nodes
def eval_geometric_fit(node_list, name=''):
    N = len(node_list)
    if N < 4:
        return {'count': N, 'rmse_m': None, 'rmse_ref_px': None, 'median_m': None, 'p95_m': None, 'occupancy_8x8': f'{N}/64', 'cv': None, 'scale': None, 'rot_deg': None, 'tx_m': None, 'ty_m': None}
    src_pts = np.array([[n['grid_x'] + n['dx_sub'], n['grid_y'] + n['dy_sub']] for n in node_list], dtype=np.float32)
    ref_pts = np.array([[n['grid_x'], n['grid_y']] for n in node_list], dtype=np.float32)
    
    M_sim, inliers = cv2.estimateAffinePartial2D(src_pts, ref_pts)
    pred_ref = (M_sim[:, :2] @ src_pts.T + M_sim[:, 2:3]).T
    res_m = np.linalg.norm(ref_pts - pred_ref, axis=1) * gsd
    res_ref_px = res_m / 2.00
    
    rmse_m = float(np.sqrt(np.mean(res_m**2)))
    rmse_ref_px = float(np.sqrt(np.mean(res_ref_px**2)))
    med_m = float(np.median(res_m))
    p95_m = float(np.percentile(res_m, 95))
    
    grid_8x8 = np.zeros((8, 8), dtype=int)
    for n in node_list:
        bin_x = min(7, int(n['grid_x'] / 1339 * 8))
        bin_y = min(7, int(n['grid_y'] / 5825 * 8))
        grid_8x8[bin_y, bin_x] += 1
    occ = int(np.count_nonzero(grid_8x8))
    cv_val = float(grid_8x8.std() / grid_8x8.mean()) if grid_8x8.mean() > 0 else float('nan')
    
    scale = float(np.sqrt(M_sim[0, 0]**2 + M_sim[1, 0]**2))
    rot_deg = float(np.degrees(np.arctan2(M_sim[1, 0], M_sim[0, 0])))
    tx_m = float(M_sim[0, 2] * gsd)
    ty_m = float(M_sim[1, 2] * gsd)
    
    return {
        'count': N, 'rmse_m': rmse_m, 'rmse_ref_px': rmse_ref_px,
        'median_m': med_m, 'p95_m': p95_m,
        'occupancy_8x8': f'{occ}/64 ({occ/64*100:.1f}%)', 'cv': cv_val,
        'scale': scale, 'rot_deg': rot_deg, 'tx_m': tx_m, 'ty_m': ty_m,
        'M_sim': M_sim.tolist()
    }

# L1: VECTOR FIELD MEDIAN FILTER
def apply_vector_median_filter(node_list, k=8):
    if len(node_list) <= k:
        return node_list, np.array([]), 0.0, 0.0
    coords = np.array([[n['grid_x'] * gsd, n['grid_y'] * gsd] for n in node_list], dtype=np.float32)
    disps_m = np.array([[n['dx_sub'] * gsd, n['dy_sub'] * gsd] for n in node_list], dtype=np.float32)
    
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k+1)
    
    deviations = []
    for i in range(len(node_list)):
        nbrs = indices[i, 1:]
        med_dx = np.median(disps_m[nbrs, 0])
        med_dy = np.median(disps_m[nbrs, 1])
        dev = np.hypot(disps_m[i, 0] - med_dx, disps_m[i, 1] - med_dy)
        deviations.append(float(dev))
        
    deviations = np.array(deviations)
    med_dev = float(np.median(deviations))
    mad = float(np.median(np.abs(deviations - med_dev)))
    thresh_3mad = 3.0 * mad
    
    filtered_nodes = [node_list[i] for i in range(len(node_list)) if deviations[i] <= thresh_3mad]
    return filtered_nodes, deviations, med_dev, thresh_3mad

print("\n--- L1: VECTOR FIELD MEDIAN FILTER ON GENUINE ---")
l1_gen_nodes, gen_devs, gen_med_dev, gen_th_3mad = apply_vector_median_filter(gen_acc, k=8)
stats_before_l1 = eval_geometric_fit(gen_acc, 'Before L1')
stats_after_l1 = eval_geometric_fit(l1_gen_nodes, 'After L1')

print(f"Deviations: min={gen_devs.min():.2f}m, max={gen_devs.max():.2f}m, median={gen_med_dev:.2f}m, mean={gen_devs.mean():.2f}m")
print(f"MAD: {np.median(np.abs(gen_devs - gen_med_dev)):.2f} m, 3x MAD Threshold: {gen_th_3mad:.2f} m")
print(f"Node count before: {stats_before_l1['count']}, after L1: {stats_after_l1['count']} ({stats_after_l1['count']/stats_before_l1['count']*100:.1f}%)")
print(f"RMSE before: {stats_before_l1['rmse_m']:.2f}m ({stats_before_l1['rmse_ref_px']:.2f} ref px), after: {stats_after_l1['rmse_m']:.2f}m ({stats_after_l1['rmse_ref_px']:.2f} ref px)")
print(f"Median residual before: {stats_before_l1['median_m']:.2f}m, after: {stats_after_l1['median_m']:.2f}m")
print(f"p95 residual before: {stats_before_l1['p95_m']:.2f}m, after: {stats_after_l1['p95_m']:.2f}m")
print(f"Occupancy 8x8 before: {stats_before_l1['occupancy_8x8']}, after: {stats_after_l1['occupancy_8x8']}")
print(f"CV before: {stats_before_l1['cv']:.2f}, after: {stats_after_l1['cv']:.2f}")

# Histogram / bimodal check
hist_counts, bin_edges = np.histogram(gen_devs, bins=10)
print(f"Deviation Histogram Counts: {hist_counts.tolist()}")
print(f"Deviation Bin Edges: {np.round(bin_edges, 1).tolist()}")

# L2: RANSAC ON DISPLACEMENT FIELD
print("\n--- L2: RANSAC FIT & THRESHOLD SWEEP ON GENUINE ---")
def run_ransac_fit(node_list, th_m):
    if len(node_list) < 4:
        return {'inliers': [], 'count': 0, 'ratio_pct': 0.0, 'rmse_m': None, 'scale': None, 'rot_deg': None, 'tx_m': None, 'ty_m': None, 'cond_num': None}
    src_pts = np.array([[n['grid_x'] + n['dx_sub'], n['grid_y'] + n['dy_sub']] for n in node_list], dtype=np.float32)
    ref_pts = np.array([[n['grid_x'], n['grid_y']] for n in node_list], dtype=np.float32)
    
    th_px = th_m / gsd
    M_sim, inliers = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=th_px, maxIters=10000, confidence=0.999)
    if inliers is None:
        return {'inliers': [], 'count': 0, 'ratio_pct': 0.0, 'rmse_m': None, 'scale': None, 'rot_deg': None, 'tx_m': None, 'ty_m': None, 'cond_num': None}
    
    inl_mask = (inliers.ravel() == 1)
    inl_nodes = [node_list[i] for i in range(len(node_list)) if inl_mask[i]]
    n_inl = len(inl_nodes)
    inl_ratio = n_inl / len(node_list) * 100.0
    
    if n_inl >= 2:
        pred_ref = (M_sim[:, :2] @ src_pts[inl_mask].T + M_sim[:, 2:3]).T
        res_m = np.linalg.norm(ref_pts[inl_mask] - pred_ref, axis=1) * gsd
        rmse_m = float(np.sqrt(np.mean(res_m**2)))
        scale = float(np.sqrt(M_sim[0, 0]**2 + M_sim[1, 0]**2))
        rot_deg = float(np.degrees(np.arctan2(M_sim[1, 0], M_sim[0, 0])))
        tx_m = float(M_sim[0, 2] * gsd)
        ty_m = float(M_sim[1, 2] * gsd)
        U, S, Vt = np.linalg.svd(M_sim[:, :2])
        cond_num = float(S[0] / S[-1]) if S[-1] > 1e-6 else float('inf')
        return {
            'inliers': inl_nodes, 'count': n_inl, 'ratio_pct': inl_ratio,
            'rmse_m': rmse_m, 'rmse_ref_px': rmse_m / 2.00,
            'scale': scale, 'rot_deg': rot_deg, 'tx_m': tx_m, 'ty_m': ty_m,
            'cond_num': cond_num, 'M_sim': M_sim.tolist()
        }
    else:
        return {'inliers': [], 'count': n_inl, 'ratio_pct': inl_ratio, 'rmse_m': None, 'scale': None, 'rot_deg': None, 'tx_m': None, 'ty_m': None, 'cond_num': None}

ransac_sweeps = [50.0, 25.0, 10.0, 5.0, 2.0, 1.0]
gen_ransac_results = {}
for th in ransac_sweeps:
    res = run_ransac_fit(gen_acc, th)
    gen_ransac_results[f"{th:.0f}m"] = res
    print(f"Th={th:4.1f}m: Inliers={res['count']:3d} ({res['ratio_pct']:5.2f}%), RMSE={res['rmse_m'] if res['rmse_m'] else 0.0:5.2f}m, Scale={res['scale'] if res['scale'] else 0.0:.5f}, Rot={res['rot_deg'] if res['rot_deg'] else 0.0:6.3f} deg, Tx={res['tx_m'] if res['tx_m'] else 0.0:6.1f}m, Ty={res['ty_m'] if res['ty_m'] else 0.0:6.1f}m, Cond={res['cond_num'] if res['cond_num'] else 0.0:.4f}")

# Also test RANSAC on L1-filtered nodes (compound pipeline L1 + L2)
gen_l1_l2_results = {}
print("\n--- Compound Pipeline L1 + L2 (RANSAC on L1-filtered nodes) ---")
for th in ransac_sweeps:
    res = run_ransac_fit(l1_gen_nodes, th)
    gen_l1_l2_results[f"{th:.0f}m"] = res
    print(f"L1 + RANSAC({th:4.1f}m): Inliers={res['count']:3d} ({res['ratio_pct']:5.2f}%), RMSE={res['rmse_m'] if res['rmse_m'] else 0.0:5.2f}m, Scale={res['scale'] if res['scale'] else 0.0:.5f}, Rot={res['rot_deg'] if res['rot_deg'] else 0.0:6.3f} deg")

# Spatial distribution of 10m RANSAC inliers
inl_10m = gen_ransac_results['10m']['inliers']
print(f"\nSpatial distribution of 10m inliers (N={len(inl_10m)}):")
for n in inl_10m:
    print(f"  Node grid_x={n['grid_x']:4d}, grid_y={n['grid_y']:4d} | dx={n['dx_sub']*gsd:6.1f}m, dy={n['dy_sub']*gsd:6.1f}m | peak={n['peak']:.3f}")

# L4: EVALUATE ALL SEVEN CONTROLS THROUGH COMPLETE L1 + L2 PIPELINE
print("\n--- L4: RUNNING ALL 7 CONTROLS THROUGH L1 + L2 PIPELINE ---")
ctrl_names = ['rot90', 'rot180', 'rot270', 'vflip', 'hflip', 'offset', 'uniform_noise']
control_eval = {}

for c_name in ctrl_names:
    t_start = time.time()
    target_img = pc_tmc_u8.copy()
    if c_name == 'rot90':
        c_img = cv2.rotate(target_img, cv2.ROTATE_90_CLOCKWISE)
        c_mask = cv2.rotate(valid_mask.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE) > 0
    elif c_name == 'rot180':
        c_img = cv2.rotate(target_img, cv2.ROTATE_180)
        c_mask = cv2.rotate(valid_mask.astype(np.uint8), cv2.ROTATE_180) > 0
    elif c_name == 'rot270':
        c_img = cv2.rotate(target_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        c_mask = cv2.rotate(valid_mask.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE) > 0
    elif c_name == 'vflip':
        c_img = cv2.flip(target_img, 0)
        c_mask = cv2.flip(valid_mask.astype(np.uint8), 0) > 0
    elif c_name == 'hflip':
        c_img = cv2.flip(target_img, 1)
        c_mask = cv2.flip(valid_mask.astype(np.uint8), 1) > 0
    elif c_name == 'offset':
        c_img = np.roll(target_img, 500, axis=0)
        c_mask = np.roll(valid_mask, 500, axis=0)
    elif c_name == 'uniform_noise':
        np.random.seed(42)
        valid_vals = target_img[valid_mask]
        c_img = np.random.uniform(valid_vals.min(), valid_vals.max(), target_img.shape).astype(target_img.dtype)
        c_img[~valid_mask] = 0
        c_mask = valid_mask.copy()
        
    # Match size if rotated
    if c_img.shape != pc_ortho_u8.shape:
        pad_img = np.zeros_like(pc_ortho_u8)
        pad_m = np.zeros_like(valid_mask)
        mh = min(pc_ortho_u8.shape[0], c_img.shape[0])
        mw = min(pc_ortho_u8.shape[1], c_img.shape[1])
        pad_img[:mh, :mw] = c_img[:mh, :mw]
        pad_m[:mh, :mw] = c_mask[:mh, :mw]
        c_img = pad_img
        c_mask = pad_m
        
    c_acc, c_tot, c_rejs, c_mp, c_pks = run_phase_correlation(pc_ortho_u8, c_img, c_mask)
    
    # 1. Apply L1 Vector Median Filter
    c_l1_nodes, c_devs, c_med_dev, c_th = apply_vector_median_filter(c_acc, k=8)
    
    # 2. Apply L2 RANSAC sweeps
    c_ransac_sweeps = {}
    for th in ransac_sweeps:
        c_res = run_ransac_fit(c_acc, th)
        c_ransac_sweeps[f"{th:.0f}m"] = {
            'count': c_res['count'], 'ratio_pct': c_res['ratio_pct'],
            'rmse_m': c_res['rmse_m'], 'scale': c_res['scale'],
            'rot_deg': c_res['rot_deg'], 'tx_m': c_res['tx_m'], 'ty_m': c_res['ty_m'],
            'inliers': c_res['inliers']
        }
        
    # 3. Apply Compound L1 + L2 (RANSAC on L1 nodes)
    c_l1_l2_sweeps = {}
    for th in ransac_sweeps:
        c_l1_res = run_ransac_fit(c_l1_nodes, th)
        c_l1_l2_sweeps[f"{th:.0f}m"] = {
            'count': c_l1_res['count'], 'ratio_pct': c_l1_res['ratio_pct'],
            'rmse_m': c_l1_res['rmse_m'], 'scale': c_l1_res['scale'],
            'rot_deg': c_l1_res['rot_deg'], 'inliers': c_l1_res['inliers']
        }
        
    # KS test on post-filter peak distributions (L1 and L2 at 10m)
    pks_gen_l1 = np.array([n['peak'] for n in l1_gen_nodes]) if len(l1_gen_nodes) > 0 else np.array([0.0])
    pks_c_l1 = np.array([n['peak'] for n in c_l1_nodes]) if len(c_l1_nodes) > 0 else np.array([0.0])
    ks_l1 = stats.ks_2samp(pks_gen_l1, pks_c_l1)
    
    pks_gen_l2_10m = np.array([n['peak'] for n in gen_ransac_results['10m']['inliers']]) if gen_ransac_results['10m']['count'] > 0 else np.array([0.0])
    pks_c_l2_10m = np.array([n['peak'] for n in c_ransac_sweeps['10m']['inliers']]) if len(c_ransac_sweeps['10m']['inliers']) > 0 else np.array([0.0])
    ks_l2_10m = stats.ks_2samp(pks_gen_l2_10m, pks_c_l2_10m)
    
    control_eval[c_name] = {
        'pre_filter_accepted': len(c_acc),
        'pre_filter_ratio_pct': len(c_acc) / c_tot * 100.0,
        'l1_accepted': len(c_l1_nodes),
        'l1_ratio_pct': len(c_l1_nodes) / c_tot * 100.0,
        'l1_mad_th': c_th,
        'ransac_sweeps': c_ransac_sweeps,
        'l1_l2_sweeps': c_l1_l2_sweeps,
        'ks_l1_D': float(ks_l1.statistic),
        'ks_l1_p': float(ks_l1.pvalue),
        'ks_l2_10m_D': float(ks_l2_10m.statistic),
        'ks_l2_10m_p': float(ks_l2_10m.pvalue)
    }
    
    print(f"{c_name:15s} | Pre: {len(c_acc):3d} ({len(c_acc)/c_tot*100:5.2f}%) | L1: {len(c_l1_nodes):3d} ({len(c_l1_nodes)/c_tot*100:5.2f}%) | RANSAC(10m): {c_ransac_sweeps['10m']['count']:2d} ({c_ransac_sweeps['10m']['ratio_pct']:5.2f}%) | RANSAC(5m): {c_ransac_sweeps['5m']['count']:2d} ({c_ransac_sweeps['5m']['ratio_pct']:5.2f}%) | time: {time.time()-t_start:.1f}s")

# Delta_shuffle across all filter stages
r_gen_pre = len(gen_acc) / gen_tot * 100.0
r_gen_l1 = len(l1_gen_nodes) / gen_tot * 100.0
r_gen_ransac_10m = gen_ransac_results['10m']['count'] / gen_tot * 100.0
r_gen_ransac_5m = gen_ransac_results['5m']['count'] / gen_tot * 100.0

max_ctrl_pre = max([c['pre_filter_ratio_pct'] for c in control_eval.values()])
max_ctrl_l1 = max([c['l1_ratio_pct'] for c in control_eval.values()])
max_ctrl_ransac_10m = max([c['ransac_sweeps']['10m']['count'] / 1120 * 100.0 for c in control_eval.values()])
max_ctrl_ransac_5m = max([c['ransac_sweeps']['5m']['count'] / 1120 * 100.0 for c in control_eval.values()])

delta_pre = r_gen_pre - max_ctrl_pre
delta_l1 = r_gen_l1 - max_ctrl_l1
delta_ransac_10m = r_gen_ransac_10m - max_ctrl_ransac_10m
delta_ransac_5m = r_gen_ransac_5m - max_ctrl_ransac_5m

print("\n--- DELTA_SHUFFLE SUMMARY ACROSS STAGES ---")
print(f"Pre-filter:     Genuine = {r_gen_pre:5.2f}%, Max Ctrl = {max_ctrl_pre:5.2f}%, Delta_shuffle = {delta_pre:+5.2f}%")
print(f"Post-L1:        Genuine = {r_gen_l1:5.2f}%, Max Ctrl = {max_ctrl_l1:5.2f}%, Delta_shuffle = {delta_l1:+5.2f}%")
print(f"Post-L2 (10m):  Genuine = {r_gen_ransac_10m:5.2f}%, Max Ctrl = {max_ctrl_ransac_10m:5.2f}%, Delta_shuffle = {delta_ransac_10m:+5.2f}%")
print(f"Post-L2 (5m):   Genuine = {r_gen_ransac_5m:5.2f}%, Max Ctrl = {max_ctrl_ransac_5m:5.2f}%, Delta_shuffle = {delta_ransac_5m:+5.2f}%")

# Save complete results
out_stage_l = {
    'l1_vector_median': {
        'k': 8,
        'mad_m': float(np.median(np.abs(gen_devs - gen_med_dev))),
        'threshold_3mad_m': gen_th_3mad,
        'stats_before': stats_before_l1,
        'stats_after': stats_after_l1,
        'deviation_percentiles_m': {
            'min': float(gen_devs.min()), 'p10': float(np.percentile(gen_devs, 10)),
            'p25': float(np.percentile(gen_devs, 25)), 'p50': float(np.median(gen_devs)),
            'p75': float(np.percentile(gen_devs, 75)), 'p90': float(np.percentile(gen_devs, 90)),
            'p95': float(np.percentile(gen_devs, 95)), 'max': float(gen_devs.max())
        },
        'histogram': {'counts': hist_counts.tolist(), 'bin_edges': bin_edges.tolist()}
    },
    'l2_ransac': {
        'threshold_sweeps': gen_ransac_results,
        'compound_l1_l2_sweeps': gen_l1_l2_results
    },
    'l4_controls': control_eval,
    'delta_shuffle': {
        'pre_filter': delta_pre,
        'post_l1': delta_l1,
        'post_l2_10m': delta_ransac_10m,
        'post_l2_5m': delta_ransac_5m
    }
}

os.makedirs('assets/real_cache', exist_ok=True)
with open('assets/real_cache/stage_l_geometric_filter_results.json', 'w') as f:
    json.dump(out_stage_l, f, indent=2)
print("Saved assets/real_cache/stage_l_geometric_filter_results.json")

