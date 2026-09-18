import numpy as np
import rasterio
import cv2
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy import stats
import json
import time
import sys

from src.phase_congruency import compute_phase_congruency

print("=== STAGE K EVALUATION SCRIPT ===")
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

# Valid mask
valid_mask = (ortho_reproj > 0) & (tmc_shifted > 0)

print("Step 2: Computing Phase Congruency (K2a)...")
t0 = time.time()
pc_ortho_u8, pc_ortho_f = compute_phase_congruency(ortho_blur, nscale=4, norient=6)
pc_tmc_u8, pc_tmc_f = compute_phase_congruency(tmc_shifted, nscale=4, norient=6)
print(f"Phase congruency completed in {time.time() - t0:.1f}s")

print("Step 3: Computing Gradient Orientation (K2e)...")
def get_gradient_orientation(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0.8)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mask = mag > 1e-3
    ux = np.zeros_like(gx)
    uy = np.zeros_like(gy)
    ux[mask] = gx[mask] / mag[mask]
    uy[mask] = gy[mask] / mag[mask]
    return ux, uy

ux_o, uy_o = get_gradient_orientation(ortho_blur)
ux_t, uy_t = get_gradient_orientation(tmc_shifted)

# Define Correlator Function
def correlate_grid(ref_img, src_img, valid_m, grid_type='rect', n_x=16, n_y=70, srch_sz=128, tpl_sz=64, peak_thresh=0.40, is_vec=False, ref_y=None, src_y=None):
    half_t = tpl_sz // 2
    half_s = srch_sz // 2
    radius = half_s - half_t
    
    H, W = ref_img.shape[:2]
    
    # Generate node coordinates
    if grid_type == 'rect':
        xs = np.linspace(half_s + 10, W - half_s - 10, n_x, dtype=int)
        ys = np.linspace(half_s + 10, H - half_s - 10, n_y, dtype=int)
        nodes = [(x, y, ix, iy) for iy, y in enumerate(ys) for ix, x in enumerate(xs)]
    elif grid_type == 'conforming':
        # Erode valid mask by search radius to get safe interior bounds
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
        # Check nodata in valid mask
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
            
        if not is_vec:
            tpl = ref_img[y - half_t : y + half_t, x - half_t : x + half_t].astype(np.float32)
            srch = src_img[y - half_s : y + half_s, x - half_s : x + half_s].astype(np.float32)
            
            if tpl.std() < 1e-4 or srch.std() < 1e-4:
                rej_nd += 1
                all_peaks.append(0.0)
                continue
                
            c = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
        else:
            tpl_x = ref_img[y - half_t : y + half_t, x - half_t : x + half_t]
            tpl_y = ref_y[y - half_t : y + half_t, x - half_t : x + half_t]
            srch_x = src_img[y - half_s : y + half_s, x - half_s : x + half_s]
            srch_y = src_y[y - half_s : y + half_s, x - half_s : x + half_s]
            
            cx = cv2.matchTemplate(srch_x, tpl_x, cv2.TM_CCORR)
            cy = cv2.matchTemplate(srch_y, tpl_y, cv2.TM_CCORR)
            c = (cx + cy) / float(tpl_sz * tpl_sz)
            
        _, mx, _, loc = cv2.minMaxLoc(c)
        all_peaks.append(float(mx))
        
        if mx < peak_thresh:
            rej_p += 1
            continue
            
        # Uniqueness test: secondary peak within 90%
        c_supp = c.copy()
        py, px = loc[1], loc[0]
        c_supp[max(0, py-2):min(c.shape[0], py+3), max(0, px-2):min(c.shape[1], px+3)] = -1.0
        second_mx = cv2.minMaxLoc(c_supp)[1]
        if second_mx >= 0.90 * mx:
            rej_u += 1
            continue
            
        # Parabolic sub-pixel fit
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

print("\n--- STEP 4: BENCHMARKING GRID TYPES & ARMS ---")

# Evaluate K1 on 32x32 original grid
acc_k1_32, tot_k1_32, rej_k1_32, mp_k1_32, pks_k1_32 = correlate_grid(ortho_blur, tmc_shifted, valid_mask, grid_type='rect', n_x=32, n_y=32)
print(f"K1 Raw NCC (32x32 rect): accepted={len(acc_k1_32)}/{tot_k1_32} ({len(acc_k1_32)/tot_k1_32*100:.2f}%), rejs(peak, uniq, rad, nodata)={rej_k1_32}, mean_peak={mp_k1_32:.4f}")

# Evaluate K3 on 16x70 rect vs conforming
acc_k1_rect, tot_k1_rect, rej_k1_rect, mp_k1_rect, pks_k1_rect = correlate_grid(ortho_blur, tmc_shifted, valid_mask, grid_type='rect', n_x=16, n_y=70)
print(f"K3 Raw NCC (16x70 rect): accepted={len(acc_k1_rect)}/{tot_k1_rect} ({len(acc_k1_rect)/tot_k1_rect*100:.2f}%), rejs(peak, uniq, rad, nodata)={rej_k1_rect}, mean_peak={mp_k1_rect:.4f}")

acc_k1_conf, tot_k1_conf, rej_k1_conf, mp_k1_conf, pks_k1_conf = correlate_grid(ortho_blur, tmc_shifted, valid_mask, grid_type='conforming', n_x=16, n_y=70)
print(f"K3 Raw NCC (16x70 conforming): accepted={len(acc_k1_conf)}/{tot_k1_conf} ({len(acc_k1_conf)/tot_k1_conf*100:.2f}%), rejs(peak, uniq, rad, nodata)={rej_k1_conf}, mean_peak={mp_k1_conf:.4f}")

# Evaluate K2 Arms on Conforming Grid (near-zero nodata waste)
print("\n--- EVALUATING THREE ARMS ON CONFORMING 16x70 GRID ---")
# Arm 1: Raw NCC
print(f"Arm 1 (Raw NCC): accepted={len(acc_k1_conf)}/{tot_k1_conf} ({len(acc_k1_conf)/tot_k1_conf*100:.2f}%), rejs={rej_k1_conf}, mean_peak={mp_k1_conf:.4f}")

# Arm 2: Log-Gabor Phase Congruency
acc_pc, tot_pc, rej_pc, mp_pc, pks_pc = correlate_grid(pc_ortho_u8, pc_tmc_u8, valid_mask, grid_type='conforming', n_x=16, n_y=70)
print(f"Arm 2 (Phase Congruency): accepted={len(acc_pc)}/{tot_pc} ({len(acc_pc)/tot_pc*100:.2f}%), rejs={rej_pc}, mean_peak={mp_pc:.4f}")

# Arm 3: Gradient Orientation
acc_go, tot_go, rej_go, mp_go, pks_go = correlate_grid(ux_o, ux_t, valid_mask, grid_type='conforming', n_x=16, n_y=70, is_vec=True, ref_y=uy_o, src_y=uy_t)
print(f"Arm 3 (Gradient Orientation): accepted={len(acc_go)}/{tot_go} ({len(acc_go)/tot_go*100:.2f}%), rejs={rej_go}, mean_peak={mp_go:.4f}")

# Pick Best Arm and Run K4 Controls
best_arm_name = "phase_congruency" if len(acc_pc) >= max(len(acc_k1_conf), len(acc_go)) else ("raw_ncc" if len(acc_k1_conf) >= len(acc_go) else "gradient_orientation")
print(f"\nBest performing arm: {best_arm_name}")

print("\n--- STEP 5: RUNNING ALL 7 CONTROLS ON BEST ARM ---")
ctrl_names = ['rot90', 'rot180', 'rot270', 'vflip', 'hflip', 'offset', 'uniform_noise']
ctrl_results = {}

for c_name in ctrl_names:
    if best_arm_name == 'phase_congruency':
        target_img = pc_tmc_u8.copy()
    elif best_arm_name == 'raw_ncc':
        target_img = tmc_shifted.copy()
        
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
        
    ref_eval = pc_ortho_u8 if best_arm_name == 'phase_congruency' else ortho_blur
    # Match size if rotated
    if c_img.shape != ref_eval.shape:
        pad_img = np.zeros_like(ref_eval)
        pad_m = np.zeros_like(valid_mask)
        mh = min(ref_eval.shape[0], c_img.shape[0])
        mw = min(ref_eval.shape[1], c_img.shape[1])
        pad_img[:mh, :mw] = c_img[:mh, :mw]
        pad_m[:mh, :mw] = c_mask[:mh, :mw]
        c_img = pad_img
        c_mask = pad_m
        
    c_acc, c_tot, c_rejs, c_mp, c_pks = correlate_grid(
        ref_eval,
        c_img,
        c_mask,
        grid_type='conforming',
        n_x=16, n_y=70
    )
    
    # Kolmogorov-Smirnov test against genuine
    ks_res = stats.ks_2samp(pks_pc if best_arm_name == 'phase_congruency' else pks_k1_conf, c_pks)
    
    ctrl_results[c_name] = {
        'accepted': len(c_acc),
        'total': c_tot,
        'ratio_pct': len(c_acc) / c_tot * 100.0,
        'rejections': c_rejs,
        'mean_peak': c_mp,
        'ks_statistic': float(ks_res.statistic),
        'ks_pvalue': float(ks_res.pvalue)
    }
    print(f"{c_name:15s}: acc={len(c_acc):3d}/{c_tot:4d} ({len(c_acc)/c_tot*100:5.2f}%), mean_peak={c_mp:.4f}, KS D={ks_res.statistic:.4f}, p={ks_res.pvalue:.2e}")

# Delta_shuffle
best_acc = acc_pc if best_arm_name == 'phase_congruency' else (acc_k1_conf if best_arm_name == 'raw_ncc' else acc_go)
best_tot = tot_pc if best_arm_name == 'phase_congruency' else (tot_k1_conf if best_arm_name == 'raw_ncc' else tot_go)
r_genuine = len(best_acc) / best_tot * 100.0
max_ctrl_r = max([c['ratio_pct'] for c in ctrl_results.values()])
delta_shuffle = r_genuine - max_ctrl_r
print(f"\nGenuine ratio: {r_genuine:.2f}%, Max control ratio: {max_ctrl_r:.2f}%, Delta_shuffle: {delta_shuffle:.2f}%")

# K5 analysis if accepted nodes > 4
k5_metrics = {}
if len(best_acc) >= 4:
    # Estimate 4-DoF similarity: [x_ref, y_ref] -> [x_src, y_src] = [x_ref + dx, y_ref + dy]
    src_pts = np.array([[n['grid_x'] + n['dx_sub'], n['grid_y'] + n['dy_sub']] for n in best_acc], dtype=np.float32)
    ref_pts = np.array([[n['grid_x'], n['grid_y']] for n in best_acc], dtype=np.float32)
    
    M_sim, inliers = cv2.estimateAffinePartial2D(src_pts, ref_pts)
    if M_sim is not None:
        pred_ref = (M_sim[:, :2] @ src_pts.T + M_sim[:, 2:3]).T
        residuals_px = np.linalg.norm(ref_pts - pred_ref, axis=1) # in common grid px (4.72 m/px)
        residuals_ref_px = residuals_px * (4.72 / 2.00) # in native NAC reference px (2.00 m/px)
        residuals_m = residuals_px * 4.72
        
        rmse_m = float(np.sqrt(np.mean(residuals_m**2)))
        rmse_ref_px = float(np.sqrt(np.mean(residuals_ref_px**2)))
        median_m = float(np.median(residuals_m))
        p95_m = float(np.percentile(residuals_m, 95))
        
        k5_metrics['rmse_m'] = rmse_m
        k5_metrics['rmse_ref_px'] = rmse_ref_px
        k5_metrics['median_m'] = median_m
        k5_metrics['p95_m'] = p95_m
        k5_metrics['similarity_matrix'] = M_sim.tolist()
        print(f"K5 Sub-pixel fit: RMSE={rmse_m:.2f} m ({rmse_ref_px:.2f} ref px), median={median_m:.2f} m, p95={p95_m:.2f} m")

# Uniform distribution metrics: 8x8 grid occupancy & CV
if len(best_acc) > 0:
    grid_8x8 = np.zeros((8, 8), dtype=int)
    for n in best_acc:
        bin_x = min(7, int(n['grid_x'] / ortho_blur.shape[1] * 8))
        bin_y = min(7, int(n['grid_y'] / ortho_blur.shape[0] * 8))
        grid_8x8[bin_y, bin_x] += 1
    occupancy = int(np.count_nonzero(grid_8x8))
    occupancy_pct = occupancy / 64.0 * 100.0
    cv_val = float(grid_8x8.std() / grid_8x8.mean()) if grid_8x8.mean() > 0 else float('nan')
    k5_metrics['occupancy_8x8'] = f"{occupancy}/64 ({occupancy_pct:.1f}%)"
    k5_metrics['cv'] = cv_val
    print(f"Distribution: 8x8 occupancy = {occupancy}/64 ({occupancy_pct:.1f}%), CV = {cv_val:.2f}")

# Save results JSON
out_data = {
    'k1_raw_ncc_32x32': {
        'accepted': len(acc_k1_32), 'total': tot_k1_32, 'ratio_pct': len(acc_k1_32)/tot_k1_32*100.0,
        'rejections': rej_k1_32, 'mean_peak': mp_k1_32
    },
    'k3_raw_ncc_16x70_rect': {
        'accepted': len(acc_k1_rect), 'total': tot_k1_rect, 'ratio_pct': len(acc_k1_rect)/tot_k1_rect*100.0,
        'rejections': rej_k1_rect, 'mean_peak': mp_k1_rect
    },
    'k3_raw_ncc_16x70_conforming': {
        'accepted': len(acc_k1_conf), 'total': tot_k1_conf, 'ratio_pct': len(acc_k1_conf)/tot_k1_conf*100.0,
        'rejections': rej_k1_conf, 'mean_peak': mp_k1_conf
    },
    'k2_arms_comparison': {
        'raw_ncc': {
            'accepted': len(acc_k1_conf), 'total': tot_k1_conf, 'ratio_pct': len(acc_k1_conf)/tot_k1_conf*100.0,
            'rejections': rej_k1_conf, 'mean_peak': mp_k1_conf
        },
        'phase_congruency': {
            'accepted': len(acc_pc), 'total': tot_pc, 'ratio_pct': len(acc_pc)/tot_pc*100.0,
            'rejections': rej_pc, 'mean_peak': mp_pc
        },
        'gradient_orientation': {
            'accepted': len(acc_go), 'total': tot_go, 'ratio_pct': len(acc_go)/tot_go*100.0,
            'rejections': rej_go, 'mean_peak': mp_go
        }
    },
    'best_arm': best_arm_name,
    'delta_shuffle': delta_shuffle,
    'controls': ctrl_results,
    'k5_metrics': k5_metrics,
    'nodes': best_acc
}

with open('assets/real_cache/stage_k_dense_results.json', 'w') as f:
    json.dump(out_data, f, indent=2)
print("Saved assets/real_cache/stage_k_dense_results.json successfully.")
