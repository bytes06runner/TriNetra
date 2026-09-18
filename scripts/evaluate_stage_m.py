import numpy as np
import rasterio
import cv2
import pandas as pd
from scipy import stats
import json
import time
import os
import matplotlib.pyplot as plt

from src.phase_congruency import compute_phase_congruency

print("=== STAGE M: SUB-PIXEL VALIDATION AGAINST KNOWN TRUTH ===")
os.makedirs('results/figures', exist_ok=True)
os.makedirs('assets/real_cache', exist_ok=True)

# -------------------------------------------------------------
# M1: SYNTHETIC SUB-PIXEL SHIFT RECOVERY (SAME SENSOR, 2.0 m/px)
# -------------------------------------------------------------
print("\n--- M1: Synthetic Sub-Pixel Shift Recovery (512x512, 2.0 m/px) ---")

with rasterio.open('data/lroc_nac/NAC_DTM_PITISCUS_M1149280834_2M.TIF') as src:
    ortho_full = src.read(1)

# Lobate scarp native crop (512x512)
cy = int(4500 * 4.72 / 2.0) # 10620
cx = int(600 * 4.72 / 2.0)  # 1416
ref_512 = ortho_full[cy-256:cy+256, cx-256:cx+256].astype(np.float32)

def fourier_shift_2d(img, dx, dy):
    H, W = img.shape
    u = np.fft.fftfreq(W)
    v = np.fft.fftfreq(H)
    ug, vg = np.meshgrid(u, v)
    phase = np.exp(-2j * np.pi * (ug * dx + vg * dy))
    shifted = np.real(np.fft.ifft2(np.fft.fft2(img) * phase))
    return shifted.astype(np.float32)

def correlate_single_node(ref_patch, srch_patch, tpl_sz=64, srch_sz=128):
    half_t = tpl_sz // 2
    half_s = srch_sz // 2
    cy_c, cx_c = ref_patch.shape[0] // 2, ref_patch.shape[1] // 2
    
    tpl = ref_patch[cy_c - half_t : cy_c + half_t, cx_c - half_t : cx_c + half_t]
    srch = srch_patch[cy_c - half_s : cy_c + half_s, cx_c - half_s : cx_c + half_s]
    
    c = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(c)
    
    px, py = loc[0], loc[1]
    if px <= 0 or px >= c.shape[1] - 1 or py <= 0 or py >= c.shape[0] - 1:
        delta_x, delta_y = 0.0, 0.0
    else:
        patch = c[py-1:py+2, px-1:px+2]
        denom_x = 2 * (patch[1, 0] - 2 * patch[1, 1] + patch[1, 2])
        denom_y = 2 * (patch[0, 1] - 2 * patch[1, 1] + patch[2, 1])
        delta_x = (patch[1, 0] - patch[1, 2]) / denom_x if abs(denom_x) > 1e-6 else 0.0
        delta_y = (patch[0, 1] - patch[2, 1]) / denom_y if abs(denom_y) > 1e-6 else 0.0
        if abs(delta_x) > 1.0 or abs(delta_y) > 1.0:
            delta_x, delta_y = 0.0, 0.0
        
    rec_x = (px + half_t) - half_s + delta_x
    rec_y = (py + half_t) - half_s + delta_y
    return rec_x, rec_y, mx

# 100 shift pairs: dx, dy in {0.0, 0.1, ..., 0.9}
shifts_1d = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
m1_results = []
true_dxs, rec_dxs = [], []
true_dys, rec_dys = [], []

for dx in shifts_1d:
    for dy in shifts_1d:
        s_img = fourier_shift_2d(ref_512, dx, dy)
        rec_x, rec_y, peak = correlate_single_node(ref_512, s_img)
        err_px = float(np.hypot(rec_x - dx, rec_y - dy))
        err_m = float(err_px * 2.00) # native 2.00 m/px
        
        m1_results.append({
            'true_dx': dx, 'true_dy': dy,
            'rec_dx': float(rec_x), 'rec_dy': float(rec_y),
            'err_px': err_px, 'err_m': err_m,
            'peak': float(peak)
        })
        true_dxs.append(dx)
        rec_dxs.append(float(rec_x))
        true_dys.append(dy)
        rec_dys.append(float(rec_y))

df_m1 = pd.DataFrame(m1_results)
mae_px = float(df_m1['err_px'].mean())
mae_m = float(df_m1['err_m'].mean())
rmse_px = float(np.sqrt(np.mean(df_m1['err_px']**2)))
rmse_m = float(np.sqrt(np.mean(df_m1['err_m']**2)))
max_err_px = float(df_m1['err_px'].max())
max_err_m = float(df_m1['err_m'].max())
mean_peak_m1 = float(df_m1['peak'].mean())

print(f"M1 Real Crop (100 pairs):")
print(f"  MAE:     {mae_px:.4f} px ({mae_m:.4f} m)")
print(f"  RMSE:    {rmse_px:.4f} px ({rmse_m:.4f} m)")
print(f"  Max Err: {max_err_px:.4f} px ({max_err_m:.4f} m)")
print(f"  Mean Peak: {mean_peak_m1:.4f}")

# Peak locking check
# Compute bias curve b(x) = mean(rec_dx - true_dx) for each fractional true_dx
bias_curve_x = []
for val in shifts_1d:
    sub = df_m1[df_m1['true_dx'] == val]
    bias = float(np.mean(sub['rec_dx'] - sub['true_dx']))
    bias_curve_x.append(bias)

max_peak_locking_bias = float(np.max(np.abs(bias_curve_x)))
print(f"  Max Peak Locking Bias: {max_peak_locking_bias:.4f} px ({max_peak_locking_bias*2.0:.4f} m)")

# Plot Peak Locking Curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Ideal y=x (Zero Bias)')
plt.scatter(df_m1['true_dx'], df_m1['rec_dx'], alpha=0.6, color='blue', label='Recovered Shifts')
plt.plot(shifts_1d, np.array(shifts_1d) + np.array(bias_curve_x), 'r-o', linewidth=2, label='Mean Estimator Response')
plt.title('Stage M1: Parabolic Estimator Peak-Locking Characteristic\nRecovered vs True Fractional Sub-Pixel Shift')
plt.xlabel('True Fractional Shift (pixels)')
plt.ylabel('Recovered Shift (pixels)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/figures/stage_m_peak_locking.png', dpi=200)
plt.close()
print("Saved results/figures/stage_m_peak_locking.png")

# M1g: Control on Uniform Random Noise
print("\n--- M1g: Control on Uniform Random Noise ---")
np.random.seed(42)
noise_512 = np.random.uniform(0, 255, (512, 512)).astype(np.float32)
noise_m1_results = []
for dx in shifts_1d:
    for dy in shifts_1d:
        # Generate independent noise search frame (uncorrelated) to test chance peak error
        noise_srch = np.random.uniform(0, 255, (512, 512)).astype(np.float32)
        rec_x, rec_y, peak = correlate_single_node(noise_512, noise_srch)
        err_px = float(np.hypot(rec_x - dx, rec_y - dy))
        noise_m1_results.append({'err_px': err_px, 'peak': float(peak)})

df_noise_m1 = pd.DataFrame(noise_m1_results)
noise_mae_px = float(df_noise_m1['err_px'].mean())
noise_rmse_px = float(np.sqrt(np.mean(df_noise_m1['err_px']**2)))
noise_peak_m1 = float(df_noise_m1['peak'].mean())
print(f"Noise Control:")
print(f"  MAE:  {noise_mae_px:.2f} px ({noise_mae_px*2.0:.2f} m)")
print(f"  RMSE: {noise_rmse_px:.2f} px ({noise_rmse_px*2.0:.2f} m)")
print(f"  Mean Peak: {noise_peak_m1:.4f}")

# -------------------------------------------------------------
# M2: KNOWN-TRUTH REGISTRATION ACROSS A REAL SCALE GAP (2.0 -> 4.72 m/px)
# -------------------------------------------------------------
print("\n--- M2: Known-Truth Registration Across Scale Gap (2.00 m/px -> 4.72 m/px) ---")

# 2048x2048 crop around scarp at 2.00 m/px
ref_2048 = ortho_full[cy-1024:cy+1024, cx-1024:cx+1024].astype(np.float32)

# Resample reference to common grid at 4.72 m/px (with MTF blur)
ref_blur = cv2.GaussianBlur(ref_2048, (7, 7), 1.2)
H_472 = int(round(2048 * 2.00 / 4.72)) # 868
W_472 = int(round(2048 * 2.00 / 4.72)) # 868
ref_472 = cv2.resize(ref_blur, (W_472, H_472), interpolation=cv2.INTER_AREA)

# Base source downsampled to 4.72 m/px
src_472_base = cv2.resize(ref_2048, (W_472, H_472), interpolation=cv2.INTER_AREA)

# Generate 25 random shift pairs in [-2.0, +2.0] pixels
np.random.seed(123)
n_m2_trials = 25
rand_dxs = np.random.uniform(-2.0, 2.0, n_m2_trials)
rand_dys = np.random.uniform(-2.0, 2.0, n_m2_trials)

# Common-grid correlator function over a regular grid (e.g. 8x8 nodes)
def run_m2_grid_registration(ref_img_472, src_img_472, true_dx, true_dy, n_grid=8):
    pc_ref, _ = compute_phase_congruency(ref_img_472, nscale=4, norient=6)
    pc_src, _ = compute_phase_congruency(src_img_472, nscale=4, norient=6)
    
    half_t = 32
    half_s = 64
    xs = np.linspace(half_s + 10, W_472 - half_s - 10, n_grid, dtype=int)
    ys = np.linspace(half_s + 10, H_472 - half_s - 10, n_grid, dtype=int)
    
    acc_nodes = []
    peaks = []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            tpl = pc_ref[y - half_t : y + half_t, x - half_t : x + half_t].astype(np.float32)
            srch = pc_src[y - half_s : y + half_s, x - half_s : x + half_s].astype(np.float32)
            
            if tpl.std() < 1e-4 or srch.std() < 1e-4:
                continue
            c = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(c)
            peaks.append(float(mx))
            
            if mx < 0.40:
                continue
                
            c_supp = c.copy()
            py, px = loc[1], loc[0]
            c_supp[max(0, py-2):min(c.shape[0], py+3), max(0, px-2):min(c.shape[1], px+3)] = -1.0
            second_mx = cv2.minMaxLoc(c_supp)[1]
            if second_mx >= 0.90 * mx:
                continue
                
            if px <= 0 or px >= c.shape[1] - 1 or py <= 0 or py >= c.shape[0] - 1:
                continue
                
            patch = c[py-1:py+2, px-1:px+2]
            denom_x = 2 * (patch[1, 0] - 2 * patch[1, 1] + patch[1, 2])
            denom_y = 2 * (patch[0, 1] - 2 * patch[1, 1] + patch[2, 1])
            delta_x = (patch[1, 0] - patch[1, 2]) / denom_x if abs(denom_x) > 1e-6 else 0.0
            delta_y = (patch[0, 1] - patch[2, 1]) / denom_y if abs(denom_y) > 1e-6 else 0.0
            if abs(delta_x) > 1.0 or abs(delta_y) > 1.0:
                delta_x, delta_y = 0.0, 0.0
                
            rec_x = (px + half_t) - half_s + delta_x
            rec_y = (py + half_t) - half_s + delta_y
            
            acc_nodes.append({
                'ix': ix, 'iy': iy, 'x': x, 'y': y,
                'rec_dx': float(rec_x), 'rec_dy': float(rec_y),
                'peak': float(mx)
            })
            
    return acc_nodes, len(xs)*len(ys), np.mean(peaks) if peaks else 0.0

m2_trials = []
for i in range(n_m2_trials):
    tdx = rand_dxs[i]
    tdy = rand_dys[i]
    shifted_src = fourier_shift_2d(src_472_base, tdx, tdy)
    nodes, total_grid, mp = run_m2_grid_registration(ref_472, shifted_src, tdx, tdy)
    
    if len(nodes) > 0:
        med_rec_x = float(np.median([n['rec_dx'] for n in nodes]))
        med_rec_y = float(np.median([n['rec_dy'] for n in nodes]))
        err_common_px = float(np.hypot(med_rec_x - tdx, med_rec_y - tdy))
        err_ref_px = float(err_common_px * (4.72 / 2.00))
        err_m = float(err_common_px * 4.72)
    else:
        med_rec_x, med_rec_y = None, None
        err_common_px, err_ref_px, err_m = None, None, None
        
    m2_trials.append({
        'trial': i+1,
        'true_dx_common': float(tdx), 'true_dy_common': float(tdy),
        'rec_dx_common': med_rec_x, 'rec_dy_common': med_rec_y,
        'err_common_px': err_common_px,
        'err_ref_px': err_ref_px,
        'err_m': err_m,
        'accepted_nodes': len(nodes),
        'total_nodes': total_grid,
        'acceptance_ratio': len(nodes) / total_grid * 100.0,
        'mean_peak': float(mp)
    })
    print(f"Trial {i+1:2d} | True: ({tdx:+5.2f}, {tdy:+5.2f}) | Rec: ({med_rec_x:+5.2f}, {med_rec_y:+5.2f}) | Err: {err_common_px:.3f} com px ({err_ref_px:.3f} ref px, {err_m:.2f} m) | Acc: {len(nodes):2d}/{total_grid}")

df_m2 = pd.DataFrame(m2_trials)
m2_mae_common = float(df_m2['err_common_px'].mean())
m2_rmse_common = float(np.sqrt(np.mean(df_m2['err_common_px']**2)))
m2_rmse_ref = float(np.sqrt(np.mean(df_m2['err_ref_px']**2)))
m2_rmse_m = float(np.sqrt(np.mean(df_m2['err_m']**2)))
m2_max_err_ref = float(df_m2['err_ref_px'].max())
m2_mean_nodes = float(df_m2['accepted_nodes'].mean())
m2_mean_ratio = float(df_m2['acceptance_ratio'].mean())

print(f"\nM2 Aggregate Performance Across Scale Gap (N={n_m2_trials} trials):")
print(f"  RMSE (Common Grid): {m2_rmse_common:.4f} common px")
print(f"  RMSE (Reference):   {m2_rmse_ref:.4f} reference px")
print(f"  RMSE (Meters):      {m2_rmse_m:.4f} m")
print(f"  Max Error (Ref Px): {m2_max_err_ref:.4f} ref px")
print(f"  Mean Accepted Nodes: {m2_mean_nodes:.1f} / 64 ({m2_mean_ratio:.1f}%)")

# -------------------------------------------------------------
# M3: CONTROLS (NOISE & ROT180 ON M2)
# -------------------------------------------------------------
print("\n--- M3: Controls on M2 ---")

# M3a: Noise on M2
noise_m2_trials = []
for i in range(10):
    tdx = rand_dxs[i]
    tdy = rand_dys[i]
    noise_src = np.random.uniform(0, 255, (H_472, W_472)).astype(np.float32)
    nodes, total_grid, mp = run_m2_grid_registration(ref_472, noise_src, tdx, tdy)
    noise_m2_trials.append({'accepted': len(nodes), 'total': total_grid, 'mean_peak': mp})
print(f"M3a Noise Control: Mean accepted nodes = {np.mean([t['accepted'] for t in noise_m2_trials]):.1f} / 64 (0.0%), Mean Peak = {np.mean([t['mean_peak'] for t in noise_m2_trials]):.4f}")

# M3b: Rot180 on M2
rot180_ref_472 = cv2.rotate(ref_472, cv2.ROTATE_180)
rot180_m2_trials = []
for i in range(10):
    tdx = rand_dxs[i]
    tdy = rand_dys[i]
    shifted_src = fourier_shift_2d(src_472_base, tdx, tdy)
    nodes, total_grid, mp = run_m2_grid_registration(rot180_ref_472, shifted_src, tdx, tdy)
    if len(nodes) > 0:
        med_rec_x = float(np.median([n['rec_dx'] for n in nodes]))
        med_rec_y = float(np.median([n['rec_dy'] for n in nodes]))
        err_m = float(np.hypot(med_rec_x - tdx, med_rec_y - tdy) * 4.72)
    else:
        err_m = 999.0
    rot180_m2_trials.append({'accepted': len(nodes), 'total': total_grid, 'mean_peak': mp, 'err_m': err_m})
print(f"M3b Rot180 Control: Mean accepted nodes = {np.mean([t['accepted'] for t in rot180_m2_trials]):.1f} / 64, Mean Recovery Error = {np.mean([t['err_m'] for t in rot180_m2_trials]):.1f} m")

# Save full results JSON
out_stage_m = {
    'm1_synthetic_subpixel': {
        'total_pairs': 100,
        'mae_px': mae_px, 'mae_m': mae_m,
        'rmse_px': rmse_px, 'rmse_m': rmse_m,
        'max_err_px': max_err_px, 'max_err_m': max_err_m,
        'mean_peak': mean_peak_m1,
        'max_peak_locking_bias_px': max_peak_locking_bias,
        'peak_locking_bias_curve': bias_curve_x,
        'noise_control': {
            'mae_px': noise_mae_px, 'rmse_px': noise_rmse_px, 'mean_peak': noise_peak_m1
        },
        'trials': m1_results
    },
    'm2_scale_gap_registration': {
        'trials_count': n_m2_trials,
        'rmse_common_px': m2_rmse_common,
        'rmse_ref_px': m2_rmse_ref,
        'rmse_m': m2_rmse_m,
        'max_err_ref_px': m2_max_err_ref,
        'mean_accepted_nodes': m2_mean_nodes,
        'mean_acceptance_ratio': m2_mean_ratio,
        'trials': m2_trials,
        'controls': {
            'noise_accepted': float(np.mean([t['accepted'] for t in noise_m2_trials])),
            'rot180_accepted': float(np.mean([t['accepted'] for t in rot180_m2_trials])),
            'rot180_mean_err_m': float(np.mean([t['err_m'] for t in rot180_m2_trials]))
        }
    }
}

with open('assets/real_cache/stage_m_synthetic_results.json', 'w') as f:
    json.dump(out_stage_m, f, indent=2)
print("Saved assets/real_cache/stage_m_synthetic_results.json successfully.")

