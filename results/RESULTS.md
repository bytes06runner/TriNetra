# TriNetra (त्रिनेत्र) — Reviewer Executive Summary

This one-page summary summarizes the empirical registration results for Chandrayaan-2 planetary correspondence (Problem Statement SIH26166).

---

## 1. Measured Observation Datasets

All measurements were conducted on authentic Chandrayaan-2 PDS4 flight products downloaded from the ISSDC repository:
- **OHRC (0.26 m/px):** `ch2_ohr_ncp_20211023T0027462822_d_img_d18` (Lines 10000:14000, Samples 4000:8000; Center: -69.58019°S, 32.28800°E).
- **TMC-2 (4.72 m/px):** `ch2_tmc_ncn_20230130T1900132182_d_img_d32` (South Pole nadir track; Center: -69.57911°S, 32.27507°E and -70.85000°S, 32.26000°E).
- **IIRS (68.38 m/px):** `ch2_iir_nri_20231003T2152304115_d_img_d18` (Raw Level-1 hyperspectral cube, Lines 510:630, Samples 60:180; Center: -70.85000°S, 32.26000°E).

---

## 2. Transformation Conditioning & Plausibility

| Configuration | Inliers (RANSAC) | Inliers (≤15px) | Recovered Scale | Recovered Rotation | Condition Number | Validity Gate Status | First Failing Criterion |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hop 1: Baseline SIFT** | 5 / 409 | 5 / 409 | 0.9664 | -151.15° | 2352526.5 | 🛑 **GATED** | inlier_consensus |
| **Hop 1: Fine-Tuned LoFTR** | **49 / 217** | **47 / 217** | **1.0086** | **-4.38°** | **3646.5** | ✅ **PASSED** | None (All passed) |
| **Hop 2: Baseline SIFT** | 6 / 272 | 6 / 272 | 0.4210 | +64.19° | 354252.5 | 🛑 **GATED** | inlier_consensus |
| **Hop 2: Phase Congruency + LoFTR** | **124 / 307** | **133 / 307** | **1.0580** | **-0.34°** | **688.9** | ✅ **PASSED** | None (All passed) |

---

## 3. Reprojection RMSE & Sub-Pixel Status

> **Sub-pixel accuracy (RMSE < 1 px) is NOT achieved on either hop.**
> - **Hop 1:** RMSE `9.23 px` at TMC-2 GSD 4.72 m/px = `43.6 m`.
> - **Hop 2:** RMSE `9.05 px` at IIRS GSD 68.38 m/px = `618.5 m`.
> These are structural localization results across an 18.15× and a 14.49× resolution divide. The problem statement target of sub-pixel accuracy is not met by the current global 4-DoF model.

---

## 4. Match Point Distribution (Stated as a Limitation)

Match points are **not uniformly distributed** across the full image area:
- **Hop 1 (OHRC ↔ TMC-2):** Inliers occupy `22 / 64` cells (34.4% coverage) on an 8×8 grid, with a count coefficient of variation (CV) of `0.715`.
- **Hop 2 (TMC-2 ↔ IIRS):** Inliers occupy `39 / 64` cells (60.9% coverage), with a CV of `0.866`.
- Inliers cluster predominantly along high-contrast crater rims and sunlit ridges; shadowed crater basins lack sufficient texture to support keypoint correspondences.

---

## 5. Spaceflight Validity Gate Rejections

The spaceflight validity gate evaluated three sequential rules:
1. **Inlier consensus floor:** inliers ≥ 20 AND ratio ≥ 15.0%.
2. **Numerical stability:** condition number cond(H) ≤ 100,000.
3. **Physical plausibility:** |rotation| ≤ 30.0° AND scale error ≤ 25.0%.

- **Hop 1 SIFT Rejection:** Failed on inlier consensus (5 < 20, 1.2% < 15%), ill-conditioning ($cond = 2.4\times 10^6$), and unphysical rotation ($-151.15^\circ$).
- **Hop 2 SIFT Rejection:** Failed on inlier consensus (6 < 20, 2.2% < 15%), ill-conditioning ($cond = 3.5\times 10^5$), unphysical rotation ($+64.19^\circ$), and scale collapse ($0.4210$, 57.9% error).

---

## 6. Known Limitations

1. **Gate Width in Ground Distance:** The 15.0 px RANSAC threshold corresponds to **70.80 m** on Hop 1 (TMC-2 GSD 4.72 m/px) and **1,025.70 m** on Hop 2 (IIRS GSD 68.38 m/px). Tightening the gate to 5.0 px (23.6 m on Hop 1, 341.9 m on Hop 2) reduces inliers to 10 on Hop 1 and 24 on Hop 2.
2. **Terrain Slope Correlation (F3 Result):** Reprojection residuals show only a weak, statistically non-significant correlation with LOLA DEM terrain slope ($r = +0.151, p = 0.301$ on Hop 1; $r = +0.125, p = 0.166$ on Hop 2). The hypothesis that residual error is purely topographic parallax is unproven.
3. **Pre-Scaling Context:** Because crops were pre-scaled to standard display canvas sizes (1000×1000 and 800×800) before matching, the recovered scale reflects residual scale differences between crops (~1.01 and ~1.06), not the unnormalized raw GSD ratios.
