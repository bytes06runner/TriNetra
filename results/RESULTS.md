# TriNetra (त्रिनेत्र) — Reviewer Executive Summary

This one-page summary summarizes the empirical registration results for Chandrayaan-2 planetary correspondence (Problem Statement SIH26166).

---

## 1. Measured Observation Datasets

> **Read this first:** Both crops were independently decimated to a common canvas size before matching, which absorbs the nominal 18.15x and 14.49x sensor GSD ratios. Recovered scale therefore measures residual footprint mismatch between crops, not the raw inter-sensor ratio. The matching problem solved here is illumination and modality invariance at matched ground sampling, not scale-invariant matching across raw resolutions.

All measurements were conducted on authentic Chandrayaan-2 PDS4 flight products downloaded from the ISSDC repository:
- **OHRC (0.26 m/px):** `ch2_ohr_ncp_20211023T0027462822_d_img_d18` (Lines 10000:14000, Samples 4000:8000; Center: -69.58019°S, 32.28800°E; Sun Azimuth: 298.43°, Elevation: 9.13°, Roll: +15.76°).
- **TMC-2 (4.72 m/px):** `ch2_tmc_ncn_20230130T1900132182_d_img_d32` (South Pole nadir track; Center: -69.57911°S, 32.27507°E and -70.85000°S, 32.26000°E; Sun Azimuth: 53.02°, Elevation: 17.22°, Roll: -0.02°).
- **IIRS (68.38 m/px):** `ch2_iir_nri_20231003T2152304115_d_img_d18` (Raw Level-1 hyperspectral cube, Lines 510:630, Samples 60:180; Center: -70.85000°S, 32.26000°E; Sun Azimuth: 277.20°, Elevation: 2.29°).
- **Cross-Illumination Disparity:** The solar azimuth angle difference between OHRC (298.43°) and TMC-2 (53.02°) is **114.6°** ($360^\circ - (298.43^\circ - 53.02^\circ)$). The spacecraft roll offset is **15.8°**.

---

## 2. Transformation Conditioning & Plausibility

*Note: Scale and rotation parameters are estimated in pre-scaled canvas space (1000×1000 for Hop 1, 800×800 for Hop 2), not raw sensor space.*

| Configuration | Inliers (RANSAC) | Inliers (≤15px) | Recovered Scale | Recovered Rotation | Condition Number | Validity Gate Status | First Failing Criterion |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hop 1: Baseline SIFT** | 5 / 409 | 5 / 409 | 0.9664 | -151.15° | 2352526.5 | **GATED** | inlier_consensus |
| **Hop 1: Fine-Tuned LoFTR** | **49 / 217** | **47 / 217** | **1.0086** | **-4.38°** | **3646.5** | **PASS** | None (All passed) |
| **Hop 2: Baseline SIFT** | 6 / 272 | 6 / 272 | 0.4210 | +64.19° | 354252.5 | **GATED** | inlier_consensus |
| **Hop 2: Phase Congruency + LoFTR** | **124 / 307** | **133 / 307** | **1.0580** | **-0.34°** | **688.9** | **PASS** | None (All passed) |

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

## 5. Spaceflight Validity Gate & Ablation Analysis

The spaceflight validity gate evaluated three sequential rules:
1. **Inlier consensus floor:** inliers ≥ 20 AND ratio ≥ 15.0%.
2. **Numerical stability:** condition number cond(H) ≤ 100,000.
3. **Physical plausibility:** |rotation| ≤ 30.0° AND scale error ≤ 25.0% relative to expected canvas scale 1.0.

- **Expected Scale Setting:** The expected scale in the physical plausibility rule is set to 1.0 because both crops were pre-scaled to a common canvas size before matching. Any fitted scale factor diverging by >25% from 1.0 in canvas space indicates unphysical geometric distortion.
- **Ablation Finding (G2):** The gate catches degenerate geometry independently of match count: when the inlier consensus rule is disabled in ablation, both SIFT runs are still rejected by the conditioning rule alone (Hop 1 cond = 2.4e+06, Hop 2 cond = 3.5e+05, threshold 1.0e+05), while both passing runs remain valid.
- **Hop 1 SIFT Rejection:** Failed on inlier consensus (5 < 20, 1.2% < 15%), ill-conditioning ($cond = 2.4\times 10^6$), and unphysical rotation ($-151.15^\circ$).
- **Hop 2 SIFT Rejection:** Failed on inlier consensus (6 < 20, 2.2% < 15%), ill-conditioning ($cond = 3.5\times 10^5$), unphysical rotation ($+64.19^\circ$), and scale collapse ($0.4210$, 57.9% error).

---

## 6. Known Limitations

1. **Gate Width in Ground Distance:** The 15.0 px RANSAC threshold corresponds to **70.80 m** on Hop 1 (TMC-2 GSD 4.72 m/px) and **1,025.70 m** on Hop 2 (IIRS GSD 68.38 m/px). Tightening the gate to 5.0 px (23.6 m on Hop 1, 341.9 m on Hop 2) reduces inliers to 10 on Hop 1 and 24 on Hop 2.
2. **Terrain Slope Correlation & Roadmap Justification:** Residuals are not explained by local terrain slope in our measurements ($r = +0.151, p = 0.301$ on Hop 1; $r = +0.125, p = 0.166$ on Hop 2). The error budget is unresolved and likely combines point-spread blur across the scale gap, unmodelled lens distortion, and keypoint localisation uncertainty. Because slope does not explain residual magnitude, the empirical justification for the Thin-Plate Spline (TPS) non-rigid refinement roadmap item is weakened.
3. **Residual Scale Gap:** On Hop 1, the fitted scale is 1.0086 (residual gap of 0.0674 / 6.74% relative to polar footprint ratio 0.9412, or 0.0086 / 0.86% relative to unit canvas). On Hop 2, the fitted scale is 1.0580 compared to the expected canvas footprint ratio of 0.9997 (8203.36 m / 8205.60 m); this 0.0583 (5.83%) residual scale gap is unexplained.
