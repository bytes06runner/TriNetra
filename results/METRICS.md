# TriNetra Evaluation Metrics Scorecard (ISRO SIH26166)

> **Disclosure:** Both crops were independently decimated to a common canvas size before matching, which absorbs the nominal 18.15x and 14.49x sensor GSD ratios. Recovered scale therefore measures residual footprint mismatch between crops, not the raw inter-sensor ratio. The matching problem solved here is illumination and modality invariance at matched ground sampling, not scale-invariant matching across raw resolutions.

> **Notice:** All metrics are empirically measured from authentic Chandrayaan-2 flight crops at Shiv Shakti Point (-69.58°S) and South Pole (-70.85°S).
> Deterministic seed: `42`. Lunar radius: `1,737,400 m`. RANSAC inlier threshold: `15.0 px`.

---

## 1. Primary Evaluation Scorecard (Problem Statement Deliverable 3)

| Metric | Hop 1: Baseline SIFT | Hop 1: Fine-Tuned LoFTR | Hop 2: Baseline SIFT | Hop 2: Phase Congruency + LoFTR |
|:---|:---:|:---:|:---:|:---:|
| **Sensor Pair** | OHRC ↔ TMC-2 | OHRC ↔ TMC-2 | TMC-2 ↔ IIRS | TMC-2 ↔ IIRS |
| **Resolution Gap** | 18.15× (0.26 ↔ 4.72 m/px) | 18.15× (0.26 ↔ 4.72 m/px) | 14.49× (4.72 ↔ 68.38 m/px) | 14.49× (4.72 ↔ 68.38 m/px) |
| **Inliers (RANSAC Consensus)** | 5 | **49** [^1] | 6 | **124** [^1] |
| **Total Matches** | 409 | 217 | 272 | 307 |
| **Inlier Threshold (px)** | 15.0 px | 15.0 px | 15.0 px | 15.0 px |
| **Inlier Threshold (m)** | 70.8 m | 70.8 m | 1025.7 m | 1025.7 m |
| **Inlier Ratio (RANSAC)** | 1.22% | **22.58%** | 2.21% | **40.39%** |
| **Matches Strictly ≤ 15.0 px** | 5 (1.22%) | **47 (21.66%)** [^1] | 6 (2.21%) | **133 (43.32%)** [^1] |
| **Flight Gate Status (F4)** | GATED (inlier_consensus) | PASS (all criteria) | GATED (inlier_consensus) | PASS (all criteria) |
| **Reprojection RMSE (px)** | 5.34 px | 9.23 px | 4.54 px | 9.05 px |
| **Reprojection RMSE (m)** | 25.2 m | 43.6 m | 310.5 m | 618.5 m |
| **Cached RMSE Cross-Check** | 5.3391 (Δ=0.00001) | 9.23 (Δ=0.00155) | 4.5413 (Δ=0.00001) | 9.05 (Δ=0.00488) |
| **Sub-Pixel Achieved?** | **No** (RMSE ≥ 1 px) | **No** (RMSE ≥ 1 px) | **No** (RMSE ≥ 1 px) | **No** (RMSE ≥ 1 px) |
| **Max Residual (px)** | 7.40 px | 15.69 px [^2] | 6.51 px | 17.48 px [^2] |
| **Median Residual (px)** | 4.99 px | 9.01 px | 4.19 px | 7.89 px |
| **95th-Percentile Residual** | 7.08 px | 14.10 px | 6.24 px | 14.46 px |
| **8×8 Grid Occupancy** | 5 / 64 (non-uniform) | 22 / 64 (non-uniform) | 5 / 64 (non-uniform) | 39 / 64 (non-uniform) |
| **8×8 Grid Coverage (%)** | 7.8% | 34.4% | 7.8% | 60.9% |
| **Grid Count CV (std/mean)** | 0.000 | 0.715 | 0.333 | 0.866 |
| **NN Mean Distance (px)** | 232.29 px | 39.68 px | 86.20 px | 28.30 px |
| **Matrix Condition Number** | 2352526.5 (ill-conditioned) | 3646.5 (stable) | 354252.5 (ill-conditioned) | 688.9 (stable) |
| **Fitted Scale Factor** | 0.9664 | 1.0086 | 0.4210 (diverges 58%) | 1.0580 |
| **Fitted Rotation (deg)** | -151.15° (Degenerate) | -4.38° | 64.19° (Degenerate) | -0.34° |
| **Fitted Translation (tx, ty)** | (835.5, 1255.2) | (-39.1, 46.4) | (297.0, 246.8) | (-26.8, -2.9) |

[^1]: **Inlier Count Discrepancy Explanation (F5a):** RANSAC consensus inlier selection occurs under a minimal-sample hypothesis at threshold 15.0 px. OpenCV then performs an unweighted linear least-squares re-fit of the transformation matrix on all consensus inliers to minimize global sum-of-squared errors. This slight shift causes 2 points on Hop 1 and 6 points on Hop 2 to fall slightly outside 15.0 px, yielding 47 matches on Hop 1 (vs 49 consensus inliers) and 133 matches on Hop 2 (127 consensus + 6 non-consensus points within 15.0 px of the final least-squares plane). Both figures are reported side by side.

[^2]: **Max Residual Explanation (F5b):** The maximum inlier residual reaches 15.69 px (Hop 1) and 17.48 px (Hop 2) because the final transformation matrix $H$ is least-squares re-estimated across the entire consensus set after RANSAC sampling, which slightly widens residuals for boundary points compared to the initial minimal solver hypothesis.

---

## 2. Transform Stability Under Gate Tightening (F2)

For each configuration, the 4-DoF similarity transform was independently re-fitted by linear least-squares using only the matches strictly within each gate (skipping gates with < 4 matches):

### Hop 1: Fine-Tuned EfficientLoFTR

| Gate (px) | Matches Used (n) | Status | Refitted Scale | Refitted Rot (deg) | Refitted tx | Refitted ty | Condition No. | Fit RMSE (px) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 15.0 px | 47 | VALID | 1.0096 | -4.36° | -38.7 | +45.5 | 3537.3 | 8.83 px |
| 10.0 px | 34 | VALID | 1.0056 | -4.42° | -37.2 | +47.9 | 3666.9 | 7.12 px |
| 5.0 px | 10 | VALID | 1.0056 | -4.29° | -36.5 | +47.8 | 3592.5 | 3.19 px |
| 3.0 px | 4 | VALID | 1.0087 | -4.13° | -38.2 | +43.4 | 3320.7 | 1.15 px |

> **Interpretation:** *Refitting on progressively tighter subsets changes the recovered scale by 0.004 and the rotation by 0.28 degrees.*

### Hop 2: Phase Congruency + LoFTR

| Gate (px) | Matches Used (n) | Status | Refitted Scale | Refitted Rot (deg) | Refitted tx | Refitted ty | Condition No. | Fit RMSE (px) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 15.0 px | 133 | VALID | 1.0544 | -0.12° | -25.2 | -3.7 | 619.8 | 9.16 px |
| 10.0 px | 87 | VALID | 1.0545 | -0.03° | -24.0 | -4.3 | 565.5 | 6.54 px |
| 5.0 px | 24 | VALID | 1.0555 | -0.20° | -24.8 | -3.1 | 595.9 | 3.08 px |
| 3.0 px | 12 | VALID | 1.0580 | -0.17° | -25.5 | -4.2 | 634.7 | 1.98 px |

> **Interpretation:** *Refitting on progressively tighter subsets changes the recovered scale by 0.004 and the rotation by 0.17 degrees.*

---

## 3. Residual vs Terrain Slope Analysis (F3)

We empirically tested whether reprojection residuals correlate with local terrain slope by sampling the LOLA south polar DEM (`south_pole_subset.tif`) at every inlier match coordinate:

| Configuration | Inliers Evaluated (n) | Pearson r (Residual vs Slope) | p-value (Pearson) | Spearman ρ (Residual vs Slope) | p-value (Spearman) | Pearson r (vs |Elev - Mean|) | Spearman ρ (vs |Elev - Mean|) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hop 1: OHRC ↔ TMC-2** | 49 | +0.1509 | 0.3006 | +0.1454 | 0.3189 | -0.1353 | -0.2425 |
| **Hop 2: TMC-2 ↔ IIRS** | 124 | +0.1253 | 0.1656 | +0.1348 | 0.1355 | -0.0985 | -0.1225 |

> **Scientific Finding:** Residuals are not explained by local terrain slope in our measurements (r = +0.151, p = 0.301 Hop 1; r = +0.125, p = 0.166 Hop 2). The error budget is unresolved and likely combines point-spread blur across the scale gap, unmodelled lens distortion, and keypoint localisation uncertainty.

---

## 4. Threshold Sweeps & Noise-Floor Baseline (F6)

Evaluated on the full match set under the original fitted matrix $H$. The indicative uniform-random expectation is scaled from the 15 px count by $(r/15)^2$:

### Hop 1: Fine-Tuned EfficientLoFTR (Total Matches = 217)

| Gate (px) | Gate (m at TMC-2 GSD) | Observed Matches | Observed Ratio (%) | Fit RMSE (px) | Fit RMSE (m) | Uniform-Random Expectation (indicative) | Observed / Expected Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 15.0 px | 70.8 m | 47 / 217 | 21.66% | 8.859 px | 41.8 m | 47.00 | 1.00x |
| 10.0 px | 47.2 m | 34 / 217 | 15.67% | 7.188 px | 33.9 m | 20.89 | 1.63x |
| 5.0 px | 23.6 m | 10 / 217 | 4.61% | 3.409 px | 16.1 m | 5.22 | 1.91x |
| 3.0 px | 14.2 m | 4 / 217 | 1.84% | 2.094 px | 9.9 m | 1.88 | 2.13x |
| 2.0 px | 9.4 m | 2 / 217 | 0.92% | 1.956 px | 9.2 m | 0.84 | 2.39x |
| 1.0 px | 4.7 m | 0 / 217 | 0.00% | 0.000 px | 0.0 m | 0.21 | 0.00x |

### Hop 2: Phase Congruency + LoFTR (Total Matches = 307)

| Gate (px) | Gate (m at IIRS GSD) | Observed Matches | Observed Ratio (%) | Fit RMSE (px) | Fit RMSE (m) | Uniform-Random Expectation (indicative) | Observed / Expected Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 15.0 px | 1025.7 m | 133 / 307 | 43.32% | 9.329 px | 637.9 m | 133.00 | 1.00x |
| 10.0 px | 683.8 m | 87 / 307 | 28.34% | 6.732 px | 460.3 m | 59.11 | 1.47x |
| 5.0 px | 341.9 m | 24 / 307 | 7.82% | 3.208 px | 219.4 m | 14.78 | 1.62x |
| 3.0 px | 205.1 m | 12 / 307 | 3.91% | 2.188 px | 149.6 m | 5.32 | 2.26x |
| 2.0 px | 136.8 m | 3 / 307 | 0.98% | 1.173 px | 80.2 m | 2.36 | 1.27x |
| 1.0 px | 68.4 m | 1 / 307 | 0.33% | 0.428 px | 29.3 m | 0.59 | 1.69x |

---

## 5. Multi-Hop Transform Composition (OHRC → TMC-2 → IIRS)

- **Composed Transform Matrix:** $\mathbf{H}_{\text{OHRC} \to \text{IIRS}} = \mathbf{H}_{\text{TMC-2} \to \text{IIRS}} \cdot \mathbf{H}_{\text{OHRC} \to \text{TMC-2}}$
- **Composed Scale Factor:** `1.0671`
- **Composed Rotation:** `-4.72°`
- **Composed Translation:** `(-67.9, 46.4)`
- **Condition Number:** `6336.3`
- **Linear Error Propagation RMSE (px):** `13.31 px`
- **Linear Error Propagation RMSE (m):** `910.3 m` (at IIRS GSD 68.38 m/px)
- **Scientific Caveat:** *This is error propagation through the transform chain, not a direct end-to-end measurement on a shared OHRC<->IIRS overlap.*

---

## 6. Physical Specifications & Ground Conversions (F7)

### Lunar Coordinate Header & Planet Constant (A1)
Every match point CSV and GeoJSON contains the exact IAU lunar physical model coordinate header:
```text
# Local planar approximation about anchor (lat, lon). Lunar radius 1737400 m. Valid only within this crop. Not geodetic coordinates.
```
- **Lunar Mean Radius:** `R = 1,737,400.0 m`
- **Metres Per Degree Latitude:** `π × 1737400 / 180 = 30,323.35 m/deg`
- **Metres Per Degree Longitude:** `30,323.35 × cos(latitude) m/deg`

### Physical Meaning of the 15.0 px RANSAC Gate (F7b)
- **Ground Metric at Hop 1 (TMC-2 4.72 m/px):** `15.0 px × 4.72 m/px = 70.80 m`.
- **Ground Metric at Hop 2 (IIRS 68.38 m/px):** `15.0 px × 68.38 m/px = 1,025.70 m`.

### Band Range Provenance (F7a & A4)
- **Code Source of Truth (`src/pds_loader.py`):** `iirs_to_grey()` uses `max_nm=2000.0`, isolating bands 1–77.
- **Spectral Coverage:** Verified from PDS4 XML calibration table (`ch2_iir_nri_20231003T2152304115_d_img_d18.xml`): Band 1 center wavelength is `712.3 nm`, Band 2 is `729.2 nm`, Band 77 is `1993.1 nm`.
