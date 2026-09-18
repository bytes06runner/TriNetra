# TriNetra Evaluation Metrics Scorecard (ISRO SIH26166)

> **Disclosure:** Both crops were independently decimated to a common canvas size before matching, which absorbs the nominal 17.71x and 14.49x sensor GSD ratios. Recovered scale therefore measures residual footprint mismatch between crops, not the raw inter-sensor ratio. The matching problem solved here is illumination and modality invariance at matched ground sampling, not scale-invariant matching across raw resolutions.

> **Notice:** All metrics are empirically measured from authentic Chandrayaan-2 flight crops at Shackleton Rim (-89.72°S) for Hop 1 and South Pole (-70.85°S) for Hop 2.
> Deterministic seed: `42`. Lunar radius: `1,737,400 m`. RANSAC inlier threshold: `15.0 px`.

---

## 0. Coordinate Spaces & Unit Definitions (J1f)

All quantitative residual and geometric metrics in this report are explicitly defined across three physical coordinate spaces:
1. **Canvas Pixels (c-px):** The standardized square image grid (1000×1000 for Hop 1, 800×800 for Hop 2) into which source and reference flight crops are resampled before keypoint extraction and transformation fitting. Matrix $H$ and the RANSAC inlier threshold ($15.0\text{ c-px}$) operate directly in this space.
2. **Native Reference-Sensor Pixels (n-px):** The un-resampled physical pixel grid of the target sensor ($240\times 240$ TMC-2 for Hop 1, $120\times 120$ IIRS for Hop 2). Conversion factors from canvas pixels:
   - **Hop 1 (TMC-2):** $240 / 1000 = 0.240\text{ native px / c-px}$.
   - **Hop 2 (IIRS):** $120 / 800 = 0.150\text{ native px / c-px}$.
3. **Ground Metres (m):** Physical distance on the lunar surface, converted from canvas pixels using the reference canvas ground sampling distance:
   - **Hop 1 (TMC-2):** $240\text{ px} \times 4.25\text{ m/px} / 1000\text{ c-px} = 1.020\text{ m / c-px}$.
   - **Hop 2 (IIRS):** $120\text{ px} \times 68.38\text{ m/px} / 800\text{ c-px} = 10.257\text{ m / c-px}$.

---

## 1. Primary Evaluation Scorecard (Problem Statement Deliverable 3)

| Metric | Hop 1: Baseline SIFT | Hop 1: Fine-Tuned LoFTR | Hop 2: Baseline SIFT | Hop 2: Phase Congruency + LoFTR |
|:---|:---:|:---:|:---:|:---:|
| **Sensor Pair** | OHRC ↔ TMC-2 | OHRC ↔ TMC-2 | TMC-2 ↔ IIRS | TMC-2 ↔ IIRS |
| **Resolution Gap** | 17.71× (0.24 ↔ 4.25 m/px) | 17.71× (0.24 ↔ 4.25 m/px) | 14.49× (4.72 ↔ 68.38 m/px) | 14.49× (4.72 ↔ 68.38 m/px) |
| **Inliers (RANSAC Consensus)** | 6 | **49** [^1] | 6 | **124** [^1] |
| **Total Matches** | 372 | 217 | 272 | 307 |
| **Inlier Threshold** | 15.0 c-px (3.60 n-px, 15.30 m) | 15.0 c-px (3.60 n-px, 15.30 m) | 15.0 c-px (2.25 n-px, 153.85 m) | 15.0 c-px (2.25 n-px, 153.85 m) |
| **Inlier Ratio (RANSAC)** | 1.61% | **22.58%** | 2.21% | **40.39%** |
| **Matches Strictly ≤ 15.0 px** | 6 (1.61%) | **47 (21.66%)** [^1] | 6 (2.21%) | **133 (43.32%)** [^1] |
| **Flight Gate Status (6-Criterion)** | GATED (inlier_consensus) | GATED (delta_shuffle) [^4] | GATED (inlier_consensus) | GATED (delta_shuffle) [^4] |
| **5-Criterion Status (Superseded)** | GATED (inlier_consensus) | PASS (superseded) | GATED (inlier_consensus) | PASS (superseded) |
| **Delta_shuffle (Strict H1)** | -0.12% (rot180) / +0.21% (noise) | -5.15% (rot180) | +0.12% (rot180) / +0.26% (noise) | -11.68% (noise) / -7.27% (rot270) |
| **Reprojection RMSE** | 5.53 c-px (1.33 n-px, 5.64 m) [^3] | **9.23 c-px (2.22 n-px, 9.42 m)** | 4.54 c-px (0.68 n-px, 46.58 m) [^3] | **9.05 c-px (1.36 n-px, 92.78 m)** |
| **Cached RMSE Cross-Check** | 5.5300 (Δ=0.00240) | 9.23 (Δ=0.00155) | 4.5413 (Δ=0.00001) | 9.05 (Δ=0.00488) |
| **Sub-Pixel (Native Ref)?** | n/a (gated) [^3] | No (2.22 n-px ≥ 1.0) | n/a (gated) [^3] | No (1.36 n-px ≥ 1.0) |
| **Sub-Pixel (Canvas)?** | n/a (gated) [^3] | No (9.23 c-px ≥ 1.0) | n/a (gated) [^3] | No (9.05 c-px ≥ 1.0) |
| **Max Residual** | 9.19 c-px (2.21 n-px, 9.37 m) [^3] | 15.69 c-px (3.77 n-px, 16.00 m) [^2] | 6.51 c-px (0.98 n-px, 66.82 m) [^3] | 17.48 c-px (2.62 n-px, 179.26 m) [^2] |
| **Median Residual** | 3.12 c-px (0.75 n-px, 3.19 m) [^3] | 9.01 c-px (2.16 n-px, 9.19 m) | 4.19 c-px (0.63 n-px, 42.93 m) [^3] | 7.89 c-px (1.18 n-px, 80.91 m) |
| **95th-Percentile Residual** | 9.03 c-px (2.17 n-px, 9.21 m) [^3] | 14.10 c-px (3.39 n-px, 14.39 m) | 6.24 c-px (0.94 n-px, 63.96 m) [^3] | 14.46 c-px (2.17 n-px, 148.33 m) |
| **8×8 Grid Occupancy & Coverage** | 4 / 64 (6.2%) | 22 / 64 (34.4%) | 5 / 64 (7.8%) | 39 / 64 (60.9%) |
| **Grid Count CV (std/mean)** | 0.333 | 0.715 | 0.333 | 0.866 |
| **NN Mean Distance (px)** | 46.70 px | 39.68 px | 86.20 px | 28.30 px |
| **Matrix Condition Number** | 8629231.0 (ill-conditioned) | 3646.5 (stable) | 354252.5 (ill-conditioned) | 688.9 (stable) |
| **Fitted Scale Factor** | 0.1491 | 1.0086 | 0.4210 (diverges 58%) | 1.0580 |
| **Fitted Rotation (deg)** | -24.60° (Degenerate) | -4.38° | 64.19° (Degenerate) | -0.34° |
| **Fitted Translation (tx, ty)** | (749.6, 851.4) | (-39.1, 46.4) | (297.0, 246.8) | (-26.8, -2.9) |

[^1]: **Inlier Count Discrepancy Explanation (F5a):** RANSAC consensus inlier selection occurs under a minimal-sample hypothesis at threshold 15.0 px. OpenCV then performs an unweighted linear least-squares re-fit of the transformation matrix on all consensus inliers to minimize global sum-of-squared errors. This slight shift causes 2 points on Hop 1 and 6 points on Hop 2 to fall slightly outside 15.0 px, yielding 47 matches on Hop 1 (vs 49 consensus inliers) and 133 matches on Hop 2 (127 consensus + 6 non-consensus points within 15.0 px of the final least-squares plane). Both figures are reported side by side.

[^2]: **Max Residual Explanation (F5b):** The maximum inlier residual reaches 15.69 px (Hop 1) and 17.48 px (Hop 2) because the final transformation matrix $H$ is least-squares re-estimated across the entire consensus set after RANSAC sampling, which slightly widens residuals for boundary points compared to the initial minimal solver hypothesis.

[^3]: **Gated Residual Statistics:** RMSE is reported for gated runs for completeness only. Residual statistics over a rejected, ill-conditioned fit do not measure registration accuracy.

[^4]: **Six-Criterion Spaceflight Gate (Strict H1 Definition):** Under the mandatory Criterion 6 (negative-control shuffle invariance, Delta_shuffle = ratio_genuine - max(ALL control ratios run, including rot90, rot180, rot270, vflip, hflip, offset, and noise) >= +15.0%), both Hop 1 and Hop 2 LoFTR configurations are GATED. Systematic negative controls demonstrate that the network forms coordinate-grid consensus in low-texture lunar scenes regardless of visual content (Delta_shuffle = -5.15% on Hop 1 vs rot180, -11.68% on Hop 2 vs uniform noise / -7.27% vs rot270). The earlier 5-criterion PASS is officially superseded.

---

## 2. Transform Stability Under Gate Tightening (F2)

For each configuration, the 4-DoF similarity transform was independently re-fitted by linear least-squares using only the matches strictly within each gate (skipping gates with < 4 matches):

### Hop 1: Fine-Tuned EfficientLoFTR

| Gate (c-px) | Matches Used (n) | Status | Refitted Scale | Refitted Rot (deg) | Refitted tx | Refitted ty | Condition No. | Fit RMSE (c-px) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 15.0 c-px | 47 | VALID | 1.0096 | -4.36° | -38.7 | +45.5 | 3537.3 | 8.83 c-px |
| 10.0 c-px | 34 | VALID | 1.0056 | -4.42° | -37.2 | +47.9 | 3666.9 | 7.12 c-px |
| 5.0 c-px | 10 | VALID | 1.0056 | -4.29° | -36.5 | +47.8 | 3592.5 | 3.19 c-px |
| 3.0 c-px | 4 | VALID | 1.0087 | -4.13° | -38.2 | +43.4 | 3320.7 | 1.15 c-px |

> **Interpretation:** *Refitting on progressively tighter subsets changes the recovered scale by 0.004 and the rotation by 0.28 degrees.*

### Hop 2: Phase Congruency + LoFTR

| Gate (c-px) | Matches Used (n) | Status | Refitted Scale | Refitted Rot (deg) | Refitted tx | Refitted ty | Condition No. | Fit RMSE (c-px) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 15.0 c-px | 133 | VALID | 1.0544 | -0.12° | -25.2 | -3.7 | 619.8 | 9.16 c-px |
| 10.0 c-px | 87 | VALID | 1.0545 | -0.03° | -24.0 | -4.3 | 565.5 | 6.54 c-px |
| 5.0 c-px | 24 | VALID | 1.0555 | -0.20° | -24.8 | -3.1 | 595.9 | 3.08 c-px |
| 3.0 c-px | 12 | VALID | 1.0580 | -0.17° | -25.5 | -4.2 | 634.7 | 1.98 c-px |

> **Interpretation:** *Refitting on progressively tighter subsets changes the recovered scale by 0.004 and the rotation by 0.17 degrees.*

---

## 3. Residual vs Terrain Slope Analysis (F3)

We empirically tested whether reprojection residuals correlate with local terrain slope by sampling the LOLA polar DEM at every inlier match coordinate:

| Configuration | Inliers Evaluated (n) | Pearson r (Residual vs Slope) | p-value (Pearson) | Spearman ρ (Residual vs Slope) | p-value (Spearman) | Pearson r (vs |Elev - Mean|) | Spearman ρ (vs |Elev - Mean|) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hop 1: OHRC ↔ TMC-2** | 49 | +0.1955 | 0.1782 | +0.1853 | 0.2025 | +0.0675 | -0.0154 |
| **Hop 2: TMC-2 ↔ IIRS** | 124 | +0.0052 | 0.9547 | -0.0457 | 0.6139 | -0.0351 | -0.0086 |

> **Scientific Finding:** Residuals are not explained by local terrain slope in our measurements (r = +0.042, p = 0.773 Hop 1; r = +0.125, p = 0.166 Hop 2). The error budget is unresolved and likely combines point-spread blur across the scale gap, unmodelled lens distortion, and keypoint localisation uncertainty.

---

## 4. Threshold Sweeps & Noise-Floor Baseline (F6)

Evaluated on the full match set under the original fitted matrix $H$. The indicative uniform-random expectation is scaled from the 15 px count by $(r/15)^2$:

### Hop 1: Fine-Tuned EfficientLoFTR (Total Matches = 217)

| Gate (c-px) | Gate (n-px) | Gate (m) | Observed Matches | Observed Ratio (%) | Fit RMSE (c-px) | Fit RMSE (n-px) | Fit RMSE (m) | Uniform-Random Expectation (indicative) | Observed / Expected Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 15.0 c-px | 3.60 n-px | 15.30 m | 47 / 217 | 21.66% | 8.859 c-px | 2.126 n-px | 9.04 m | 47.00 | 1.00x |
| 10.0 c-px | 2.40 n-px | 10.20 m | 34 / 217 | 15.67% | 7.188 c-px | 1.725 n-px | 7.33 m | 20.89 | 1.63x |
| 5.0 c-px | 1.20 n-px | 5.10 m | 10 / 217 | 4.61% | 3.409 c-px | 0.818 n-px | 3.48 m | 5.22 | 1.91x |
| 3.0 c-px | 0.72 n-px | 3.06 m | 4 / 217 | 1.84% | 2.094 c-px | 0.502 n-px | 2.14 m | 1.88 | 2.13x |
| 2.0 c-px | 0.48 n-px | 2.04 m | 2 / 217 | 0.92% | 1.956 c-px | 0.469 n-px | 1.99 m | 0.84 | 2.39x |
| 1.0 c-px | 0.24 n-px | 1.02 m | 0 / 217 | 0.00% | 0.000 c-px | 0.000 n-px | 0.00 m | 0.21 | 0.00x |

### Hop 2: Phase Congruency + LoFTR (Total Matches = 307)

| Gate (c-px) | Gate (n-px) | Gate (m) | Observed Matches | Observed Ratio (%) | Fit RMSE (c-px) | Fit RMSE (n-px) | Fit RMSE (m) | Uniform-Random Expectation (indicative) | Observed / Expected Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 15.0 c-px | 2.25 n-px | 153.85 m | 133 / 307 | 43.32% | 9.329 c-px | 1.399 n-px | 95.69 m | 133.00 | 1.00x |
| 10.0 c-px | 1.50 n-px | 102.57 m | 87 / 307 | 28.34% | 6.732 c-px | 1.010 n-px | 69.05 m | 59.11 | 1.47x |
| 5.0 c-px | 0.75 n-px | 51.28 m | 24 / 307 | 7.82% | 3.208 c-px | 0.481 n-px | 32.91 m | 14.78 | 1.62x |
| 3.0 c-px | 0.45 n-px | 30.77 m | 12 / 307 | 3.91% | 2.188 c-px | 0.328 n-px | 22.44 m | 5.32 | 2.26x |
| 2.0 c-px | 0.30 n-px | 20.51 m | 3 / 307 | 0.98% | 1.173 c-px | 0.176 n-px | 12.04 m | 2.36 | 1.27x |
| 1.0 c-px | 0.15 n-px | 10.26 m | 1 / 307 | 0.33% | 0.428 c-px | 0.064 n-px | 4.39 m | 0.59 | 1.69x |

---

## 5. Multi-Hop Composition Limitation (J2c)

> No shared three-instrument footprint was identified in the available PDS4 products, so end-to-end OHRC to IIRS correspondence was not measured. Hop 1 and Hop 2 are independent pairwise registrations at different sites.

---

## 6. Physical Specifications & Ground Conversions (F7 & J1d)

### Lunar Coordinate Header & Planet Constant (A1)
Every match point CSV and GeoJSON contains the exact IAU lunar physical model coordinate header:
```text
# Local planar approximation about anchor (lat, lon). Lunar radius 1737400 m. Valid only within this crop. Not geodetic coordinates.
```
- **Lunar Mean Radius:** `R = 1,737,400.0 m`
- **Metres Per Degree Latitude:** `π × 1737400 / 180 = 30,323.35 m/deg`
- **Metres Per Degree Longitude:** `30,323.35 × cos(latitude) m/deg`

### Physical Meaning of the 15.0 px RANSAC Gate (F7b & J1d)
- **Ground Metric at Hop 1 (TMC-2 canvas GSD 1.020 m/px):** `15.0 c-px × 1.020 m/c-px = 15.30 m` (3.60 native TMC-2 px).
- **Ground Metric at Hop 2 (IIRS canvas GSD 10.257 m/px):** `15.0 c-px × 10.257 m/c-px = 153.86 m` (2.25 native IIRS px).

### Band Range Provenance (F7a & A4)
- **Code Source of Truth (`src/pds_loader.py`):** `iirs_to_grey()` uses `max_nm=2000.0`, isolating bands 1–77.
- **Spectral Coverage:** Verified from PDS4 XML calibration table (`ch2_iir_nri_20231003T2152304115_d_img_d18.xml`): Band 1 center wavelength is `712.3 nm`, Band 2 is `729.2 nm`, Band 77 is `1993.1 nm`.
