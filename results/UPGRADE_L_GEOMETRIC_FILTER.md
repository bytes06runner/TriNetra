# Stage L: Geometric Consistency Filtering at Pitiscus

**Site:** Pitiscus Lobate Scarp (DTM `PITISCUS`, Sitename: Pitiscus Lobate Scarp)  
**Coordinates:** Latitude -51.25 deg S, Longitude 31.25 deg E  
**Reference Product:** LROC NAC DTM Orthophoto (`NAC_DTM_PITISCUS_M1149280834_2M.TIF`, GSD 2.00 m/px)  
**Target Product:** Chandrayaan-2 TMC-2 Calibrated Image (`ch2_tmc_ncn_20230130T1900132182_d_img_d32`, GSD 4.72 m/px)  
**Reference Georeferencing Provenance:**  
- `lola_avg`: 0.71 m  
- `lola_rms`: 0.92 m  
- `adjust_rms`: 0.75 m  
- `relat_le`: 1.079 m  
- `triang_rms`: 0.094  
- `num_profil`: 21.0 LOLA tracks  
- `conv_angle`: 24.16 deg  

---

## 1. Vector Field Median Filter (L1)

To detect and remove isolated rim-hopping mismatches, a local vector median filter ($k = 8$ nearest accepted neighbors) was evaluated on the 324 Stage K Phase Congruency accepted displacement vectors.

### Neighborhood Median Deviation Statistics
- **Minimum Deviation:** 6.83 m (1.45 common-grid px)
- **Maximum Deviation:** 232.77 m (49.32 common-grid px)
- **Median Deviation:** 103.99 m (22.03 common-grid px)
- **Mean Deviation:** 101.88 m (21.58 common-grid px)
- **Median Absolute Deviation (MAD):** 36.05 m
- **Filtering Threshold ($3 \times \text{MAD}$):** **108.16 m** (22.92 common-grid px)

### Distribution of Deviations (L1e)
- **Percentiles:** p10 = 41.11 m, p25 = 66.27 m, p50 = 103.99 m, p75 = 136.38 m, p90 = 166.12 m, p95 = 180.05 m
- **Histogram (10 equal-width bins between 6.8 m and 232.8 m):**  
  Counts: `[20, 35, 46, 45, 64, 47, 33, 26, 7, 1]`  
  Bin Edges (m): `[6.8, 29.4, 52.0, 74.6, 97.2, 119.8, 142.4, 165.0, 187.6, 210.2, 232.8]`
- **Modality Diagnostic:** The distribution is **unimodal** with a broad single peak between 75 m and 140 m. There is no bimodal separation between a tight coherent cluster ($< 10\text{ m}$) and rim-hopping outliers.

### Performance Before and After L1 Filtering (L1d)

| Metric | Before L1 (Stage K Input) | After L1 ($3 \times \text{MAD} \le 108.16\text{ m}$) | Change |
| :--- | :--- | :--- | :--- |
| Accepted Nodes | 324 / 1,120 (28.93%) | 177 / 1,120 (15.80%) | -147 nodes (-45.4%) |
| Fit Model | 4-DoF Similarity | 4-DoF Similarity | - |
| Fitted Scale | 1.00465 | 0.99511 | -0.95% |
| Fitted Rotation | 0.4520 deg | 0.1574 deg | -0.29 deg |
| Fitted Translation | (124.30 m, -115.62 m) | (50.40 m, 49.35 m) | Shifted |
| Displacement RMSE (Common Grid) | 122.53 m (25.96 px) | 91.83 m (19.46 px) | -30.70 m (-25.1%) |
| Displacement RMSE (Reference Frame) | 61.26 reference px | 45.91 reference px | -15.35 reference px |
| Median Residual | 109.98 m (23.30 px) | 82.04 m (17.38 px) | -27.94 m |
| 95th Percentile (p95) Residual | 202.52 m (42.91 px) | 152.25 m (32.26 px) | -50.27 m |
| 8x8 Spatial Occupancy | 48 / 64 (75.0%) | 44 / 64 (68.8%) | -4 cells (-6.2%) |
| Coefficient of Variation (CV) | 0.80 | 0.96 | +0.16 |

---

## 2. Global RANSAC on the Displacement Field (L2)

RANSAC was evaluated over the 324 displacement vectors to fit a 4-DoF similarity transform ($x_{\text{ref}} = s \cos\theta \cdot x + s \sin\theta \cdot y + t_x$, $y_{\text{ref}} = -s \sin\theta \cdot x + s \cos\theta \cdot y + t_y$).

### Threshold Sweep (50 m down to 1 m)

| Inlier Threshold | Inlier Count | Inlier Ratio (of 324) | Inlier Ratio (of 1,120) | Inlier RMSE (m) | Inlier RMSE (Ref Px) | Fitted Scale | Fitted Rotation | Translation ($t_x, t_y$) | Condition Number |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50.0 m (10.59 px) | 54 | 16.67% | 4.82% | 29.93 m | 14.97 px | 1.00355 | 0.4668 deg | (135.07 m, -98.75 m) | 1.0000 |
| 25.0 m (5.30 px) | 23 | 7.10% | 2.05% | 14.49 m | 7.25 px | 1.00470 | 0.4435 deg | (124.08 m, -120.14 m) | 1.0000 |
| 10.0 m (2.12 px) | 10 | 3.09% | 0.89% | 5.13 m | 2.57 px | 1.00459 | 0.4527 deg | (124.04 m, -113.68 m) | 1.0000 |
| 5.0 m (1.06 px) | 7 | 2.16% | 0.62% | 1.83 m | 0.92 px | 1.00442 | 0.4454 deg | (124.12 m, -111.83 m) | 1.0000 |
| 2.0 m (0.42 px) | 5 | 1.54% | 0.45% | 0.62 m | 0.31 px | 1.00432 | 0.4482 deg | (124.32 m, -111.68 m) | 1.0000 |
| 1.0 m (0.21 px) | 4 | 1.23% | 0.36% | 0.36 m | 0.18 px | 1.00444 | 0.4555 deg | (124.55 m, -112.55 m) | 1.0000 |

### Spatial Distribution of the 10 m Inliers
The 10 inliers surviving at the 10.0 m threshold are distributed longitudinally down the center of the swath:
- (grid_x = 872, grid_y = 288): $\Delta x = -130.5\text{ m}, \Delta y = +74.1\text{ m}$, peak = 0.835
- (grid_x = 422, grid_y = 762): $\Delta x = -103.5\text{ m}, \Delta y = +81.2\text{ m}$, peak = 0.570
- (grid_x = 358, grid_y = 921): $\Delta x = -96.3\text{ m}, \Delta y = +80.9\text{ m}$, peak = 0.569
- (grid_x = 739, grid_y = 2029): $\Delta x = -62.9\text{ m}, \Delta y = +42.6\text{ m}$, peak = 0.653
- (grid_x = 581, grid_y = 2899): $\Delta x = -30.7\text{ m}, \Delta y = +30.4\text{ m}$, peak = 0.503
- (grid_x = 543, grid_y = 3532): $\Delta x = -1.6\text{ m}, \Delta y = +24.5\text{ m}$, peak = 0.569
- (grid_x = 598, grid_y = 4482): $\Delta x = +27.7\text{ m}, \Delta y = -6.4\text{ m}$, peak = 0.519
- (grid_x = 642, grid_y = 4720): $\Delta x = +47.8\text{ m}, \Delta y = -17.0\text{ m}$, peak = 0.883
- (grid_x = 505, grid_y = 5274): $\Delta x = +59.0\text{ m}, \Delta y = -25.0\text{ m}$, peak = 0.464
- (grid_x = 608, grid_y = 5353): $\Delta x = +61.2\text{ m}, \Delta y = -19.4\text{ m}$, peak = 0.478

All 10 inliers cluster within a corridor $x \in [358, 872]$ along the central north-south trajectory directly tracking the Pitiscus lobate scarp tectonic fault ridge.

---

## 3. Scale, Rotation, and Anisotropy Sanity (L3)

- **Fitted Scale:** $s = 1.00459$ at 10 m threshold (deviation from unity: $+0.46\%$, consistent with topographic relief differences between the DTM ellipsoid and the camera trajectory).
- **Fitted Rotation:** $\theta = 0.4527\text{ deg}$ (less than $0.5^\circ$, consistent with minor uncompensated spacecraft yaw drift).
- **Condition Number:** Exactly $1.0000$ (algebraic consequence of the conformal similarity parameterization $A = \begin{bmatrix} a & -b \\ b & a \end{bmatrix}$ where singular values $\sigma_1 = \sigma_2 = s = 1.00459$).
- **Sanity Verification:** The geometric model fitted to the coherent subset is physical and well-conditioned.

---

## 4. Control Validation of the Filtered Pipeline (L4)

All seven controls were processed through the exact same two-stage filtering pipeline (Phase Congruency correlation $\rightarrow$ L1 vector median filter $\rightarrow$ L2 RANSAC threshold sweep).

### Survival of Controls Across Pipeline Stages

| Configuration | Pre-Filter Accepted (Stage K) | Pre-Filter Ratio | Post-L1 Filter Accepted | Post-L1 Ratio | RANSAC 10m Inliers | RANSAC 10m Ratio | RANSAC 5m Inliers | RANSAC 5m Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Genuine Image | 324 | 28.93% | 177 | 15.804% | 10 | 0.89% | 7 | 0.62% |
| Control: Rot 90 | 197 | 17.59% | 145 | 12.95% | 15 | 1.34% | 12 | 1.07% |
| Control: Rot 180 | 318 | 28.39% | 158 | 14.11% | 12 | 1.07% | 6 | 0.54% |
| Control: Rot 270 | 239 | 21.34% | 183 | 16.34% | 16 | 1.43% | 12 | 1.07% |
| Control: V-Flip | 254 | 22.68% | 68 | 6.07% | 9 | 0.80% | 6 | 0.54% |
| Control: H-Flip | 290 | 25.89% | 105 | 9.38% | 9 | 0.80% | 6 | 0.54% |
| Control: Offset (500 px) | 283 | 25.27% | 111 | 9.91% | 10 | 0.89% | 6 | 0.54% |
| Control: Uniform Noise | 0 | 0.00% | 0 | 0.00% | 0 | 0.00% | 0 | 0.00% |

### Critical Finding on Rot180 and Controls (L4c)
- **Rot180 Survival:** In L1, rot180 retained 158 nodes (14.11% of grid), directly comparable to genuine's 177 nodes (15.804%). In L2 RANSAC at 10 m, rot180 produced **12 consensus inliers (1.07%)**, **higher** than genuine's 10 inliers (0.89%).
- **Other Controls:** Rot90 produced 15 inliers (1.34%), Rot270 produced 16 inliers (1.43%), and Offset produced 10 inliers (0.89%).
- **Post-Filter Kolmogorov-Smirnov Tests Against Genuine (L2 at 10 m):**
  - Rot90 vs Genuine: $D = 0.3000$, $p = 0.593$
  - Rot180 vs Genuine: $D = 0.3833$, $p = 0.318$
  - Rot270 vs Genuine: $D = 0.5125$, $p = 0.053$
  - V-Flip vs Genuine: $D = 0.2667$, $p = 0.787$
  - H-Flip vs Genuine: $D = 0.2444$, $p = 0.874$
  - Offset vs Genuine: $D = 0.4000$, $p = 0.418$
  All controls produce post-filter correlation distributions statistically indistinguishable from genuine ($p > 0.05$).

### Shuffle Margin ($\Delta_{\text{shuffle}}$) Across Stages

$$\Delta_{\text{shuffle}} = \text{Ratio}_{\text{genuine}} - \max(\text{all control ratios})$$

- **Pre-Filter (Stage K):** $28.93\% - 28.39\% = \mathbf{+0.54\%}$
- **Post-L1 Median Filter:** $15.80\% - 16.34\% = \mathbf{-0.54\%}$
- **Post-L2 RANSAC (10 m):** $0.89\% - 1.43\% = \mathbf{-0.54\%}$
- **Post-L2 RANSAC (5 m):** $0.62\% - 1.07\% = \mathbf{-0.45\%}$

Geometric consistency filtering causes the shuffle margin to go **negative** ($\Delta_{\text{shuffle}} = -0.54\%$).

---

## 5. Diagnostic Conclusion (L6)

Per L6 instructions:
> "Report whether the failure is: no coherent subset exists at any RANSAC threshold, or a subset exists but controls produce an equally coherent one. These are different findings and the distinction matters. Then stop."

**The experimental finding is definitive:**  
A coherent subset **does** exist on genuine data (10 nodes at 10 m threshold with RMSE = 5.13 m, 7 nodes at 5 m threshold with RMSE = 1.83 m, aligned along the linear Pitiscus lobate scarp with scale = 1.0044 and rotation = 0.45 deg).  
**However, rotated and shifted controls produce an equally coherent (and slightly larger) consensus subset:**
- Rot270 produces 16 inliers with RMSE = 4.41 m
- Rot90 produces 15 inliers with RMSE = 4.88 m
- Rot180 produces 12 inliers with RMSE = 4.97 m
- Spatial Offset produces 10 inliers with RMSE = 4.79 m

### Physical Mechanism
Across lunar highland terrain containing hundreds of circular impact craters per square kilometer, drawing random spatial correspondences produces 200–300 candidate vectors. In any spatial field of 200–300 vectors distributed across a narrow strip, random alignment produces 10–16 vectors that satisfy a 4-DoF similarity within 10 m purely by combinatorial chance. Because controls achieve equal or higher consensus counts, geometric consistency filtering cannot distinguish genuine lunar terrain correspondences from chance crater alignments.

---

## 6. Deliverable Artifacts Generated

- **10m Inliers CSV:** `results/stage_l_inliers_10m.csv`
- **10m Inliers GeoJSON:** `results/stage_l_inliers_10m.geojson`
- **Filtered Quiver Displacement Plot:** `results/figures/stage_l_quiver_plot_filtered.png`
- **Filtered Peak Correlation Map:** `results/figures/stage_l_peak_map_filtered.png`
- **Filtered Displacement GeoTIFF:** `results/stage_l_displacement_field_filtered.tif`
- **Results Cache:** `assets/real_cache/stage_l_geometric_filter_results.json`

---

## 7. Recommended Slide Summary Sentence

> "Applying vector median and RANSAC geometric consistency filtering to the phase congruency field isolated a coherent 10-node subset along the Pitiscus lobate scarp with 5.1 m residual RMSE, but rotated and spatially offset controls formed equally coherent consensus subsets (12 to 16 inliers at 4.4 to 5.0 m RMSE), confirming that 2D geometric filtering on repetitive crater morphology cannot eliminate combinatorial chance consensus (post-filter shuffle margin remains negative at -0.54%)."

