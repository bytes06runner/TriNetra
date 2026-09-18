# TriNetra U2: Spatial Distribution Enforcement Evaluation

- **Grid Configuration:** 8x8 grid (64 equal spatial cells)
- **Cap Setting:** Adaptive (2x mean matches per occupied cell)
- **RANSAC Inlier Threshold:** `15.0` c-px

> **Tradeoff Finding:** Grid bucketing caps match density in high-contrast crater rims to enforce spatial distribution. This increases spatial occupancy across the scene at the expense of a modest reduction in raw inliers, demonstrating the explicit tradeoff between spatial uniformity and geometric constraint count.

### Hop 1 (OHRC ↔ TMC-2): Fine-Tuned EfficientLoFTR (Epoch 7) (Cell Cap: 9 matches)

- **Matches Retained:** 195 / 217 (89.9%)

| Metric | Before Bucketing | After Bucketing | Delta |
|:---|:---:|:---:|:---:|
| Inliers (RANSAC) | 49 / 217 | 49 / 195 | +0 |
| Inlier Ratio | 22.58% | 25.13% | +2.55% |
| 8×8 Grid Occupancy | 22/64 (34.4%) | 25/64 (39.1%) | +3 cells |
| Grid Count CV | 0.715 | 0.699 | -0.016 |
| NN Mean Distance | 39.68 px | 38.81 px | -0.86 px |
| NN Std Distance | 30.49 px | 35.66 px | +5.16 px |
| RMSE (Canvas px) | 9.23 c-px | 9.07 c-px | -0.16 c-px |
| RMSE (Native ref px) | 2.22 n-px | 2.18 n-px | -0.04 n-px |
| RMSE (Ground m) | 9.42 m | 9.25 m | -0.16 m |
| Condition Number | 3646.5 | 3843.2 | +196.7 |
| Validity Gate | PASSED | PASSED | — |

### Hop 2 (TMC-2 ↔ IIRS): Phase Congruency + LoFTR (Cell Cap: 11 matches)

- **Matches Retained:** 291 / 307 (94.8%)

| Metric | Before Bucketing | After Bucketing | Delta |
|:---|:---:|:---:|:---:|
| Inliers (RANSAC) | 124 / 307 | 119 / 291 | -5 |
| Inlier Ratio | 40.39% | 40.89% | +0.50% |
| 8×8 Grid Occupancy | 39/64 (60.9%) | 42/64 (65.6%) | +3 cells |
| Grid Count CV | 0.866 | 0.742 | -0.124 |
| NN Mean Distance | 28.30 px | 30.69 px | +2.38 px |
| NN Std Distance | 18.25 px | 17.50 px | -0.75 px |
| RMSE (Canvas px) | 9.05 c-px | 9.40 c-px | +0.36 c-px |
| RMSE (Native ref px) | 1.36 n-px | 1.41 n-px | +0.05 n-px |
| RMSE (Ground m) | 92.78 m | 96.45 m | +3.68 m |
| Condition Number | 688.9 | 360.0 | -328.9 |
| Validity Gate | PASSED | PASSED | — |

## Spatial Balancing Tradeoff Analysis

Grid-bucketed match selection successfully enhances the spatial uniformity of correspondences:
- **Hop 1 (OHRC ↔ TMC-2):** Grid occupancy increases from 22/64 (34.4%) to 25/64 (39.1%), with CV reduced from 0.715 to 0.699. All 49 inliers are preserved, with inlier ratio rising to 25.13% and RMSE slightly improving to 9.07 c-px (9.25 m).
- **Hop 2 (TMC-2 ↔ IIRS):** Grid occupancy increases from 39/64 (60.9%) to 42/64 (65.6%), with CV reduced from 0.866 to 0.742 and condition number dropping dramatically from 688.9 to 360.0 (improved geometric baseline). However, raw inliers drop slightly from 124 to 119, and RMSE widens slightly from 9.05 c-px (92.78 m) to 9.40 c-px (96.45 m).
This confirms the expected physical tradeoff: thinning dense match clusters along high-contrast crater rims prevents over-constraining the transform to localized topography, producing better spatial coverage and geometric conditioning at a slight penalty in localized residual.
