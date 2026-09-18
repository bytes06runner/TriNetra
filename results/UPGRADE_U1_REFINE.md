# TriNetra U1: Sub-Pixel Correspondence Refinement Evaluation

- **Method:** `phase_correlate`
- **Patch Size:** `15x15` px
- **Confidence Floor:** `0.3`
- **RANSAC Inlier Threshold:** `15.0` c-px

> **Methodological Summary:** Match coordinates produced by LoFTR are refined before 4-DoF similarity fitting via local patch intensity alignment. If correlation confidence falls below threshold or shift exceeds half-patch, the original coordinate is retained.

### Hop 1 (OHRC ↔ TMC-2): Fine-Tuned EfficientLoFTR (Epoch 7)

- **Matches Refined:** 161 / 217 (74.2%)
- **Matches Rejected:** 56 / 217 (25.8%)
- **Mean Sub-pixel Shift:** 1.019 px

| Metric | Before Refinement | After Refinement | Delta |
|:---|:---:|:---:|:---:|
| Inliers (RANSAC) | 49 / 217 | 48 / 217 | -1 |
| Inlier Ratio | 22.58% | 22.12% | -0.46% |
| RMSE (Canvas px) | 9.23 c-px | 9.21 c-px | -0.03 c-px |
| RMSE (Native ref px) | 2.22 n-px | 2.21 n-px | -0.01 n-px |
| RMSE (Ground m) | 9.42 m | 9.39 m | -0.03 m |
| Condition Number | 3646.5 | 3799.8 | +153.2 |
| Validity Gate | PASSED | PASSED | — |

### Hop 2 (TMC-2 ↔ IIRS): Phase Congruency + LoFTR

- **Matches Refined:** 245 / 307 (79.8%)
- **Matches Rejected:** 62 / 307 (20.2%)
- **Mean Sub-pixel Shift:** 0.850 px

| Metric | Before Refinement | After Refinement | Delta |
|:---|:---:|:---:|:---:|
| Inliers (RANSAC) | 124 / 307 | 125 / 307 | +1 |
| Inlier Ratio | 40.39% | 40.72% | +0.33% |
| RMSE (Canvas px) | 9.05 c-px | 9.04 c-px | -0.00 c-px |
| RMSE (Native ref px) | 1.36 n-px | 1.36 n-px | -0.00 n-px |
| RMSE (Ground m) | 92.78 m | 92.75 m | -0.02 m |
| Condition Number | 688.9 | 685.9 | -3.0 |
| Validity Gate | PASSED | PASSED | — |

## Discussion & Error Budget Interpretation

Sub-pixel correspondence refinement via local phase correlation adjusts reference coordinates by an average of ~0.85 to 1.02 pixels.
- On **Hop 1**, RMSE changed from 9.23 c-px to 9.21 c-px (-0.03 c-px / -0.03 m), with inliers at 48 vs 49.
- On **Hop 2**, RMSE changed from 9.05 c-px to 9.04 c-px (-0.00 c-px / -0.02 m), with inliers at 125 vs 124.
Refinement confirms that sub-pixel accuracy in native reference pixels (target < 1.0 n-px) is not bottlenecked by keypoint integer rounding; instead, the 2.21 n-px (Hop 1) and 1.36 n-px (Hop 2) residual floors reflect physical differences across the 17.7x/14.5x optical scale gap, unmodelled oblique pushbroom foreshortening, and local topography.
