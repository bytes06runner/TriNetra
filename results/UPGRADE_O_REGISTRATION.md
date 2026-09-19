# Stage O: Chandrayaan-2 TMC-2 to LROC NAC Orthophoto Registration at Pitiscus

**Date:** 2026-09-19
**Runner:** `scripts/evaluate_stage_o.py`
**Package:** `src/lunar_reg/` (grid, dense NCC, models, held-out validation, products)
**Results cache:** `assets/real_cache/stage_o_pitiscus_registration.json`
**Products:** `results/stage_o/`
**Tests:** `tests/test_lunar_reg.py`

---

## 0. Summary

This is the project's first Chandrayaan-2 to lunar-reference registration that passes the full negative-control battery.

| Item | Value |
|:---|:---|
| Source (moving) | Chandrayaan-2 TMC-2 `ch2_tmc_ncn_20230130T1900132182_d_img_d32`, 4.72 m/px |
| Reference (fixed) | LROC NAC DTM orthophoto `NAC_DTM_PITISCUS_M1149280834_2M.TIF`, 2 m/px, LOLA-controlled (`lola_rms` 0.92 m) |
| Sun azimuth (local) | TMC-2 60.87 deg, NAC 342.55 deg, separation 78.32 deg; elevation 27.59 vs 38.42 deg |
| Scale gap | 2.36x (native), evaluated on the 4.72 m TMC-2 grid |
| Nodes evaluated / accepted | 1,781 / 1,713 (96.2%), mean NCC peak 0.681 |
| Model | Second-order polynomial (poly2), selected by held-out error |
| Consensus | 1,607 inliers at 1.5 px (90.23% of nodes evaluated) |
| In-sample RMSE | 0.634 px = 2.99 m |
| **Held-out, spatial blocks (5 along-track blocks, extrapolating)** | **RMSE 1.00 px (4.73 m), median 0.69 px; 66.6% within 1 px, 91.8% within 2 px, 96.4% within 3 px** |
| Held-out, random 5-fold (interpolating) | RMSE 0.76 px (3.60 m), median 0.51 px; 82.5% within 1 px, 98.2% within 3 px |
| Controls (all seven) | 0 inliers each; acceptance at most 0.11% |
| **Delta_shuffle** | **+90.23 pp** (gate +15 pp: PASS) |
| Independent closure (Stage N estimator on the product) | residual shift -0.23 to +0.86 px, PSR 182-319, 5 tiles |
| Uniformity | inliers in 54 of the 56 grid cells that contain nodes (96.4%), CV 0.41 |

**What these numbers are.** They are residuals against the NAC orthophoto, not errors against ground truth. The reference carries `lola_rms` 0.92 m (0.19 TMC-2 px) of its own. No figure here is claimed to be better than that, and none is claimed as sub-pixel accuracy for cross-sensor registration (see section 6).

---

## 1. Why this works now when Stages J-L did not

Stage N showed that the Stage J2c bulk offset was wrong by 600 px (2.83 km). Stages K and L searched ±32 px around the wrong place, so no node could contain its true match. Stage O seeds the same kind of dense correlation with the Stage N estimate, (+87.65, +799.51) px, read from the Stage N results file, and with nothing else changed, raw NCC acceptance rises from 0.18% (Stage K Arm 1) to 96.2%.

The 78 deg sun-azimuth difference did not stop intensity correlation. In the checkerboard figure the same craters carry opposite shading in the two images, yet 64 px NCC templates still peak at 0.68 on average. Phase congruency, which Stages K-M relied on, is not used here.

A second correction: the Stage J-M grid sampled the NAC orthophoto without rasterio's half-pixel centre offset, placing it 1 m east and 1 m south of the stated map point (0.21 px on this grid). `src/lunar_reg/pitiscus_grid.py` fixes this. The GeoTIFF check in section 4 confirms the fix.

---

## 2. Method (pre-declared in the runner docstring)

1. **Common grid.** Equirectangular Moon (R = 1,737,400 m, standard parallel -51 deg, CM 180 deg), 4.72 m. NAC is sampled at exact map points. TMC-2 is sampled through the ISRO geometry CSV (0-based Scan/Pixel), without a DEM.
2. **Prior.** The Stage N whole-strip phase-correlation offset. The runner refuses to start unless Stage N marked that estimate significant.
3. **Dense NCC.** 48 px node lattice, 64 px template on TMC-2, ±32 px search on NAC around node + prior, separable parabolic peak refinement. A node is accepted if peak ≥ 0.5, second peak < 0.9 × peak, and the peak is not on the search edge.
4. **Model selection.** Translation, similarity, affine, poly2 and poly3 were each fitted by seeded RANSAC (1.5 px) plus least squares. The winner has the lowest spatial-block held-out RMSE (trimmed at 3 px), with the simpler model preferred within 0.05 px.
5. **Controls.** All seven replace the NAC reference under the same prior and settings: rot90 and rot270 (blockwise on 1339 px square blocks, because the strip is not square), rot180, vflip, hflip, a non-overlapping offset (NAC rolled 2,000 rows = 9.4 km along-track), and uniform noise inside the NAC valid mask.
6. **Products.** A registered GeoTIFF (TMC-2 warped into the NAC map frame by inverting the model through fixed-point iteration; final update 3.1e-7 px), a tie-point CSV, an inlier GeoJSON, and figures.

---

## 3. Model selection

| Model | In-sample inliers @ 1.5 px | In-sample RMSE | Spatial-block held-out RMSE | Within 1 px | Within 3 px | Random 5-fold held-out RMSE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| translation | 124 | 0.983 px | 1.944 px | 0.02 | 0.12 | 1.955 px |
| similarity | 172 | 1.033 px | 1.973 px | 0.01 | 0.10 | 1.833 px |
| affine | 1,042 | 0.630 px | 1.932 px | 0.07 | 0.39 | 0.876 px |
| **poly2 (chosen)** | **1,607** | **0.634 px** | **1.002 px** | **0.67** | **0.96** | **0.762 px** |
| poly3 | 1,697 | 0.581 px | 1.269 px | 0.48 | 0.89 | 0.618 px |

poly3 fits better in-sample and under random folds, but does worse when it has to extrapolate across along-track blocks. That is overfitting, and spatial-block validation is what caught it. Only 124 nodes agree with a pure translation at 1.5 px: the displacement field between the rasters has a clear linear and quadratic component, consistent with the 0.3 deg rotation Stage N saw.

Per-fold spatial-block RMSE (trimmed): 1.175, 0.756, 1.066, 0.806, 1.160 px. The two end blocks extrapolate furthest; their fractions within 3 px are 0.924 and 0.895.

---

## 4. Checks on the result itself

| Check | Result |
|:---|:---|
| Seven-control battery | 0 inliers for every control; genuine 1,607 |
| Independent estimator (Stage N phase correlation, numpy only, no shared code) on the registered product vs the NAC grid | residual shifts (-0.23, -0.07), (-0.17, -0.21), (+0.86, +0.39), (-0.03, -0.06), (+0.41, -0.20) px |
| GeoTIFF georeferencing: NAC re-sampled onto the product grid by `rasterio.warp.reproject` (independent of the Stage O grid code) | residuals (-0.24, -0.08), (-0.18, -0.22), (+0.86, +0.39), (-0.03, -0.07) px: same as the internal closure to within 0.02 px, so no half-pixel error in the product |
| Visual | `results/stage_o/figures/stage_o_checkerboard.png`: crater rims continue across checkerboard tiles after registration and not before |

The tile at (248, 3072) keeps a +0.86 px residual after registration. The spatial held-out map (`stage_o_residual_maps.png`) shows a band of larger residuals around grid x 700-900, y 2,300-2,800. A single global quadratic cannot follow the local distortion there. Relief displacement is a likely candidate: the TMC-2 geometry has no DEM and the scarp has relief. That is not yet tested.

---

## 5. Products

| File | Content |
|:---|:---|
| `results/stage_o/registered_tmc2_pitiscus.tif` | TMC-2 resampled into the NAC map frame, 4.72 m, float32, deflate, NAC CRS; tags carry source, reference, model and `lola_rms` |
| `results/stage_o/tiepoints_tmc2_nac_pitiscus.csv` | 1,713 accepted nodes: TMC-2 raw line/sample (0-based, in the `.img`), grid coordinates, NAC map x/y, lon/lat, NAC native pixel, NCC peak and second peak, fit residual, inlier flag, spatial fold, held-out residual |
| `results/stage_o/tiepoints_tmc2_nac_pitiscus_inliers.geojson` | 1,607 inliers, lon/lat on the Moon sphere |
| `results/stage_o/figures/stage_o_checkerboard.png` | before / after checkerboard and colour overlay, 512 px tile |
| `results/stage_o/figures/stage_o_residual_maps.png` | in-sample and spatial-block held-out residual maps |

---

## 6. What may and may not be claimed

**May be claimed.**
- A Chandrayaan-2 TMC-2 image is registered to a LOLA-controlled LROC NAC orthophoto across a 78 deg sun-azimuth difference and a 2.36x resolution difference. The registration passes all seven negative controls with Delta_shuffle +90.23 pp.
- Held-out residuals against that reference: 1.00 px RMSE (4.73 m) when extrapolating across along-track blocks, and 0.76 px (3.60 m) when interpolating. Both exclude the 3.6% (spatial) and 1.8% (random) of held-out nodes beyond 3 px, which are reported separately.
- A registered product and 1,713 tie points in both the source product's own line/sample coordinates and the reference map coordinates.

**Must not be claimed.**
- Ground-truth accuracy. The reference has `lola_rms` 0.92 m, and a residual against it is not an error against the Moon.
- Sub-pixel accuracy for cross-sensor registration (CLAUDE.md hard rule). The random-fold held-out RMSE is below 1 TMC-2 px, but the spatial-block figure is 1.00 px and a residual is not an accuracy. Whether this rule should be revisited given held-out evidence is a decision for the team, not something this report settles.
- Generality. This is one pair at one site. The prior came from a strip-scale phase correlation that happened to work here; other pairs need the same checks.
- Anything about OHRC or IIRS. They are not part of this result.

---

## 7. Next steps this result opens

1. Local model: piecewise or thin-plate refinement over the poly2 residual field, validated by the same spatial-block scheme, to test whether the x 700-900 band is relief.
2. DEM-aware TMC-2 geometry: apply relief displacement from the NAC DTM (`NAC_DTM_PITISCUS`) and check whether the residual band shrinks.
3. Second site: repeat the whole Stage N + O chain on an independent TMC-2 / NAC DTM pair before any general claim.
4. Replace the in-code 2,000-row roll control with a true non-overlapping NAC product once a second site exists.
