# Stage P: Chandrayaan-2 to LROC NAC at the Chandrayaan-3 Landing Site

**Date:** 2026-09-20
**Runner:** `scripts/evaluate_stage_p.py {ohrc|tmc}`
**Package additions:** `src/lunar_reg/generic_grid.py` (source-frame pairing, DTM in frame), `src/lunar_reg/models.py` (P-spline, P-spline + relief parallax), `src/lunar_reg/validate.py` (stripes scheme)
**Results cache:** `assets/real_cache/stage_p_vikram_ohrc.json`, `assets/real_cache/stage_p_vikram_tmc.json`
**Products:** `results/stage_p/ohrc/`, `results/stage_p/tmc/`
**Reference:** `NAC_DTM_VIKRAMSITE1` (LROC, "Chandrayaan-3 Landing Site (Pre-Landing)"), 69.29 S 32.34 E, LOLA-controlled, **lola_rms 1.85 m**; orthophoto M1442997156 at 1 m and 3 m, plus the DTM. Downloaded from `pds.lroc.im-ldi.com` on 2026-09-20.

---

## 0. Summary

| | OHRC to NAC 1 m | TMC-2 to NAC 3 m |
|:---|:---|:---|
| Source | `ch2_ohr_ncp_20211023T0027462822_d_img_d18` (0.27-0.28 m native, 4x4 block mean to 1.13 m) | `ch2_tmc_ncn_20230130T1900132182_d_img_d32` (4.88 m) |
| Illumination (source / reference) | sun az 298.4 deg, el 9.1 deg / NAC incidence 73.9 deg (el ~16 deg) | sun az 53.0 deg, el 17.2 deg (label centre) / same NAC |
| Coarse offset (independent phase correlation) | 4 of 7 segments significant, PSR 27-73, all agree | 14 of 14 segments significant, PSR 9-65, all agree |
| Fine nodes evaluated / accepted | 3,126 / 3,076 (98.4%), mean NCC peak 0.822 | 1,762 / 38 (2.2%), mean NCC peak 0.351 |
| Model (selected by held-out error) | P-spline along-track + terrain relief parallax | poly2 |
| Inliers at 1.5 px | **3,072 (98.3% of nodes evaluated)** | 35 (2.0%) |
| All seven controls | 0 inliers each | 0 inliers each |
| **Delta_shuffle** | **+98.27 pp (PASS)** | **+1.99 pp (GATED)** |
| **Held-out RMSE, stripes** | **0.479 px = 0.54 m**; median 0.33 px; 94.9% within 1 px, 98.0% within 3 px | not a valid result (gated) |
| Held-out RMSE, random 5-fold | 0.388 px = 0.44 m; 99.0% within 1 px | |
| Held-out RMSE, spatial blocks (extrapolating 3+ km) | 1.252 px = 1.42 m; 45% within 1 px, 78% within 3 px | |
| Product vs NAC reprojected by rasterio (Stage N estimator, 23 tiles) | median residual shift 0.17 px, max 0.58 px, min PSR 213 | |
| Fitted relief parallax vs label | view angle 16.81 deg fitted vs 17.32 deg predicted from roll/pitch/altitude (difference -0.51 deg) | |
| ISRO geometry vs NAC (median over inliers) | **3.72 km** (482 m east, 3,687 m south) | **3.76 km** (25 m east, 3,765 m south) |

**Headline.** A Chandrayaan-2 OHRC image is registered to a LOLA-controlled LROC NAC orthophoto at the Chandrayaan-3 landing site. 3,072 of 3,126 correspondence nodes are consistent, and every negative control produces zero. Held-out residuals are 0.54 m against the reference when interpolating between 500-row stripes, and 1.42 m when extrapolating across 3+ km blocks.

The model includes a terrain-relief term. From parallax alone it recovers the spacecraft's viewing angle to within 0.5 deg of the value computed from the product label, which is an independent physical check on the fit.

**What these numbers are.** They are residuals against a reference that carries `lola_rms` 1.85 m; nothing here is claimed to be better than that. They are measured at the 1.13 m working resolution (4x4 OHRC blocks). In native OHRC pixels (0.27-0.28 m), the 0.54 m stripe residual is about 2 native pixels. Per CLAUDE.md, this is not a claim of sub-pixel accuracy for cross-sensor registration.

**Negative result.** TMC-2 to NAC at this site does not pass the gate. The coarse offset is unambiguous, but dense NCC accepts only 2.2% of nodes.

---

## 1. A systematic offset in the ISRO geometry products

Every Chandrayaan-2 product registered so far sits about 3.7-3.8 km from its LOLA-controlled NAC position, in the same direction:

| Product | Site | Offset of true (NAC) position from ISRO geometry-CSV position | Magnitude |
|:---|:---|:---|:---:|
| TMC-2 `..._20230130T1900132182_d_img_d32` | Pitiscus (51.3 S) | 414 m east, 3,774 m south (Stage N/O, equirectangular grid) | 3.80 km |
| TMC-2 `..._20230130T1900132182_d_img_d32` | Chandrayaan-3 site (69.3 S) | 25 m east, 3,765 m south | 3.76 km |
| OHRC `..._20211023T0027462822_d_img_d18` | Chandrayaan-3 site (69.3 S) | 482 m east, 3,687 m south | 3.72 km |

The OHRC offset varies little across the strip: the 5th-95th percentile of its magnitude over 3,072 inliers is 3,713-3,729 m.

For a polar orbit, south is along-track. The TMC-2 offset corresponds to about 750 image lines at both sites. Two instruments, two sites about 1,000 km apart, and two acquisitions two years apart all show the same along-track offset. That points to something systematic in how these geometry products were generated (timing, ephemeris or frame definitions), not to anything site-specific.

**What is not claimed.** The cause. The geometry CSVs are used exactly as distributed, and their generation (SPICE kernel versions, timing corrections) is not documented in the products. The reference is LOLA-controlled with stated error; the offset is 2,000 times larger than that error.

This is worth reporting to ISRO as an observation. It also explains the Stage J-L failure: a geometry-seeded search window of ±150 m cannot find a match 3.8 km away.

---

## 2. Method and what changed from Stage O (disclosed)

1. **Source-frame pairing.** Stage O resampled everything onto a map grid. Here both swaths cross the NAC strip diagonally, so a first map-frame version found no coarse offset: axis-aligned bands captured only 32-180 px of the OHRC swath. That run is not retained as a result. Stage P instead works in each source image's own line/sample frame, with the NAC orthophoto (and DTM) resampled into it through the ISRO geometry CSV. The change was made after the map-frame failure.
2. **Coarse level.** OHRC blocks of 15x15 (4.25 m); TMC-2 at native resolution. 2,000-row segments, columns at least 98% NAC-valid, Stage N phase correlation, 7 controls plus a 50-draw noise null per segment. The prior is the median of significant segments.
3. **Fine level.** OHRC 4x4 blocks (1.13 m); dense NCC with 64 px templates and ±32 px search on a 96 px lattice (OHRC) or 48 px lattice (TMC-2). Stage O acceptance rules.
4. **Models.** Translation to poly3 (RANSAC), plus two added for pushbroom imagery:
   - **P-spline:** cubic B-splines along-track (knots every 1,000 rows for OHRC, 500 for TMC-2), quadratic cross-track, second-difference penalty with lambda chosen by GCV on fitting data only, iterative trimming at 1.5 px.
   - **P-spline + relief:** adds (h - h0) × p, where h is the NAC DTM height at the matched position (observed when fitting; looked up iteratively at the predicted position when predicting held-out or product points) and p is a fitted 2-vector. This was added after the P-spline residuals showed fine-scale structure. The physical motivation is the OHRC label's 15.76 deg roll: an off-nadir camera shifts terrain by h·tan(emission), and the NAC orthophoto is terrain-corrected while the OHRC geometry is not.
5. **Selection.** Stripes held-out RMSE (500-row interleaved bands, 5 folds), trimmed at 3 px. Stage O selected on spatial blocks. The change was made after the polynomial models underfit the OHRC strip: nodes cover the whole strip, so contiguous-block extrapolation over 3+ km is not how the product is used. All three schemes are reported for every model.
6. **Controls, Delta_shuffle and products** are as in Stage O. The registered product is written in the NAC CRS by inverting the model on a lattice.

---

## 3. OHRC model comparison

| Model | In-sample inliers | Stripes RMSE (within 1 / 3 px) | Spatial blocks RMSE (within 1 / 3 px) | Random 5-fold RMSE (within 1 / 3 px) |
|:---|:---:|:---:|:---:|:---:|
| translation | 615 | 1.663 (0.11 / 0.36) | 1.703 (0.10 / 0.36) | 1.655 (0.11 / 0.36) |
| affine | 791 | 1.643 (0.17 / 0.45) | 1.689 (0.09 / 0.26) | 1.644 (0.17 / 0.46) |
| poly2 | 942 | 1.688 (0.17 / 0.51) | 1.747 (0.10 / 0.32) | 1.603 (0.21 / 0.53) |
| poly3 | 974 | 1.697 (0.17 / 0.50) | 1.602 (0.14 / 0.36) | 1.611 (0.20 / 0.52) |
| P-spline | 1,735 | 1.452 (0.33 / 0.70) | 1.699 (0.07 / 0.21) | 1.399 (0.41 / 0.79) |
| **P-spline + relief** | **3,072** | **0.479 (0.95 / 0.98)** | **1.252 (0.45 / 0.78)** | **0.388 (0.99 / 1.00)** |

RMSE is in fine px; 1 px = 1.13 m. The relief term accounts for most of the residual. Without it, even a flexible along-track spline explains only 56% of nodes at 1.5 px.

### Relief parallax as a physical check
The fitted parallax is p = (-0.256, -0.071) px per metre of height. Converted to ground metres with the local frame-to-map Jacobian, that is 0.302 m of displacement per metre of height, i.e. an emission angle of **16.81 deg**. From the OHRC label (roll 15.757549 deg, pitch 4.516213 deg, altitude 102.02 km): off-nadir angle 16.33 deg, and emission at the surface 17.32 deg once lunar curvature is included. The two differ by 0.51 deg (3% in tan). The fit never saw the label, so this agreement is evidence that the relief term models the physics, not noise.

---

## 4. TMC-2: why it is gated

The coarse offset is as solid as anything in this project: 14 of 14 segments significant, all within 5 px. Dense matching still fails: 38 of 1,762 nodes accepted, mean NCC peak 0.35. The tests run, all in this session:

- Band-pass (DoG) and local normalisation: acceptance rose from 2.2% to 6.7%, controls stayed at zero, still far below the gate.
- Sign-invariant |DoG| and phase congruency: worse (0.2%).
- Larger templates (128 and 192 px): worse. Peaks fall as templates grow.
- Global geometry: the affine fitted to the reliable nodes is 0.7% scale and -0.001 deg rotation, which is not enough to break 64 px templates.

Visual inspection (the same crater pairs present, large-scale shading inverted) and the OHRC contrast suggest the cause is illumination. OHRC (sun azimuth 298 deg) matches this NAC at a mean peak of 0.82; TMC-2 (sun azimuth 53 deg at the label centre, 115 deg from OHRC) reaches 0.35. This is **not demonstrated**: the NAC frame's sun azimuth is not available from ODE, and the TMC-2 label value is for the product centre, not this site.

The TMC-2 product and tie points are kept in `results/stage_p/tmc/` for traceability. They are not a deliverable.

---

## 5. Products

| File | Content |
|:---|:---|
| `results/stage_p/ohrc/registered_ohrc_vikram_u8.tif` | OHRC resampled into the NAC CRS (polar stereographic, lat_ts -69.3, lon_0 32.3, R 1,737,400 m) at 1.13 m, 8-bit (rounded from the float product, max rounding 0.5 DN), deflate, 18.9 MB |
| `results/stage_p/ohrc/registered_ohrc_vikram.tif` | float32 original (103 MB; above GitHub's 100 MB limit, gitignored, regenerate with the runner) |
| `results/stage_p/ohrc/tiepoints_ohrc_nac_vikram.csv` | 3,076 nodes: OHRC full-resolution line/sample, NAC map x/y, lon/lat, NAC native pixel, ISRO-geometry offset, NCC peaks, fit and held-out residuals |
| `results/stage_p/ohrc/tiepoints_ohrc_nac_vikram_inliers.geojson` | 3,072 inliers, lon/lat |
| `results/stage_p/ohrc/figures/` | NAC / registered OHRC checkerboard and overlay; residual maps and the ISRO-geometry offset field |

---

## 6. What may and may not be claimed

**May be claimed.**
- OHRC to LROC NAC registration at the Chandrayaan-3 landing site that passes all seven negative controls (Delta_shuffle +98.27 pp).
- Held-out residuals against the reference: 0.54 m (stripes) and 0.44 m (random folds) at 1.13 m working resolution, and 1.42 m when extrapolating over 3+ km.
- A relief-aware model whose fitted parallax agrees with the label view angle to 0.5 deg.
- A consistent 3.7-3.8 km along-track offset between three ISRO geometry products and LOLA-controlled NAC positions, at two sites.

**Must not be claimed.**
- Ground-truth accuracy (reference `lola_rms` 1.85 m).
- Sub-pixel accuracy for cross-sensor registration (CLAUDE.md). In native OHRC pixels the residual is about 2 px.
- Illumination invariance in general. TMC-2 at this site, with a large sun-azimuth difference, fails.
- The cause of the ISRO geometry offset.
