# Stage N: Independent Corroboration by Whole-Image Fourier Phase Correlation

**Date:** 2026-09-19
**Estimator:** `src/corroboration/phase_correlation.py` (new, numpy only)
**Runner:** `scripts/evaluate_stage_n.py`
**Results cache:** `assets/real_cache/stage_n_corroboration_results.json`
**Tests:** `tests/test_corroboration.py`

---

## 0. Headline: one result contradicts something we treated as settled

**The Stage J2c bulk offset at Pitiscus is not supported by independent evidence, and a different offset is.**

- Stage J2c reported an a-priori TMC-2 to NAC offset of (-82.12, +224.04) common-grid px (1,126 m). Stages K and L applied it before every correlation.
- On the full valid strip, independent phase correlation finds **(+87.65, +799.51) px = (+414 m, +3,774 m), magnitude 804 px = 3,796 m**, with PSR 54.45. The highest control PSR is 8.58 and the noise-null maximum is 8.35.
- The same offset comes back from four 2,000-row segments (dy +798.5 to +799.8, dx +69.5 to +90.8; PSR 19.2 to 22.2; every control at or below 8.56).
- A third, non-circular check (plain numpy NCC of 512 px tiles, no FFT) gives a mean raw NCC of **+0.468** (min +0.334, 8 tiles) at the phase-correlation lag. At the lag Stage K/L applied it gives **-0.035**, and the best of 30 random lags reaches +0.145. OpenCV's `cv2.phaseCorrelate` on the same strip returns (+87.01, +799.31), response 0.191.
- The gap between the two offsets is **600 px = 2,832 m**. After the Stage K1 bulk shift, the true residual would be about (+170, +575) px. That is about 19 times the Stage K search radius of 32 px (151 m).

**Consequence.** No Stage K or L correlation node could have contained its true match. That is enough on its own to explain why every Stage K/L control reached parity with genuine data (Delta_shuffle +0.54% and -0.54%). The GATED verdicts for Stages K and L stand, but the stated cause changes. The earlier write-ups name the 78.32 deg illumination azimuth difference plus crater self-similarity as the root cause. The evidence here says the windows were searching roughly 2.8 km from the right place. Raw intensity NCC reaches +0.47 at the correct lag despite the 78 deg azimuth difference, so illumination did not prevent correlation at this tile scale.

**What this does not establish.**
- It is not ground truth. The phase-correlation offset is the offset between the two rasters as built on the common grid. The reference's own error is `lola_rms` 0.92 m, and nothing here claims to be better than that. At this stage it is a tie-point-free global offset, not a registration.
- It does not say where the error comes from. The common-grid reprojection recipe (geometry CSV, scans 239000 to 247000) is shared with Stage K, so a reprojection error would affect both pipelines equally. The 3.8 km could be TMC-2 pointing or timing, or it could be in the reprojection.
- It does not explain how J2c got (-82.12, +224.04). The script that produced it is not in the repository. I tested one hypothesis: that a ~1024-row window wrapped the +799 px shift to about -225. It was not confirmed, because 1024-row windows at three positions return no significant peak (PSR 6.2 to 7.7).
- It is not sub-pixel. Tiles re-extracted at the phase-correlation lag leave residuals that drift from dx +3.50 to -18.45 px along the strip (dy -0.19 to -1.41 px). A single translation does not describe the full 27 km strip; the drift implies roughly -0.3 deg of relative rotation.
- The strip analysis was **post-hoc**. The pre-declared design (768 px tiles) found nothing because an 800 px shift leaves those tiles with no overlap at all. I added the strip analysis after seeing that. It carries the full control battery and three independent confirmations, but it was not pre-registered.

---

## 1. Why a second estimator, and what makes it independent

Every earlier result came from one pipeline: keypoint or template matching, then RANSAC similarity fitting, parabolic peak refinement and phase congruency. Noise against noise reached 80.87% consensus in that pipeline. A transform can look geometrically perfect and still be wrong. So Stage N adds a second estimator that shares no code with the first and has to agree independently.

| Property | Primary pipeline | Stage N estimator |
|:---|:---|:---|
| Correspondence | Keypoints (LoFTR / SIFT) or 64 px template NCC per node | One global cross-power spectrum over the whole image |
| Geometry fit | RANSAC 4-DoF similarity (`cv2.estimateAffinePartial2D`) | None (translation only) |
| Sub-pixel | 3x3 parabolic fit | Matrix-multiply upsampled DFT, factor 100 (Guizar-Sicairos 2008) |
| Representation | Phase congruency (Hop 2, Stages K-M) | Raw intensity, Hann window |
| Significance | Inlier ratio vs controls (Delta_shuffle) | Peak-to-sidelobe ratio (PSR) vs controls and a noise null |
| Dependencies | OpenCV, project modules | numpy only |

`tests/test_corroboration.py` enforces this. It parses the module's AST and fails on any import other than `numpy` and `typing`, and it fails if the source mentions `matchTemplate`, `estimateAffinePartial2D`, `findHomography`, `compute_phase_congruency`, `phaseCorrelate` or `evaluate_flight_gate`.

**Shared inputs (common-mode risk, stated plainly).** Both pipelines read the same cached canvases for Hop 1 and Hop 2, and the same common-grid reprojection recipe at Pitiscus. A bug in the cached canvases or in the reprojection would affect both. Stage N guards against bugs in matching, fitting and scoring, not in data preparation.

**Shift convention.** `estimate_shift(fixed, moving)` returns s such that content at p in `fixed` appears at p + s in `moving`. The primary pipeline's H is converted to the same quantity: the displacement H(c) - c at the canvas centre c. For Pitiscus it is M(c + b) + b - c, where b is the J2c bulk offset and M is the Stage K or L fit (tmc_shifted to NAC).

**Provenance.** `assert_cache_provenance()` passed for `polar_flight_hop1.npz` to `real_flight_hop1_finetuned.npz`. For Hop 2, `disp_tmc`, `disp_iirs`, `raw_iirs_crop`, `iirs_id` and `center_lat` are identical between `real_flight_hop2.npz` and `real_flight_hop2_phase_congruency.npz`.

---

## 2. Pre-declared settings

These were fixed in the runner docstring before any result was inspected.

- Hann window on both images.
- Band limit set to the coarser sensor's Nyquist on the evaluation canvas: Hop 1 0.12 cyc/px (0.5 x 240/1000), Hop 2 0.075 cyc/px (0.5 x 120/800). No band limit for Pitiscus or M2, which are evaluated on the native 4.72 m TMC-2 grid.
- Agreement tolerance: 2.0 px in the evaluation frame.
- Significance: genuine PSR must exceed (a) the maximum PSR over 200 uniform-noise references and (b) every core control. **Delta_PSR = PSR_genuine - max(core-control PSR) > 0.**
- Core controls, each replacing the reference: rot90, rot180, rot270, vflip, hflip, non-overlapping real terrain, uniform noise.

---

## 3. Results

### 3.1 Hop 1: OHRC to TMC-2, Shackleton Rim (fine-tuned LoFTR H)

Primary displacement at the canvas centre is (+2.26, +10.70) c-px, with primary scale 1.0086 and rotation -4.38 deg. The offset control is the TMC-2 canvas from `real_flight_hop1_shivshakti.npz`, a different site about 2,000 km away. 1 c-px = 1.020 m = 0.24 native TMC-2 px.

| Variant | PC (dx, dy) c-px | Gap to primary | PSR | Max control PSR | Noise-null max | Delta_PSR | Significant | Agree (2 px) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A band-limited (declared) | (+0.36, -60.30) | 71.02 c-px / 17.05 n-px / 72.4 m | 7.08 | 7.48 (rot180) | 8.64 | -0.40 | No | No |
| B full-band | (+74.00, -60.05) | 100.75 c-px / 24.18 n-px / 102.8 m | 7.68 | 7.67 (rot90) | 9.82 | +0.02 | No | No |
| C conditioned on primary scale+rotation | (-139.83, +167.43) | 211.56 c-px / 50.77 n-px / 215.8 m | 7.65 | 8.15 (vflip) | 8.77 | -0.50 | No | No |

### 3.2 Hop 2: TMC-2 to IIRS, South Pole (phase congruency + LoFTR H)

Phase correlation runs on the raw intensity canvases, not the phase-congruency maps. Primary displacement at the centre is (-1.16, +17.82) c-px, with scale 1.0580 and rotation -0.34 deg. The offset control is the IIRS proxy from `real_overlapping_pair.npz`, a north polar site, resized to 800x800. 1 c-px = 10.257 m = 0.15 native IIRS px.

| Variant | PC (dx, dy) c-px | Gap to primary | PSR | Max control PSR | Noise-null max | Delta_PSR | Significant | Agree (2 px) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A band-limited (declared) | (-68.63, -94.94) | 131.40 c-px / 19.71 n-px / 1,347.8 m | 6.09 | 6.85 (rot90) | 7.87 | -0.77 | No | No |
| B full-band | (-114.74, -2.71) | 115.42 c-px / 17.31 n-px / 1,183.8 m | 7.19 | 7.27 (offset) | 8.63 | -0.09 | No | No |
| C conditioned on primary scale+rotation | (-56.23, -241.87) | 265.47 c-px / 39.82 n-px / 2,722.9 m | 6.05 | 7.08 (vflip) | 8.77 | -1.03 | No | No |

**Reading Hops 1 and 2.** Phase correlation finds no significant peak on either hop, under any variant, including when handed the primary pipeline's own scale and rotation. Its argmax lands 71 to 265 c-px from the primary estimate, but a non-significant argmax is effectively a random location. So this is not evidence that the primary transform is wrong. It is an absence of corroboration. Neither estimator separates the genuine pair from fabricated references. That is consistent with the existing Criterion 6 GATED verdicts (Hop 1 Delta_shuffle -5.15%, Hop 2 -11.68%) and adds nothing in either direction.

**Cross-check.** `cv2.phaseCorrelate` returns (+47.88, -18.76) on Hop 1 and (+14.34, -32.72) on Hop 2, with responses 0.020 and 0.013. It does not match variant B's argmax. That is expected on surfaces at the noise floor: the argmax depends on whitening details when no peak exists. Where a real peak exists, the two agree (Pitiscus strip: +87.01, +799.31 against +87.65, +799.51).

### 3.3 Pitiscus: TMC-2 to LROC NAC orthophoto, 4.72 m common grid (pre-declared tiles)

Seven 768 px tiles, fully valid in the NAC mask. Primary estimates at the tile centres are near (-76, +218) px: J2c bulk (-82.12, +224.04); K5 bulk plus 324-node similarity (-76.13, +217.68) and L2 bulk plus 10 m RANSAC similarity (-76.26, +217.93), both at tile 3.

| Tile (x0, y0) | PC (dx, dy) px | PSR | Max control PSR | Noise-null max | Significant | Gap to J2c / K5 / L2 (px) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| (420, 0) | (-22.20, -49.83) | 6.62 | 7.41 (rot270) | 8.37 | No | 280.3 / 260.3 / 260.7 |
| (377, 768) | (+66.14, -26.93) | 7.23 | 7.71 (offset) | 8.30 | No | 291.5 / 271.2 / 271.6 |
| (333, 1536) | (-46.95, -62.11) | 7.05 | 7.63 (noise) | 9.31 | No | 288.3 / 277.5 / 277.8 |
| (289, 2304) | (+36.97, +27.79) | 7.00 | 7.14 (rot270) | 8.97 | No | 229.6 / 221.0 / 221.3 |
| (248, 3072) | (+74.13, +36.00) | 7.16 | 7.71 (rot90) | 9.58 | No | 244.5 / 242.3 / 242.5 |
| (201, 3840) | (+40.94, +52.03) | 6.51 | 7.80 (rot90) | 9.54 | No | 211.5 / 215.4 / 215.6 |
| (157, 4608) | (+43.18, -24.01) | 6.60 | 7.75 (rot270) | 8.59 | No | 277.9 / 286.8 / 286.9 |

On its own, this table would read as "Pitiscus is void, as expected". Section 3.4 shows why that reading is wrong: the true offset is larger than the tile.

### 3.4 Pitiscus: full valid strip, long segments and lag check (post-hoc)

Columns 345:985 are valid in both rasters on every row. Rotation controls on non-square strips rotate each 640x640 block, and any leftover rows are rotated 180 deg so no genuine geometry survives. The whole-strip offset control uses TMC-2 scans 150000:155825 from the same product, about 90 km along-track and zero overlap. Rolling the strip was rejected as a control, because a circular estimator still finds rolled content (it scored PSR 17.1 in a development run). Each segment's offset control is a reference segment disjoint from where that segment's content lies.

| Window | PC (dx, dy) px | PSR | Max control PSR (which) | Noise-null max | Delta_PSR | Significant | Gap to J2c bulk |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Whole strip, 5825 x 640 | (+87.65, +799.51) | 54.45 | 8.58 (hflip) | 8.35 | +45.87 | Yes | 600.0 px (2,832 m) |
| Rows 0-2000 | (+90.81, +799.77) | 20.82 | 7.72 (noise) | 8.35 | +13.10 | Yes | 601.1 px |
| Rows 1275-3275 | (+87.96, +799.38) | 19.23 | 8.09 (noise) | 8.18 | +11.14 | Yes | 600.0 px |
| Rows 2550-4550 | (+83.82, +798.96) | 22.16 | 7.52 (rot270 blockwise) | 8.96 | +14.64 | Yes | 598.4 px |
| Rows 3825-5825 | (+69.48, +798.48) | 21.05 | 8.56 (noise) | 8.38 | +12.49 | Yes | 594.1 px |

Whole-strip control PSRs: rot90 7.48, rot180 7.97, rot270 7.13, vflip 7.72, hflip 8.58, offset 7.07, noise 7.97. Noise null over 50 draws: mean 7.49, std 0.35, max 8.35.

Non-circular NCC lag check (numpy, 512 px tiles, TMC-2 tile at q against NAC tile at q + lag):

| Lag (px) | Tiles | Mean raw NCC | Min raw NCC | Mean gradient-magnitude NCC |
|:---|:---:|:---:|:---:|:---:|
| Phase correlation (+88, +800) | 8 | +0.468 | +0.334 | +0.136 |
| Stage K/L applied (-82, +224) | 16 | -0.035 | -0.137 | +0.003 |
| Zero (0, 0) | 14 | +0.045 | -0.077 | +0.001 |
| Best of 30 random lags | | +0.145 | | |

Tiles re-extracted at the phase-correlation lag (768 px, non-circular windows) all give significant peaks (PSR 37.9 to 53.3, control max 7.0 to 8.6). Residuals are small in dy (-0.19 to -1.41 px) and drift in dx (+3.50, +2.16, +0.03, -3.27, -11.42, -18.45 px going down the strip).

### 3.5 Matched-illumination positive controls (Stage M geometry)

This is where agreement is expected: the same NAC scene on both sides, so illumination matches, across the 2.36x resampling gap. Units are 4.72 m common-grid px.

| Test | Comparison | n | RMSE px (m) | Max px | All within 2 px |
|:---|:---|:---:|:---:|:---:|:---:|
| M2a, 25 Stage M trials (circular Fourier shifts) | PC vs primary (Stage M cached estimate) | 25 | 0.088 (0.415 m) | 0.135 | Yes |
| M2a | PC vs known truth | 25 | 0.0043 | | Yes |
| M2b, non-circular: separate native windows displaced by -60 to +60 native px, resampled independently | PC vs known truth | 25 | 0.048 (0.226 m) | 0.086 | Yes |

M2a control battery (trial 1): PSR 1,826.5 against max control 7.79 (vflip) and noise-null max 9.14, Delta_PSR +1,818.8. Minimum genuine PSR was 1,293.7 across the M2a trials and 751.9 across M2b.

**Scope of the positive control.** M2a uses circular shifts, which are the ideal case for phase correlation, so agreement there is close to guaranteed and weak evidence by itself. M2b removes that advantage with real window edges and independent resampling, and still recovers the known shifts. Both use one NAC scene on both sides. They validate the two estimators against each other and against known truth under matched illumination only. They say nothing about sub-pixel accuracy for cross-sensor registration, and none is claimed. The Hann window also has a known bias on circular shifts of very smooth texture (up to about 0.13 px in the unit tests; exact when unwindowed).

---

## 4. Does independent corroboration support or contradict what we concluded?

| Earlier conclusion | Stage N finding | Verdict |
|:---|:---|:---|
| Matched-illumination scale-gap recovery works (Stage M2) | PC agrees with the primary to 0.088 px RMSE and with truth to 0.048 px (non-circular) | **Supports** |
| Hop 1 LoFTR result is void (Criterion 6) | PC finds no significant peak; Delta_PSR -0.40 / +0.02 / -0.50 | **Consistent; adds no evidence either way** |
| Hop 2 PC+LoFTR result is void (Criterion 6) | PC finds no significant peak; Delta_PSR -0.77 / -0.09 / -1.03 | **Consistent; adds no evidence either way** |
| Stage J2c: bulk offset is (-82.12, +224.04) px = 1,126 m | Significant, reproducible estimate (+87.65, +799.51) px = 3,796 m, confirmed by non-circular NCC and cv2 | **Contradicts** |
| Stages K/L are GATED | Their search windows were centred about 600 px from the true match | **Verdict stands; stated cause changes** |
| Stages J/K/M: the root cause of cross-sensor failure is the 78.32 deg illumination difference plus crater symmetry | Raw NCC +0.47 at the correct lag despite the 78 deg difference | **Contradicts as stated. The illumination effect on correlation is untested at the correct alignment.** |

**Items that need correcting in existing reports** (not edited in this stage; each needs your decision):
1. `results/UPGRADE_J_DENSE.md` section 2 (J2c), and the README's Stage J item 1: the a-priori offset figure.
2. `results/UPGRADE_K_COARSE_FINE.md` K1 ("Resolved by applying the global pointing offset") and K6, which attribute the dominant remaining failure to illumination and self-similarity.
3. `results/UPGRADE_L_GEOMETRIC_FILTER.md` sections 2, 3 and 5: the 10 m "coherent subset along the scarp" was fitted to nodes whose true match lay outside the search window.
4. `results/UPGRADE_M_SUBPIXEL_TRUTH.md` section 5 and the README's Stage M conclusion ("The sole root cause of cross-sensor divergence is the 78.32 deg solar illumination azimuth disparity").

`results/metrics.json` is unchanged. It holds no Pitiscus entries.

---

## 5. Limitations

- Translation only. It cannot corroborate scale or rotation. Variant C conditions on the primary's linear part, but that is not independent.
- The Pitiscus strip result is post-hoc (section 0).
- Data preparation is shared with the primary pipeline (section 1).
- PSR thresholds are empirical per case (noise null plus controls), not a calibrated false-alarm rate.
- On Hops 1 and 2 phase correlation is uninformative, so it cannot tell "the primary is wrong" apart from "both estimators are blind here".

---

## 6. Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 ~/miniforge3/envs/trinetra/bin/python scripts/evaluate_stage_n.py /path/to/new_output.json
```

The runner refuses to overwrite an existing output file. With no argument it writes `assets/real_cache/stage_n_corroboration_results.json`, and it aborts if that file exists. Seed 42; runtime about 190 s on CPU.
