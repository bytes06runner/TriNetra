# TriNetra (त्रिनेत्र) — Reviewer Executive Summary

This one-page summary summarizes the empirical registration results for Chandrayaan-2 planetary correspondence (Problem Statement SIH26166).

---

## 1. Measured Observation Datasets

> **Read this first:** Both crops were independently decimated to a common canvas size before matching, which absorbs the nominal 17.71x and 14.49x sensor GSD ratios. Recovered scale therefore measures residual footprint mismatch between crops, not the raw inter-sensor ratio. The matching problem solved here is illumination and modality invariance at matched ground sampling, not scale-invariant matching across raw resolutions.

All measurements were conducted on authentic Chandrayaan-2 PDS4 flight products downloaded from the ISSDC repository:
- **OHRC (0.24 m/px):** `ch2_ohr_ncp_20241115T1525004388` (Lines 1:4000, Samples 1:4000, 4000×4000 px, 960 m footprint; Center: -89.7207°S, 223.1257°E, Shackleton Rim; Sun Azimuth: 243.0°, Elevation: 0.8°, Roll: +15.19°).
- **TMC-2 (4.25 m/px):** `ch2_tmc_ncn_20231205T1906512971` (240×240 px, 1020 m footprint; Center: -89.7207°S, 223.1257°E, Shackleton Rim; Sun Azimuth: 283.3°, Elevation: 7.1°, Roll: +0.02°) for Hop 1.
- **TMC-2 (4.72 m/px):** `ch2_tmc_ncn_20230130T1900132182_d_img_d32` (South Pole nadir track; Center: -70.85000°S, 32.26000°E; Sun Azimuth: 53.02°, Elevation: 17.22°, Roll: -0.02°) for Hop 2.
- **IIRS (68.38 m/px):** `ch2_iir_nri_20231003T2152304115_d_img_d18` (Raw Level-1 hyperspectral cube, Lines 510:630, Samples 60:180; Center: -70.85000°S, 32.26000°E; Sun Azimuth: 277.20°, Elevation: 2.29°, Roll: -10.41°).
- **Cross-Illumination & Attitude Disparity:** Hop 1 (Shackleton Rim) spans a **40.28° solar azimuth difference**, **6.28° solar elevation difference**, and **15.17° spacecraft roll offset**. Hop 2 (South Pole) spans a **135.79° solar azimuth difference**, **14.93° solar elevation difference**, and **10.38° spacecraft roll offset**.

---

## 2. Transformation Conditioning & Plausibility (Master Slide 3 Summary Table)

*Note: Scale and rotation parameters are estimated in pre-scaled canvas space (1000×1000 for Hop 1, 800×800 for Hop 2), not raw sensor space. Strict Delta_shuffle is defined under H1 as:*
$$\Delta_{\text{shuffle}} = \text{ratio}_{\text{genuine}} - \max(\text{ALL control ratios run, including rot90, rot180, rot270, vflip, hflip, offset, and noise})$$
*Status is evaluated against the mandatory Six-Criterion Spaceflight Validity Gate (reject if strict $\Delta_{\text{shuffle}} < +15.0\%$). Five-criterion results are officially superseded.*

| Method | Genuine Ratio | Worst Control Ratio | Which Control | Delta_shuffle (strict, H1) | Inliers | Gate Status |
|:---|:---:|:---:|:---|:---:|:---:|:---:|
| **Hop 1: Baseline SIFT** | 1.61% (6/372) | 1.73% (6/347) | Rot 180° (noise: 1.40%) | -0.12% | 6 | GATED (blind, <20 inl) |
| **Hop 1: Fine-Tuned LoFTR** | 22.58% (49/217) | 27.73% (66/238) | Rot 180° (noise: 25.74%, offset: 24.31%) | -5.15% | 49 | GATED (artefact, <+15%) |
| **Hop 2: Baseline SIFT** | 2.21% (6/272) | 2.09% (6/287) | Rot 180° (noise: 1.95%) | +0.12% | 6 | GATED (blind, <20 inl) |
| **Hop 2: Phase Congruency + LoFTR** | 40.39% (124/307) | 52.07% (226/434) | Uniform noise (rot270: 47.66%, rot180: 43.35%) | -11.68% (-7.27% geom) | 124 | GATED (artefact, <+15%) |
| **External U4: LRO NAC Orthophoto** | 26.71% (39/146) | 41.26% (85/206) | Offset 2 km (rot180: 41.28%, rot90: 39.66%) | -14.55% (-12.95% rot) | 39 | GATED (artefact, <+15%) |
| **ORB (Hop 1 / Hop 2)** | 1.60% / 1.93% | 1.42% / 1.89% | Rot 180° (noise: 0.61% / 0.77%) | +0.18% / +0.05% | 17 / 19 | GATED (blind, <20 inl) |
| **AKAZE (Hop 1 / Hop 2)** | 0.00% (0/0) | 0.00% (0/0) | None (0 matches detected) | 0.00% | 0 | GATED (blind, 0 matches) |
| **Template NCC (7×7 grid)** | 6.12% (3/49) | 6.12% (3/49) | Rot 180° (noise: 0.00%) | 0.00% | 3 | GATED (blind, <20 inl) |
| **Pure Noise vs Pure Noise (Sanity Run)** | 80.87% (634/784) | n/a | Pure uniform random noise input | n/a (Sanity Baseline) | 634 | SANITY FAIL (Grid consensus) |

*Frequency-Domain Benchmark (Phase Correlation): Phase correlation (cv2.phaseCorrelate) operates on Fourier power spectra and outputs normalized peak intensity rather than discrete point correspondence inliers. Peak correlation responses are 0.0203 (Hop 1 genuine) vs 0.0167 (rot180) / 0.0185 (noise), and 0.0132 (Hop 2 genuine) vs 0.0151 (rot180) / 0.0248 (noise). Because the extreme cross-sensor scale gaps disperse Fourier energy, peak values remain indistinguishable from the noise floor (~0.01–0.02).*

### Threshold Justification & Calibration Status
The single operational threshold is **$\Delta_{\text{shuffle}} \ge +15.0\%$**.
- **Structural Basis:** The value is chosen for exact structural symmetry with Criterion 1 of the Spaceflight Validity Gate (`inlier_ratio_pct >= 15.0%`). It represents the minimum excess inlier consensus demanded of genuine image features above background noise or coordinate hallucination.
- **Calibration Status (Provisional):** This threshold is **NOT tuned or calibrated to our measured results**. Our genuine runs span from $-14.55\%$ to $+0.18\%$, meaning a positive threshold anywhere in that range rejects all tested configurations. Because no multi-sensor lunar benchmark dataset with verified sub-pixel ground truth currently exists, this threshold is provisional and intended as a conservative engineering safeguard against false-positive registrations.

### Failure Mode Dichotomy: Blind vs Hallucinating (H2)
The negative-control benchmark reveals two fundamentally distinct failure modes across matcher classes:
1. **Classical and Frequency Matchers (SIFT, ORB, AKAZE, Template NCC): Blind, Not Hallucinating.**
   These methods sit tightly within 0.2 percentage points of zero Delta_shuffle in both directions ($\Delta \in [-0.12\%, +0.18\%]$), with 0 to 19 inliers. Their failure is physical insensitivity: across the extreme 17x to 14x lunar scale and illumination gap, their gradient and intensity descriptors find virtually no reproducible keypoints. Crucially, they do NOT hallucinate matches on rotated or noise images; they fail to find matches everywhere equally.
2. **Deep Feature Matchers (LoFTR): Hallucinating, Not Blind.**
   In contrast, LoFTR produces hundreds of matches and high inlier counts on rotated references, non-overlapping terrain 5,000 km away, and uniform noise (peaking at 634 inliers / 80.87% consensus on pure noise vs noise). LoFTR is not blind; it actively hallucinates geometric consensus. In low-contrast scenes, the transformer self-attention and cross-attention heads default to matching coordinate positions ($p_1 \approx p_0$), which RANSAC trivially fits as a pseudo-identity similarity matrix.

### Positional Encoding is Load-Bearing (H3)
Ablation test E3 shows that disabling Rotary Position Embeddings (RoPE=False) collapses match extraction from ~220-244 raw matches down to 18-24 matches, leaving only 4 inliers (well below the 20-inlier floor). This proves that positional encoding cannot simply be removed or turned off: it is both the mathematical source of the coordinate-grid artefact and simultaneously load-bearing for the transformer architecture to produce correspondences at all. Resolving this requires learning invariant physical representations under strict shadow masking, not naive ablation of positional encodings.

---

## 3. Reprojection RMSE & Sub-Pixel Status

Registration accuracy, best configuration per hop:
  Hop 1 (OHRC to TMC-2, Shackleton Rim, 17.71x):
    RMSE 9.23 canvas px = 2.22 native TMC-2 px = 9.42 m
  Hop 2 (TMC-2 to IIRS, South Pole, 14.49x):
    RMSE 9.05 canvas px = 1.36 native IIRS px = 92.78 m

Sub-pixel accuracy in native reference pixels is not achieved on either hop (2.22 and 1.36 native px). The problem statement target is not met by the current global 4-DoF model.

---

## 4. Match Point Distribution (Stated as a Limitation)

Match points are **not uniformly distributed** across the full image area:
- **Hop 1 (OHRC ↔ TMC-2):** Inliers occupy `22 / 64` cells (34.4% coverage) on an 8×8 grid, with a count coefficient of variation (CV) of `0.715`.
- **Hop 2 (TMC-2 ↔ IIRS):** Inliers occupy `39 / 64` cells (60.9% coverage), with a CV of `0.866`.
- Inliers cluster predominantly along high-contrast crater rims and sunlit ridges; shadowed crater basins lack sufficient texture to support keypoint correspondences.

---

## 5. Spaceflight Validity Gate & Ablation Analysis

The spaceflight validity gate enforces six sequential criteria:
1. **Inlier consensus floor:** inliers ≥ 20 AND ratio ≥ 15.0%.
2. **Numerical stability:** condition number cond(H) ≤ 100,000.
3. **Physical plausibility (rotation):** |rotation| ≤ 30.0°.
4. **Physical plausibility (scale):** scale error ≤ 25.0% relative to per-hop expected canvas scale (Hop 1: 0.9412, Hop 2: 0.9997).
5. **Transformation non-degeneracy:** inliers ≥ 2 × DoF (≥ 8 for 4-DoF similarity).
6. **Negative control / shuffle invariance (Criterion 6, Strict H1):** Delta_shuffle = ratio_genuine - max(ALL control ratios run, including rot90, rot180, rot270, vflip, hflip, offset, and noise) >= +15.0%.

> **Stage G / H Audit & Superseded 5-Criterion Status:**
> Under the original five-criterion gate, Hop 1 Fine-Tuned LoFTR (22.58% inlier ratio) and Hop 2 Phase Congruency + LoFTR (40.39% inlier ratio) were reported as passing. However, systematic negative-control benchmarking (Stages C, D, E, and F) revealed that the LoFTR transformer architecture forms geometric consensus along coordinate grids in low-texture lunar scenes, producing comparable or higher inlier ratios on rotated references (Hop 1 rot180: 27.73%, Hop 2 rot270: 47.66%) and uniform noise (Hop 1: 25.74%, Hop 2: 52.07%).
>
> Under the strict H1 definition (accounting for all controls run), genuine matches are heavily outperformed: strict Delta_shuffle is -5.15% on Hop 1 (worst control: rot180 at 27.73%) and -11.68% on Hop 2 (worst control: uniform noise at 52.07%, or -7.27% vs rot270). Both headline configurations fail Criterion 6 and are **GATED**. The earlier five-criterion PASS classifications are officially superseded.

> The 15.0 px RANSAC threshold and 1e5 conditioning ceiling were selected post-hoc after inspecting the flight crop distributions. They are empirical operational criteria, not pre-registered hypotheses. On our evaluation pairs, the conditioning gate cleanly separates valid from degenerate fits (passing: [689, 3646], degenerate: [3.5e5, 8.6e6]), but threshold sensitivity has not been independently benchmarked across a wider cohort of lunar sites.

- **Expected Scale Setting:** The expected scale is set to 0.9412 on Hop 1 (960 m OHRC / 1020 m TMC-2 footprint) and 0.9997 on Hop 2. Any fitted scale factor diverging by >25% from expected indicates unphysical geometric distortion.
- **Ablation Finding (G2):** The gate catches degenerate geometry independently of match count: when the inlier consensus rule is disabled in ablation, both SIFT runs are still rejected by the conditioning rule alone (Hop 1 cond = 8.6e+06, Hop 2 cond = 3.5e+05, threshold 1.0e+05), while both LoFTR runs fail Criterion 6.
- **Hop 1 SIFT Rejection:** Failed on inlier consensus (6 < 20, 1.6% < 15%), ill-conditioning ($cond = 8.6\times 10^6$), and unphysical scale ($0.1491$, 84.2% error relative to expected 0.9412).
- **Hop 2 SIFT Rejection:** Failed on inlier consensus (6 < 20, 2.2% < 15%), ill-conditioning ($cond = 3.5\times 10^5$), unphysical rotation ($+64.19^\circ$), and scale collapse ($0.4210$, 57.9% error).

---

## 6. Known Limitations

1. **Gate Width in Ground Distance:** The 15.0 c-px RANSAC threshold corresponds to **15.30 m** on Hop 1 (3.60 native TMC-2 px) and **153.86 m** on Hop 2 (2.25 native IIRS px). Tightening the gate to 5.0 c-px (5.10 m / 1.20 native px on Hop 1, 51.29 m / 0.75 native px on Hop 2) reduces inliers to 10 on Hop 1 and 24 on Hop 2.
2. **Terrain Slope Correlation & Roadmap Justification:** Residuals are not explained by local terrain slope in our measurements ($r = +0.042, p = 0.773$ on Hop 1; $r = +0.125, p = 0.166$ on Hop 2). The error budget is unresolved and likely combines point-spread blur across the scale gap, unmodelled lens distortion, and keypoint localisation uncertainty. Because slope does not explain residual magnitude, the empirical justification for the Thin-Plate Spline (TPS) non-rigid refinement roadmap item is weakened.
3. **Residual Scale Gap:** On Hop 1, the fitted scale is 1.0086 (residual gap of 0.0674 / 7.16% relative to polar footprint ratio 0.9412). On Hop 2, the fitted scale is 1.0537 compared to the expected canvas footprint ratio of 0.9997 (scale gap of 0.0583 / 5.83%). The pushbroom along-track motion sampling anisotropy hypothesis was tested via orbital mechanics ($v_{\text{ground}} = 1563.42\text{ m/s}, t_{\text{int}} = 53.06\text{ ms} \implies \text{sampling} = 82.96\text{ m}$, predicted ratio $1.2132$), which does NOT match the measured $s_y/s_x = 1.0403$. The pushbroom explanation is unsupported by the arithmetic, and the 5.83% scale gap remains unexplained (magnitude 0.0583).
4. **Three-Instrument Overlap at Shiv Shakti Point:** A three-instrument overlap exists at Shiv Shakti Point (OHRC ch2_ohr_ncp_20211023T0027462822, TMC-2 ch2_tmc_ncn_20230130T1900132182, IIRS ch2_iir_nri_20231003T2152304115), but a single shared footprint is physically constrained by the instrument suite. OHRC and IIRS differ by 263x in ground sample distance (0.26 m/px against 68.38 m/px), and OHRC's 12,000 px cross-track swath caps any common footprint at 3,120 m, which spans approximately 46 IIRS pixels. That is below the spatial support required by the phase congruency filter bank. The inverse framing is infeasible: a 120 px IIRS crop covers 8,205 m, requiring an OHRC canvas of ~31,560 pixels - well beyond the sensor detector width. End-to-end OHRC to IIRS correspondence within one shared footprint was therefore not measured. Hop 1 and Hop 2 are independent pairwise registrations at different sites. Bridging the full 263x span requires either TMC-2 mosaicking across multiple orbits or super-resolution of the IIRS patch, both outside the scope of this submission.

---

## 7. Audit & Provenance History

1. **Hop 1 Site & Metadata Correction:** An earlier revision mislabelled the Hop 1 evaluation site as Shiv Shakti Point due to an automated cache metadata copy defect. This discrepancy was identified during verification and corrected at the source. All current Hop 1 metrics are traced directly to the authentic Chandrayaan-2 Shackleton Rim polar flight pair.
2. **Hop 1 DEM Sampling Correction:** The earlier Hop 1 terrain slope correlation had sampled the wrong DEM region (Shiv Shakti Point instead of Shackleton Rim). This was corrected to sample the authentic Site04 LOLA polar DEM at the Shackleton coordinates, yielding $r = +0.0424, p = 0.7726$.
3. **Multi-Hop Composition Removal:** Composed transform chaining was removed from reported deliverables because Hop 1 and Hop 2 operate on independent observation sites with different coordinate frames; no shared three-instrument footprint was evaluated.

