# UPGRADE U4 — LRO NAC EXTERNAL REFERENCE HOP (AUDIT & VALIDATION)

**Experiment ID:** UPGRADE_U4_NAC  
**Target Pair:** Chandrayaan-2 OHRC (`ch2_ohr_ncp_20241115T1525004388`) vs LROC NAC Orthophoto (`NAC_DTM_SHACKRDGE02_M139797542_120CM.TIF`)  
**Site:** Shackleton - de Gerlache Connecting Ridge (Center: 228.3607°E, -89.5270°S)  
**Reference Frame:** LRO NAC frame `M139797542` map-projected at 1.20 m/px  
**Source Frame:** Chandrayaan-2 OHRC calibrated array at 0.24 m/px  
**Evaluation Mode:** Zero-shot inference using fine-tuned lunar EfficientLoFTR checkpoint (`eloftr_lunar_finetuned_epoch7.pt`)  
**Final Validation Verdict:** **MEASURED NEGATIVE RESULT (VOIDED BY CONTROL EXPERIMENTS)**  

---

## 1. Reference Provenance

All external metrics are grounded against documented geodetic control from the Lunar Reconnaissance Orbiter Camera (LROC) Science Operations Center:

- **DTM Product:** `NAC_DTM_SHACKRDGE02` (Version 2, completed April 23, 2024)
- **Geodetic Control:** Tied to **27 distinct LOLA orbit tracks**
- **Mean Absolute Error against LOLA (`lola_avg`):** **0.84 m**
- **RMS Absolute Error against LOLA (`lola_rms`):** **1.06 m**
- **Global Registration Adjusted RMS (`adjust_rms`):** **0.40 m**
- **Multi-Sensor Triangulation RMS (`triang_rms`):** **0.173** (SOCET SET bundle adjustment)
- **Relative Vertical Precision at 90% Confidence (`relat_le`):** **3.41 m**
- **Convergence Angle:** **19.99°**
- **Orthophoto Source:** NAC frame `M139797542` (nadir viewing, emission angle 1.69°)
- **Orthophoto Map Scale:** **1.20 m/px** in Polar Stereographic Moon CRS ($R = 1,737,400\text{ m}$)

> [!IMPORTANT]
> **External Accuracy Floor:** The reference product carries its own documented root-mean-square error against LOLA profiles of **1.06 m**. Therefore, no registration against this orthophoto can claim external absolute accuracy better than roughly $1.06\text{ m}$, regardless of matcher performance. The reference is geodetically controlled, but is not error-free ground truth.

---

## 2. Solar Geometry Reconciliation (B0 / C5)

Authoritative solar and spacecraft attitude parameters extracted directly from authentic PDS4 product XML labels:

| Product ID | Mission / Instrument | Target Site | Sun Azimuth | Sun Elevation | Solar Incidence | Spacecraft Roll | Spacecraft Pitch | Spacecraft Yaw | PDS4 XML Tag Source |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ch2_ohr_ncp_20241115T1525004388` | CH2 OHRC | Shackleton Rim (Hop 1) | **242.98°** | **0.79°** | **89.21°** | **+15.19°** | -11.08° | +0.03° | `<sun_azimuth>`, `<sun_elevation>`, `<solar_incidence>`, `<roll>`, `<pitch>`, `<yaw>` |
| `ch2_tmc_ncn_20231205T1906512971` | CH2 TMC-2 | Shackleton Rim (Hop 1) | **283.27°** | **7.06°** | **82.94°** | **+0.02°** | -0.11° | +180.00° | `<sun_azimuth>`, `<sun_elevation>`, `<solar_incidence>`, `<roll>`, `<pitch>`, `<yaw>` |
| `ch2_ohr_ncp_20211023T0027462822` | CH2 OHRC | Shiv Shakti (Legacy) | **298.43°** | **9.13°** | **80.87°** | **+15.76°** | +4.52° | -0.00° | `<sun_azimuth>`, `<sun_elevation>`, `<solar_incidence>`, `<roll>`, `<pitch>`, `<yaw>` |
| `ch2_tmc_ncn_20230130T1900132182` | CH2 TMC-2 | South Pole (Hop 2) | **53.02°** | **17.22°** | **72.78°** | **-0.02°** | -0.02° | +0.01° | `<sun_azimuth>`, `<sun_elevation>`, `<solar_incidence>`, `<roll>`, `<pitch>`, `<yaw>` |
| `ch2_iir_nri_20231003T2152304115` | CH2 IIRS | South Pole (Hop 2) | **277.23°** | **2.29°** | **87.71°** | **-10.41°** | -0.54° | -0.10° | `<sun_azimuth>`, `<sun_elevation>`, `<solar_incidence>`, `<roll>`, `<pitch>`, `<yaw>` |
| `M139797542` (NAC Frame) | LRO NAC | Shackleton Ridge (U4) | **96.62°** | **1.45°** | **88.55°** | **nadir** | nadir | nadir | LROC EDR Label (`Sub solar azimuth`, `Incidence angle`, `Emission angle 1.69°`) |

### Solar Audit Findings
1. **Origin of the Slide Claim (114.6° Azimuth / 15.8° Roll):**
   - The slide figures describe the cross-site comparison between **Shiv Shakti OHRC (`20211023`)** (azimuth 298.43°, roll 15.76°) and **South Pole TMC-2 (`20230130`)** (azimuth 53.02°, roll -0.02°):
     $$\Delta\text{Azimuth} = |298.43^\circ - (53.02^\circ + 360^\circ)| = 114.60^\circ$$
     $$\Delta\text{Roll} = 15.76^\circ - (-0.02^\circ) = 15.78^\circ \approx 15.8^\circ$$
2. **Hop 1 Status:**
   - This pair is **NOT** Hop 1. Hop 1 was officially adopted at Shackleton Rim (`20241115` vs `20231205`). The slide claim of 114.6° is stale and describes two products from different sites 2,000 km apart. It must be replaced across all decks and repo files.
3. **Correct Authoritative Angle Differences:**
   - **Hop 1 Actual (Shackleton Rim):**
     - $\Delta\text{Azimuth} = |283.27^\circ - 242.98^\circ| = \mathbf{40.28^\circ}$
     - $\Delta\text{Elevation} = |7.06^\circ - 0.79^\circ| = \mathbf{6.28^\circ}$
     - $\Delta\text{Roll} = |15.19^\circ - 0.02^\circ| = \mathbf{15.17^\circ}$
   - **Hop 2 Actual (South Pole):**
     - $\Delta\text{Azimuth} = 360^\circ - |277.23^\circ - 53.02^\circ| = \mathbf{135.79^\circ}$
     - $\Delta\text{Elevation} = |17.22^\circ - 2.29^\circ| = \mathbf{14.93^\circ}$
     - $\Delta\text{Roll} = |-0.02^\circ - (-10.41^\circ)| = \mathbf{10.38^\circ}$
   - **External Hop U4 (OHRC 20241115 vs LROC NAC M139797542):**
     - $\Delta\text{Azimuth} = |242.98^\circ - 96.62^\circ| = \mathbf{146.36^\circ}$
     - $\Delta\text{Elevation} = |1.45^\circ - 0.79^\circ| = \mathbf{0.66^\circ}$
     - $\Delta\text{Roll / Emission} \approx \mathbf{17.11^\circ}$ (OHRC boresight slew 18.80° vs NAC nadir 1.69°)

---

## 3. Ingest & Extreme Low-Sun Crop Selection (B1, B2)

### B1. Orthophoto Ingest Verification
- **File:** `data/lroc_nac/NAC_DTM_SHACKRDGE02_M139797542_120CM.TIF` ($20.44\text{ MB}$, $21,434,528\text{ bytes}$)
- **Dimensions:** $5,116\text{ columns} \times 4,183\text{ rows}$, single band, `uint8`, nodata: `0.0`
- **Native Bounds:** Left: -13718.25 m, Bottom: -13228.85 m, Right: -7579.05 m, Top: -8209.25 m
- **Coordinate Reference System:** `PROJCS["PolarStereographic Moon", ... SPHEROID["Moon", 1737400, 0] ...]`
- **Confirmed Lunar Radius:** Exactly **1,737,400.0 m**
- **Pixel Scale:** Exactly **1.20 m/px** in Easting and Northing

### B2. Physical Illumination Analysis
- **Full Overlap Footprint:** Reprojection of the OHRC strip into the Polar Stereographic frame confirms a **14.57 km²** overlap covering OHRC lines **[21,100 : 51,700]** across all 12,000 samples.
- **Histogram-Derived Shadow Floor:**
  - In the OHRC calibrated data across the overlap line range, pixel intensities exhibit a massive noise-floor drop at $\text{DN} \le 5$ (85.88% cumulative) and flattens out by $\text{DN} \le 10$ (87.56% cumulative).
  - Physical shadow threshold chosen: **$\text{DN} \le 10$**.
  - **OHRC Overlap Shadow Fraction:** **87.56%** in full shadow; Saturated ($\text{DN} = 255$): **0.01%**.
  - **NAC Orthophoto Valid Shadow Fraction:** **2.20%** ($\le 10\text{ DN}$); Saturated ($\text{DN} = 255$): **0.90%**.
- **Crop Placement & Dynamic Range Optimization:**
  - 41 non-overlapping candidate tiles ($960\text{ m} \times 960\text{ m}$) were evaluated across the overlap region.
  - The highest scoring tile was placed at:
    - **Selenographic Coordinates:** **228.3607°E, -89.5270°S**
    - **OHRC Line/Sample Center:** Line (Scan) **39,800**, Sample (Pixel) **9,500**
    - **NAC Orthophoto Location:** Row **700**, Column **2,100**
    - **Rationale:** This tile captures the elevated ridge crest where OHRC shadow drops to **77.5%** (lowest shadow in the entire overlap), providing a standard deviation of 25.3 in OHRC and 36.5 in NAC with zero nodata edge artifacts.
- **Matched Ground Extent:**
  - Ground Footprint: **$960.0\text{ m} \times 960.0\text{ m}$**
  - OHRC Crop Dimensions: **$4000 \times 4000\text{ px}$** at 0.24 m/px
  - NAC Orthophoto Crop Dimensions: **$800 \times 800\text{ px}$** at 1.20 m/px
  - Native Scale Gap: **$5.0000\times$**
  - Standardized Canvas Resizing: Both resized to $1000 \times 1000\text{ px}$ ($0.96\text{ m/canvas px}$)
  - **Expected Canvas Scale Ratio:** Exactly **`1.0000`**

---

## 4. Stage C Validation & Controls (Why the Result is Voided)

### C1. Analysis of the Horizontal Match Lines
In `results/figures/u4_ohrc_vs_nac_crops.png`, all 39 inlier correspondence lines appear horizontal and near-parallel:
- **Fitted Translation:**
  - Canvas pixels: $t_x = +36.6733\text{ px}$, $t_y = +1.6284\text{ px}$
  - Ground metres: $t_x = +35.2064\text{ m}$, $t_y = +1.5633\text{ m}$
- **Vertical Displacement ($dy = y_{\text{ref}} - y_{\text{src}}$) Distribution:**
  - Min: **$-15.06\text{ px}$**, Max: **$+13.09\text{ px}$**
  - Mean: **$-0.3779\text{ px}$**, Std: **$6.4683\text{ px}$**
- **Why the Lines Appear Horizontal:**
  Across the side-by-side $2000 \times 1000$ canvas, points are separated by a horizontal canvas offset of $\Delta X \approx 1000 + 30 = 1030\text{ px}$. A vertical shift of $dy \in [-15, +13]\text{ px}$ produces a visual slope of $\frac{15}{1030} \approx 0.014$ ($<0.8^\circ$ deviation from horizontal).
- **Design Matrix SVD & Condition Number:**
  The affine similarity matrix condition number ($s / s = 1.00$) was trivially $1.00$ by definition of a similarity transform. The actual linear design matrix $A$ ($78 \times 4$) solving $[a, b, t_x, t_y]^T$ has:
  - Rank: **4 / 4** (full rank)
  - Singular values: $[5508.8, 5508.8, 1.795, 1.795]$
  - Condition number: **$3,069.3$** (dominated by pixel coordinate magnitude $\sim 1000$ vs constant column $1.0$).

### C2. Independent Control Experiments (The Smoking Gun)
Two rigorous scientific control tests were conducted using the identical pipeline and gate:

| Experiment | Configuration | Raw Matches | Inliers | Inlier Ratio (%) | Fitted Scale | Fitted Rotation | Gate Status | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nominal Match** | OHRC vs Genuine NAC True Site | 146 | 39 | 26.7% | 0.9926 | +0.18° | **CLEARED** | Apparent match |
| **Control A: Shuffle** | OHRC vs NAC Rotated 90° | **174** | **69** | **39.7%** | **0.9986** | **-2.41°** | **CLEARED** | **FAILED CONTROL (Structural Artefact)** |
| **Control B1: Offset** | OHRC vs NAC 2 km away (r=2500, c=1000) | **206** | **85** | **41.3%** | **1.0004** | **-0.09°** | **CLEARED** | **FAILED CONTROL (Structural Artefact)** |
| **Control B2: Offset** | OHRC vs NAC 3 km away (r=1500, c=3500) | **132** | **36** | **27.3%** | **0.9859** | **+3.85°** | **CLEARED** | **FAILED CONTROL (Structural Artefact)** |
| **Pure Noise** | Random Uniform Noise (256x256) | **495** | — | — | — | — | — | Bias to $(x, y) \approx (x', y')$ |

#### The Mechanism of False Correlation
LoFTR utilizes sinusoidal positional encodings (`npe = [256, 256, 256, 256]`). When fed images with large featureless regions (such as 77.5% deep lunar shadow), the self-attention features are dominated by positional encodings rather than photometric texture. This produces pseudo-identity correspondences ($p_1 \approx p_0$, $\Delta X \approx 0, \Delta Y \approx 0$). When evaluated under RANSAC with a generous $15.0\text{ px}$ threshold, these pseudo-identity points trivially fit an identity similarity matrix ($s \approx 1.000$, $\theta \approx 0^\circ$) regardless of whether the underlying terrain matches, is rotated by $90^\circ$, is taken from an entirely different crater, or is pure random noise.

### C3. Shadow and Distribution Honesty
- **Lit-Region Breakdown:**
  - **15 of 39 inliers (38.5%)** fall in the lit region ($\text{DN} > 10$).
  - **24 of 39 inliers (61.5%)** fall directly in regions of pitch-black sensor shadow ($\text{DN} \le 10$).
  - Points in deep shadow cannot represent genuine physical crater features; they are mathematical artifacts of the positional grid.
- **Grid Distribution & CV:**
  - Zero-shot 8x8 Grid CV: **2.104** (occupancy 18/64), clustered solely along the right-side illuminated ridge.
  - Balanced 8x8 Grid CV: **1.908** (occupancy 20/64), remaining heavily clustered ($\text{CV} \approx 2.0$, far worse than Hop 1's $0.715$ or Hop 2's $0.866$).

---

## 5. Metrics Scorecard (with C4 Exact Handling)

| Configuration | Raw Matches | Inliers | Inlier Ratio (%) | Reproj RMSE (canvas px) | Reproj RMSE (native NAC px) | Reproj RMSE (metres) | Fitted Scale (Expected: 1.0000) | Fitted Rotation (deg) | Condition Number | 8x8 Grid Occupancy | CV | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT Baseline** | 7 | 2 | 28.6% | **n/a (degenerate, 2-point exact fit)** | **n/a (degenerate, 2-point exact fit)** | **n/a (degenerate, 2-point exact fit)** | 1.5432 | -9.07° | 1.00 | 2/64 | 5.568 | **GATED** |
| **Fine-tuned Matcher (Zero-Shot)** | 146 | 39 | 26.7% | 8.1701 | 6.5361 | 7.8433 m | 0.9926 | +0.18° | 1.00 | 18/64 | 2.104 | **GATED (delta_shuffle: -5.17%, VOPO)** |
| **Fine-tuned + Refine + Balance** | 138 | 40 | 29.0% | 9.0744 | 7.2595 | 8.7114 m | 0.9718 | +0.21° | 1.00 | 16/64 | 2.236 | **GATED (delta_shuffle: -5.17%, VOPO)** |

---

## 6. Scientific Verdict & Conservative Slide Statement

### The Honest Scientific Verdict
1. **Measured Negative Result:** Direct cross-mission registration of Chandrayaan-2 OHRC against an LRO NAC orthophoto under $1.5^\circ$ grazing sun across a $146.4^\circ$ solar azimuth gap is physically constrained by deep topographic shadow (87.6% full shadow).
2. **False-Positive Immunity:** The submission pipeline's control protocol successfully detected and neutralized false-positive positional correlation. Under 90° rotation and non-overlapping spatial offset, the matcher produced identical pseudo-identity fits, proving the inliers are positional artifacts rather than genuine lunar terrain correspondences.

### Drafted Conservative Slide Sentence
> *"At the extreme grazing illumination of the lunar south pole (1.5° solar elevation), cross-mission registration of Chandrayaan-2 OHRC against an LRO NAC orthophoto across a 146.4° solar azimuth differential is physically precluded by severe topographic shadowing (87.6% shadow fraction). Rigorous negative-control testing (90° rotation and non-overlapping offset controls) revealed that apparent deep-learning consensus matches were positional-encoding artefacts rather than authentic surface correspondences, confirming that multi-mission lunar registration under opposing grazing sun requires robust physical shadow masking."*
