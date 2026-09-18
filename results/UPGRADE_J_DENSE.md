# Stage J: Dense Grid Correlation at Pitiscus against LOLA-Controlled Reference

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

## 1. Solar Geometry Sourcing (J2a)

### Chandrayaan-2 TMC-2 (`ch2_tmc_ncn_20230130T1900132182_d_img_d32`)
- **PDS4 Label Tags (`data/calibrated/20230130/ch2_tmc_ncn_20230130T1900132182_d_img_d32.xml`):**
  - `<isda:sun_azimuth>53.021211</isda:sun_azimuth>` (Azimuth 53.02 deg at product reference center -70.85 deg S, 32.26 deg E)
  - `<isda:sun_elevation>17.223731</isda:sun_elevation>` (Elevation 17.22 deg at product reference center)
  - `<isda:solar_incidence>72.776269</isda:solar_incidence>` (Incidence 72.78 deg)
- **Local Site Geometry at Pitiscus (Lat -51.25 deg, Lon 31.25 deg):**
  - Computed from product solar vector rotated to the topocentric horizon coordinates of the Pitiscus overlap:
    - **Local Solar Elevation:** 27.59 deg (Incidence: 62.41 deg)
    - **Local Solar Azimuth:** 60.87 deg (Illumination from ENE)

### LROC NAC Reference Frame (`M1149280834`)
- **Acquisition Epoch:** `2014-03-12 16:26:07 UTC`
- **Sub-solar Coordinates:** Latitude -0.96 deg, Longitude 44.84 deg
- **Local Site Geometry at Pitiscus (Lat -51.25 deg, Lon 31.25 deg):**
  - **Local Solar Elevation:** 38.42 deg (Incidence: 51.58 deg)
  - **Local Solar Azimuth:** 342.55 deg (Illumination from NNW)

### Geometric Discrepancy
- **Elevation Difference:** |38.42 deg - 27.59 deg| = 10.83 deg
- **Azimuth Difference:** |342.55 deg - 60.87 deg| = 281.68 deg (angular separation: 78.32 deg)  
  *Sun vectors are separated by 78.32 deg in azimuth, causing slope shading and crater shadows to cast in near-perpendicular directions.*

---

## 2. Common-Grid Reprojection (J2)

### Orthophoto Metadata (`NAC_DTM_PITISCUS_M1149280834_2M.TIF`)
- **Projection:** Equirectangular Moon (`Moon_localRadius`, R = 1,737,400 m)
- **Standard Parallel:** -51.0 deg
- **Central Meridian:** 180.0 deg
- **Native Bounds:**
  - X in [-2841680.00, -2835360.00] m (Width: 6,320.00 m = 3,160 px at 2.00 m/px)
  - Y in [-1566436.00, -1537336.00] m (Height: 29,100.00 m = 14,550 px at 2.00 m/px)
- **Selenographic Bounds:** Latitude [-51.6577 deg, -50.6981 deg], Longitude [31.0892 deg, 31.4204 deg]

### Common Map Grid Definition
- **Grid CRS:** Equirectangular Moon (R = 1,737,400 m, standard parallel = -51 deg, central meridian = 180 deg)
- **Common GSD:** 4.72 m/px (matching coarser sensor TMC-2)
- **Overlap Footprint:**
  - X in [-2841680.00, -2835360.00] m (6.32 km)
  - Y in [-1566436.00, -1538943.33] m (27.49 km)
  - **Dimensions:** 1,339 cols x 5,825 rows
  - **Overlap Area:** 173.75 km^2

### A-Priori Misalignment (J2c)
- **Unadjusted Co-registration Residuals:**
  - Phase correlation offset across central overlap: dx = -387.60 m (-82.12 px), dy = +1057.49 m (+224.04 px)
  - Total Euclidean A-Priori Misalignment: 1,126.29 m (238.62 common-grid px)
  - *This displacement exceeds the standard 128x128 search radius (32 px = 151.0 m).*

---

## 3. Dense Grid Correlation Benchmark (J3 and J5)

A regular grid of 32 x 32 = 1,024 correlation nodes was evaluated across the 173.75 km^2 overlap.

### Rejection Thresholds
- **Peak Floor:** Peak normalized cross-correlation NCC < 0.40
- **Uniqueness Ratio:** Secondary peak >= 0.90 * primary peak (with 5 x 5 neighborhood suppression)
- **Radius Bound:** Offset exceeds maximum search radius
- **Nodata:** Node window intersects no-data or zero-variance region

### Results: Standard Window vs Widened Window vs Controls

| Configuration | Window (Search / Template) | Search Radius | Accepted Nodes | Total Nodes | Acceptance Ratio | Mean Peak Correlation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Genuine (Standard) | 128x128 / 64x64 | 32 px (151 m) | 2 | 1024 | 0.20% | 0.2354 |
| Genuine (Widened) | 256x256 / 64x64 | 96 px (453 m) | 2 | 1024 | 0.20% | 0.3631 |
| Control: Rot 90 | 256x256 / 64x64 | 96 px (453 m) | 0 | 1024 | 0.00% | 0.0592 |
| Control: Rot 180 | 256x256 / 64x64 | 96 px (453 m) | 2 | 1024 | 0.20% | 0.3559 |
| Control: Rot 270 | 256x256 / 64x64 | 96 px (453 m) | 0 | 1024 | 0.00% | 0.0628 |
| Control: V-Flip | 256x256 / 64x64 | 96 px (453 m) | 3 | 1024 | 0.29% | 0.3507 |
| Control: H-Flip | 256x256 / 64x64 | 96 px (453 m) | 0 | 1024 | 0.00% | 0.3486 |
| Control: Offset (500 px) | 256x256 / 64x64 | 96 px (453 m) | 2 | 1024 | 0.20% | 0.3567 |
| Control: Uniform Noise | 256x256 / 64x64 | 96 px (453 m) | 0 | 1024 | 0.00% | 0.0483 |

### Detailed Rejection Counts by Cause (Genuine Runs)

| Configuration | Total | Accepted | Rej: Peak < 0.40 | Rej: Uniqueness (Ambiguity) | Rej: Radius Exceeded | Rej: Nodata |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Standard (128x128) | 1024 | 2 (0.20%) | 583 | 166 | 3 | 270 |
| Widened (256x256) | 1024 | 2 (0.20%) | 296 | 533 | 1 | 192 |

---

## 4. Control Validation Metric (J5)

Following the strict definition established in Stage H:
Delta_shuffle = ratio_genuine - max(all control ratios)

- **Genuine Ratio:** 0.20% (2 / 1,024)
- **Max Control Ratio:** 0.29% (3 / 1,024, from vertical flip)
- **Delta_shuffle:** 0.20% - 0.29% = -0.09%

### Peak Correlation Separation
- On uniform noise and perpendicular rotations (rot90, rot270), normalized cross-correlation produces flat response surfaces (mean peak <= 0.0628), demonstrating that NCC does not suffer from coordinate-grid consensus hallucinations.
- However, across repetitive lunar cratered highland terrain, arbitrary vertical and horizontal flips achieve mean peak correlations (0.3428 to 0.3559) and accepted node counts (2 to 3) identical to or slightly exceeding the genuine image (0.3631, 2 nodes), yielding Delta_shuffle <= 0.

---

## 5. Failure Mode Analysis (J7)

Per J7 instructions, the specific experimental outcomes and physical causes are documented below:

1. **A-priori Misalignment Exceeds Standard Search Window:**  
   The a-priori pointing and ephemeris offset between Chandrayaan-2 TMC-2 and the LROC NAC DTM orthophoto is approximately 1,126 m (238 common-grid px). The standard 128 x 128 window provides a search radius of only 32 px = 151 m, placing the true correspondence outside the search radius for unadjusted coordinates.

2. **Illumination Azimuth Difference (78.32 deg):**  
   The primary physical factor preventing cross-correlation matching is the large solar azimuth difference (60.87 deg ENE vs 342.55 deg NNW). On topography with relief, slope shading and crater shadows are cast in near-perpendicular directions. Normalized cross-correlation assumes a monotonic radiometric relationship and cannot bridge such severe illumination disparity without 3D shape-from-shading or synthetic hillshade rendering.

3. **Ambiguity in Widened Windows:**  
   When widening the search window to 256 x 256 (96 px = 453 m radius), the correlator suffers from repetitive texture ambiguity: 533 of 1,024 nodes fail the uniqueness check because secondary crater rims in the wider search window produce correlation peaks within 90% of the primary peak.

4. **Negative Control Margin (Delta_shuffle < 0):**  
   Because the accepted node fraction on genuine data is only 0.20% and controls produce 0.00% to 0.29%, Delta_shuffle = -0.09%. The dense correlator does not clear the safety gate.
