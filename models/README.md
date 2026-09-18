# TriNetra Fine-Tuned Model Weights

This directory contains the trained weights and deployment artifacts for TriNetra's domain-adapted feature matcher.

## Canonical Checkpoint: `eloftr_lunar_finetuned_epoch7.pt`

- **Architecture:** EfficientLoFTR (Linear Transformer with Rotary Position Embeddings / RoPE)
- **Parameter Count:** 16,050,878 parameters
- **Checkpoint Size:** ~184 MB
- **Source Initialization:** `MatchAnything-ELoFTR` (outdoor pretrained)
- **Training Epoch:** Epoch 7 (`ckpt_best.pt`, selected by optimal validation loss)
- **Input Resolution:** $256 \times 256$ with custom RoPE NPE configuration: `[256, 256, 256, 256]`

### Training Dataset & Regime
- **Dataset:** 15,000 synthetic illumination pairs generated from the LOLA 5m DEM at the Lunar South Pole (Shackleton Rim, -89.72°S, 223.13°E).
- **Physical Rendering:** Ray-traced shadow casting with horizon-angle sky-view ambient factor, Lommel-Seeliger scattering, and stratified solar elevation ($2^\circ \le \theta_{\text{elev}} \le 30^\circ$).
- **Filtering Gates:** 4-point in-flight rejection gating (valid mask $\ge 40\%$, std dev $\ge 12$ DN, shadow $\le 60\%$, near-floor $\le 70\%$).
- **Optimizer:** AdamW ($\text{lr} = 1\times 10^{-4}$), Cosine Annealing scheduler ($\eta_{\min} = 1\times 10^{-6}$), gradient accumulation across 8 steps.

### Authentic Flight Verification Metrics (Chandrayaan-2 OHRC ↔ TMC-2)
Tested on confirmed overlapping flight crop (`assets/real_cache/polar_flight_hop1.npz`):
- **Raw Feature Matches:** 217
- **Consensus Inliers:** **49** (with deterministic `cv2.setRNGSeed(42)`)
- **Inlier Consensus Ratio:** **22.58% (22.6%)**
- **Reprojection RMSE:** 9.23 c-px (2.22 native TMC-2 px, 9.42 m ground error)
- **Flight Gate Status:** **GATED (6-Criterion)** under $\Delta_{\text{shuffle}} = -5.15\% < +15.0\%$ (Initial 5-criterion PASS superseded due to coordinate-grid bias under negative controls)
- *(Note: Initial 5-criterion evaluation reported PASS with 49 inliers / 22.58% ratio; both Hop 1 and Hop 2 are gated under mandatory Criterion 6 shuffle invariance).*

### Overfitting Dynamics & Checkpoint Selection
Validation was monitored against unseen DEM illumination pairs (`shard_14.npz`) and authentic flight crops across 17 epochs:
- **Epoch 7 (Optimal):** Validation loss = 0.1213 | Flight inliers = 53 (24.4%)
- **Epoch 9:** Validation loss = 0.1609 | Flight inliers = 40 (23.7%)
- **Epoch 15:** Validation loss = 0.2156 | Flight inliers = 23 (18.9%)
- **Epoch 17:** Training halted early due to synthetic-to-real domain overfitting.

## How to Verify Locally

Run the standalone verification script:
```bash
python scripts/verify_finetuned_checkpoint.py
```

Expected output:
```text
Checkpoint loaded: Epoch 7
Model parameters: 16,025,216
Raw matches: 217
Inliers after RANSAC: 49
Inlier ratio: 22.58%
Gate: CLEARED
```
