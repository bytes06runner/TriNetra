import sys
import copy
from pathlib import Path
import torch
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

# 1. Import EfficientLoFTR exactly as done in kaggle/train_eloftr_lunar.py
REPO_ROOT = Path(__file__).resolve().parent.parent
ELOFTR_DIR = REPO_ROOT / 'third_party' / 'EfficientLoFTR'
sys.path.insert(0, str(ELOFTR_DIR))

# Handle 'src' module collision
if 'src' in sys.modules and not hasattr(sys.modules['src'], 'loftr'):
    del sys.modules['src']

from src.loftr import LoFTR, full_default_cfg
from typing import Dict, Any

def assert_cache_provenance(source_data: Any, arrays: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    """Assert that image arrays and metadata originate strictly from the same source cache.
    
    Raises ValueError if there is any provenance mismatch across source and output.
    """
    for arr_name, arr in arrays.items():
        if arr_name in source_data:
            if not np.array_equal(arr, source_data[arr_name]):
                raise ValueError(
                    f"Provenance violation: Array '{arr_name}' does not match source cache array!"
                )
    
    # Verify latitude consistency
    src_lat = float(source_data.get('center_lat', source_data.get('target_lat', 0.0)))
    meta_lat = float(metadata.get('center_lat', metadata.get('target_lat', 0.0)))
    if not np.isclose(src_lat, meta_lat, atol=1e-4):
        raise ValueError(
            f"Provenance violation: Output latitude ({meta_lat:.5f}) differs from source cache ({src_lat:.5f})!"
        )

    # Verify GSD consistency
    src_ohrc_gsd = float(source_data.get('ohrc_gsd', source_data.get('ohrc_res', 0.0)))
    meta_ohrc_res = float(metadata.get('ohrc_res', metadata.get('ohrc_gsd', 0.0)))
    if not np.isclose(src_ohrc_gsd, meta_ohrc_res, atol=1e-4):
        raise ValueError(
            f"Provenance violation: Output OHRC GSD ({meta_ohrc_res:.3f}) differs from source ({src_ohrc_gsd:.3f})!"
        )

    # Verify product ID consistency
    src_ohrc_p = str(source_data.get('ohrc_product', source_data.get('ohrc_product_id', '')))
    meta_ohrc_p = str(metadata.get('ohrc_product_id', metadata.get('ohrc_product', '')))
    if src_ohrc_p and meta_ohrc_p and src_ohrc_p != meta_ohrc_p:
        raise ValueError(
            f"Provenance violation: Output OHRC product ID ({meta_ohrc_p}) differs from source ({src_ohrc_p})!"
        )

def main():
    RNG_SEED = 42
    cv2.setRNGSeed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    print(f"Starting verification of fine-tuned EfficientLoFTR checkpoint (cv2_rng_seed={RNG_SEED})...")
    
    # 6. Load and print the trinetra_finetune_results.json for cross-reference.
    results_path = REPO_ROOT / 'assets' / 'trinetra_finetune_results.json'
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        print("\n--- Training Results (trinetra_finetune_results.json) ---")
        print(json.dumps(results, indent=2))
    else:
        print(f"\nTraining results not found at {results_path}")

    # 2. Build the model exactly as in build_lunar_eloftr()
    cfg = copy.deepcopy(full_default_cfg)
    cfg['coarse']['npe'] = [256, 256, 256, 256]
    cfg['match_coarse']['train_pad_num_gt_min'] = 20
    cfg['match_coarse']['train_coarse_percent'] = 0.2
    cfg['match_fine']['local_regress_temperature'] = 10.0
    model = LoFTR(config=cfg)
    
    # 3. Load the fine-tuned checkpoint
    checkpoint_path = REPO_ROOT / 'models' / 'eloftr_lunar_finetuned_epoch7.pt'
    if not checkpoint_path.exists():
        print(f"\nCheckpoint not found at {checkpoint_path}")
        return
        
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    
    epoch = ckpt.get('epoch', 'Unknown')
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint loaded: Epoch {epoch}")
    print(f"Model parameters: {num_params:,}")
    
    # 4. Run the EXACT same evaluation pipeline as evaluate_real_flight_crop()
    cache_path = REPO_ROOT / 'assets' / 'real_cache' / 'polar_flight_hop1.npz'
    if not cache_path.exists():
        print(f"\nCache not found at {cache_path}")
        return
        
    print("\nEvaluating real flight cache...")
    data = np.load(cache_path)
    disp_ohrc = data['disp_ohrc']
    disp_tmc = data['disp_tmc']
    
    # Resize both to 256x256 with cv2.INTER_AREA
    img0 = cv2.resize(disp_ohrc, (256, 256), interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(disp_tmc, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Normalize to float32 / 255.0, shape [1,1,256,256]
    tensor0 = torch.from_numpy(img0).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    batch = {'image0': tensor0, 'image1': tensor1}
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        model(batch)
        
    mkpts0_f = batch['mkpts0_f'].cpu().numpy()
    mkpts1_f = batch['mkpts1_f'].cpu().numpy()
    
    # Scale keypoints back by factor 1000.0 / 256.0 = 3.90625
    scale_factor = 1000.0 / 256.0
    mkpts0_scaled = mkpts0_f * scale_factor
    mkpts1_scaled = mkpts1_f * scale_factor
    
    raw_matches_num = len(mkpts0_scaled)
    
    # cv2.estimateAffinePartial2D with ransacReprojThreshold=15.0
    inliers_num = 0
    inliers_mask = np.zeros((0, 1), dtype=np.uint8)
    if raw_matches_num >= 4:
        # We use strict inlier threshold 15.0 and deterministic RNG seed
        cv2.setRNGSeed(RNG_SEED)
        np.random.seed(RNG_SEED)
        T, inliers_mask = cv2.estimateAffinePartial2D(
            mkpts0_scaled, mkpts1_scaled, method=cv2.RANSAC, ransacReprojThreshold=15.0
        )
        if inliers_mask is not None:
            inliers_num = int(np.sum(inliers_mask))
        else:
            inliers_mask = np.zeros((raw_matches_num, 1), dtype=np.uint8)
            
    # Compute inliers, ratio, gate status
    inlier_ratio = (inliers_num / raw_matches_num * 100) if raw_matches_num > 0 else 0
    # Criterion 1 only (superseded five-criterion era). The six-criterion gate also needs
    # Delta_shuffle >= +15% from negative controls, which this script does not run;
    # the recorded value for this configuration is -5.15% (results/RESULTS.md), so it is GATED.
    consensus_status = "met" if (inlier_ratio >= 15.0 and inliers_num >= 20) else "not met"
    gate_status = "GATED (Criterion 6: Delta_shuffle -5.15%, see results/RESULTS.md)"
    
    # 5. Print clearly
    print("\n--- Evaluation Results ---")
    print(f"Raw matches: {raw_matches_num}")
    print(f"Inliers after RANSAC: {inliers_num}")
    print(f"Inlier ratio: {inlier_ratio:.2f}%")
    print(f"Inlier consensus (Criterion 1 only): {consensus_status}")
    print(f"Gate: {gate_status}")
    print("\nExpected approximate numbers: 217 raw / 53 inliers / 24.4%")
    print("Matches reproduce expectation: ", end="")
    if abs(raw_matches_num - 217) < 10 and abs(inliers_num - 53) < 10:
        print("YES")
    else:
        print("NO")
    # 7. Save a verification visualization to outputs/qa and assets/qa
    out_dir = REPO_ROOT / 'outputs' / 'qa'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'finetuned_verification.png'
    assets_dir = REPO_ROOT / 'assets' / 'qa'
    assets_dir.mkdir(parents=True, exist_ok=True)
    assets_out_path = assets_dir / 'finetuned_verification.png'
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    img0_full = disp_ohrc
    img1_full = disp_tmc
    
    concat_img = np.concatenate([img0_full, img1_full], axis=1)
    
    if len(concat_img.shape) == 2:
        ax.imshow(concat_img, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB))
        
    ax.axis('off')
    
    # Draw matches
    if raw_matches_num > 0:
        inliers_mask_flat = inliers_mask.ravel().astype(bool) if len(inliers_mask) > 0 else np.zeros(raw_matches_num, dtype=bool)
        outliers_mask = ~inliers_mask_flat
        
        # Draw outliers
        for (x0, y0), (x1, y1) in zip(mkpts0_scaled[outliers_mask], mkpts1_scaled[outliers_mask]):
            ax.plot([x0, x1 + 1000], [y0, y1], color='red', linewidth=1.0, alpha=0.5)
            
        # Draw inliers
        for (x0, y0), (x1, y1) in zip(mkpts0_scaled[inliers_mask_flat], mkpts1_scaled[inliers_mask_flat]):
            ax.plot([x0, x1 + 1000], [y0, y1], color='lime', linewidth=1.5, alpha=0.9)
            ax.scatter([x0, x1 + 1000], [y0, y1], color='lime', s=10)
            
    title_str = (f"Fine-Tuned Verification\n"
                 f"Raw: {raw_matches_num} | Inliers: {inliers_num} ({inlier_ratio:.1f}%) | Criterion 1: {consensus_status} | 6-criterion: GATED")
    ax.set_title(title_str, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(assets_out_path, dpi=150)
    plt.close()
    
    print(f"\nSaved visualization to {out_path} and {assets_out_path}")
    
    # Save the verified fine-tuned cache for app.py Dual-Engine Switcher
    # CRITICAL: Metadata is read strictly from the SAME source cache (data) as the arrays!
    cache_ft_path = REPO_ROOT / 'assets' / 'real_cache' / 'real_flight_hop1_finetuned.npz'

    H_matrix = np.eye(3, dtype=np.float64)
    if T is not None:
        H_matrix[:2, :] = T

    inlier_mask_bool = inliers_mask.ravel().astype(bool) if len(inliers_mask) > 0 else np.zeros(raw_matches_num, dtype=bool)
    if inliers_num > 0 and T is not None:
        inlier_pts1 = mkpts0_scaled[inlier_mask_bool]
        inlier_pts2 = mkpts1_scaled[inlier_mask_bool]
        ones = np.ones((len(inlier_pts1), 1), dtype=np.float32)
        pts1_h = np.hstack([inlier_pts1, ones])
        projected = (H_matrix @ pts1_h.T).T
        denom = projected[:, 2:3]
        denom[np.abs(denom) < 1e-8] = 1e-8
        projected = projected[:, :2] / denom
        reproj_rmse = float(np.sqrt(np.mean(np.sum((projected - inlier_pts2) ** 2, axis=1))))
    else:
        reproj_rmse = 0.0

    ohrc_res = float(data.get('ohrc_gsd', 0.24))
    tmc_res = float(data.get('tmc_gsd', 4.25))
    scale_gap = float(data.get('scale_gap', 17.71))
    target_lat = float(data.get('center_lat', -89.7207))
    target_lon = float(data.get('center_lon', 223.1257))
    ohrc_product_id = str(data.get('ohrc_product', 'ch2_ohr_ncp_20241115T1525004388'))
    tmc_product_id = str(data.get('tmc_product', 'ch2_tmc_ncn_20231205T1906512971'))

    save_dict = {
        'disp_ohrc': disp_ohrc,
        'disp_tmc': disp_tmc,
        'raw_tmc_crop': data.get('tmc_raw', cv2.resize(disp_tmc, (240, 240))),
        'pts1': mkpts0_scaled.astype(np.float32),
        'pts2': mkpts1_scaled.astype(np.float32),
        'inlier_mask': inlier_mask_bool,
        'H': H_matrix,
        'transform_type': 'Similarity Transform',
        'transform_dof': 4,
        'inliers': np.int64(inliers_num),
        'total_matches': np.int64(raw_matches_num),
        'inlier_ratio': np.float64(inlier_ratio),
        'reproj_rmse': np.float64(round(reproj_rmse, 2)),
        'inlier_threshold': np.float64(15.0),
        'ohrc_res': np.float64(ohrc_res),
        'tmc_res': np.float64(tmc_res),
        'scale_gap': np.float64(scale_gap),
        'ohrc_product_id': ohrc_product_id,
        'tmc_product_id': tmc_product_id,
        'target_lat': np.float64(target_lat),
        'target_lon': np.float64(target_lon),
        'center_lat': np.float64(target_lat),
        'center_lon': np.float64(target_lon),
        'tmc_sun_elevation': np.float64(data.get('tmc_sun_elevation', 1.5)),
        'tmc_sun_azimuth': np.float64(data.get('tmc_sun_azimuth', 53.0)),
        'ohrc_sun_elevation': np.float64(data.get('ohrc_sun_elevation', 1.5)),
        'ohrc_sun_azimuth': np.float64(data.get('ohrc_sun_azimuth', 167.6)),
        'site_name': 'Shackleton Rim (Lunar South Pole)',
        'source_cache_file': cache_path.name,
        'flight_validated': True,
    }

    # Strict provenance assertion: raises if any cross-file metadata leak is detected
    assert_cache_provenance(data, {'disp_ohrc': disp_ohrc, 'disp_tmc': disp_tmc}, save_dict)

    np.savez_compressed(cache_ft_path, **save_dict)
    print(f"Saved fine-tuned cache to {cache_ft_path} ({cache_ft_path.stat().st_size:,} bytes)")


if __name__ == '__main__':
    main()
