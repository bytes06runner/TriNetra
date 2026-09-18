"""
src/module2_matching/subpixel_refiner.py — Sub-pixel correspondence refinement and spatial distribution bucketing.

Implements:
1. U1: Local intensity alignment via phase correlation (cv2.phaseCorrelate) or
   normalized cross-correlation (NCC) with parabolic peak interpolation.
2. U2: 8x8 grid-bucketed match selection to improve spatial distribution before
   geometric fitting.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


def refine_correspondences(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    patch_size: int = 15,
    conf_floor: float = 0.3,
    method: str = "phase_correlate",
) -> Tuple[np.ndarray, int, int, float, np.ndarray]:
    """Refine reference-side match locations to sub-pixel precision.

    Args:
        img_src: Source image array (e.g. OHRC on Hop 1, TMC-2 on Hop 2).
        img_ref: Reference image array (e.g. TMC-2 on Hop 1, IIRS on Hop 2).
        pts_src: (N, 2) coordinates on img_src.
        pts_ref: (N, 2) coordinates on img_ref.
        patch_size: Size of local patch (default 15).
        conf_floor: Minimum correlation confidence to accept refinement (default 0.3).
        method: 'phase_correlate' (default) or 'ncc'.

    Returns:
        Tuple of:
            - refined_pts_ref: (N, 2) refined reference coordinates.
            - num_refined: Count of successfully refined matches.
            - num_rejected: Count of matches kept at original coordinate.
            - mean_shift: Average sub-pixel shift magnitude among refined matches.
            - shifts: (N,) shift magnitude for all matches (0 for rejected).
    """
    pts_src = np.asarray(pts_src, dtype=np.float32)
    pts_ref = np.asarray(pts_ref, dtype=np.float32)
    refined_pts_ref = pts_ref.copy()

    img_src_f = img_src.astype(np.float32)
    img_ref_f = img_ref.astype(np.float32)

    half = patch_size // 2
    max_shift = patch_size / 2.0
    num_refined = 0
    num_rejected = 0
    shifts_list = []
    per_point_shifts = np.zeros(len(pts_src), dtype=np.float32)

    hann = cv2.createHanningWindow((patch_size, patch_size), cv2.CV_32F)

    for i in range(len(pts_src)):
        x1, y1 = float(pts_src[i, 0]), float(pts_src[i, 1])
        x2, y2 = float(pts_ref[i, 0]), float(pts_ref[i, 1])

        # Boundary checks
        if (
            x1 - half < 0
            or x1 + half >= img_src.shape[1]
            or y1 - half < 0
            or y1 + half >= img_src.shape[0]
            or x2 - half < 0
            or x2 + half >= img_ref.shape[1]
            or y2 - half < 0
            or y2 + half >= img_ref.shape[0]
        ):
            num_rejected += 1
            continue

        tpl = cv2.getRectSubPix(img_src_f, (patch_size, patch_size), (x1, y1))
        target = cv2.getRectSubPix(img_ref_f, (patch_size, patch_size), (x2, y2))

        if np.std(tpl) < 1e-3 or np.std(target) < 1e-3:
            num_rejected += 1
            continue

        if method == "phase_correlate":
            (shift, resp) = cv2.phaseCorrelate(tpl, target, hann)
            # phaseCorrelate(src, dst) gives shift of dst relative to src
            # Target is around (x2, y2); to align with template, shift is -shift
            shift_x = float(shift[0])
            shift_y = float(shift[1])
            conf = float(resp)
        elif method == "ncc":
            search_r = 4
            search_size = patch_size + 2 * search_r
            if (
                x2 - search_size / 2.0 < 0
                or x2 + search_size / 2.0 >= img_ref.shape[1]
                or y2 - search_size / 2.0 < 0
                or y2 + search_size / 2.0 >= img_ref.shape[0]
            ):
                num_rejected += 1
                continue
            search = cv2.getRectSubPix(img_ref_f, (search_size, search_size), (x2, y2))
            res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            peak_x, peak_y = max_loc
            dx, dy = 0.0, 0.0
            if 0 < peak_x < res.shape[1] - 1:
                c_left = float(res[peak_y, peak_x - 1])
                c_mid = float(res[peak_y, peak_x])
                c_right = float(res[peak_y, peak_x + 1])
                denom = 2.0 * (c_left - 2.0 * c_mid + c_right)
                if abs(denom) > 1e-6:
                    dx = (c_left - c_right) / denom
            if 0 < peak_y < res.shape[0] - 1:
                c_top = float(res[peak_y - 1, peak_x])
                c_mid = float(res[peak_y, peak_x])
                c_bot = float(res[peak_y + 1, peak_x])
                denom = 2.0 * (c_top - 2.0 * c_mid + c_bot)
                if abs(denom) > 1e-6:
                    dy = (c_top - c_bot) / denom
            shift_x = float((peak_x - search_r) + dx)
            shift_y = float((peak_y - search_r) + dy)
            conf = float(max_val)
        else:
            raise ValueError(f"Unknown refinement method: {method}")

        shift_mag = float(np.hypot(shift_x, shift_y))
        if conf < conf_floor or shift_mag > max_shift:
            num_rejected += 1
            continue

        refined_pts_ref[i] = [x2 + shift_x, y2 + shift_y]
        shifts_list.append(shift_mag)
        per_point_shifts[i] = shift_mag
        num_refined += 1

    mean_shift = float(np.mean(shifts_list)) if shifts_list else 0.0
    return refined_pts_ref, num_refined, num_rejected, mean_shift, per_point_shifts


def bucket_matches_grid(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    ref_shape: Tuple[int, int],
    grid_size: int = 8,
    cap: Optional[int] = None,
    scores: Optional[np.ndarray] = None,
    img_ref: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """Apply 8x8 grid-bucketed match selection before geometric fitting.

    Caps the number of matches retained per cell, keeping highest-confidence matches.

    Args:
        pts_src: (N, 2) source keypoints.
        pts_ref: (N, 2) reference keypoints.
        ref_shape: (H, W) of the reference image.
        grid_size: Number of cells per axis (default 8 -> 64 cells).
        cap: Maximum matches retained per cell. If None, set to round(2 * mean_per_occupied).
        scores: Optional (N,) confidence score per match.
        img_ref: Optional reference image to compute local contrast if scores is None.

    Returns:
        Tuple of:
            - bucketed_pts_src: Retained source keypoints.
            - bucketed_pts_ref: Retained reference keypoints.
            - selected_indices: Indices of retained matches from original arrays.
            - effective_cap: Cap used.
            - stats: Dictionary of bucketing statistics.
    """
    pts_src = np.asarray(pts_src, dtype=np.float32)
    pts_ref = np.asarray(pts_ref, dtype=np.float32)
    N = len(pts_src)
    if N == 0:
        return pts_src, pts_ref, np.array([], dtype=int), 0, {}

    H_img, W_img = ref_shape
    cell_w = float(W_img) / float(grid_size)
    cell_h = float(H_img) / float(grid_size)

    cell_x = np.clip((pts_ref[:, 0] / cell_w).astype(int), 0, grid_size - 1)
    cell_y = np.clip((pts_ref[:, 1] / cell_h).astype(int), 0, grid_size - 1)
    cell_indices = cell_y * grid_size + cell_x

    unique_cells, counts = np.unique(cell_indices, return_counts=True)
    mean_per_occupied = float(np.mean(counts))

    if cap is None:
        effective_cap = max(1, int(round(2.0 * mean_per_occupied)))
    else:
        effective_cap = max(1, int(cap))

    # If scores not provided, compute local patch contrast (std dev) on reference image
    if scores is None:
        if img_ref is not None:
            img_ref_f = img_ref.astype(np.float32)
            scores_list = []
            for pt in pts_ref:
                x, y = float(pt[0]), float(pt[1])
                patch = cv2.getRectSubPix(img_ref_f, (15, 15), (x, y))
                scores_list.append(float(np.std(patch)))
            scores = np.array(scores_list, dtype=np.float32)
        else:
            scores = np.ones(N, dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32)

    selected_indices = []
    for c in unique_cells:
        matches_in_cell = np.where(cell_indices == c)[0]
        if len(matches_in_cell) <= effective_cap:
            selected_indices.extend(matches_in_cell)
        else:
            cell_scores = scores[matches_in_cell]
            sorted_order = np.argsort(-cell_scores)
            selected_indices.extend(matches_in_cell[sorted_order[:effective_cap]])

    selected_indices = np.array(sorted(selected_indices), dtype=int)

    stats = {
        "original_matches": N,
        "bucketed_matches": len(selected_indices),
        "occupied_cells_before": int(len(unique_cells)),
        "mean_per_occupied": mean_per_occupied,
        "effective_cap": effective_cap,
    }

    return pts_src[selected_indices], pts_ref[selected_indices], selected_indices, effective_cap, stats
