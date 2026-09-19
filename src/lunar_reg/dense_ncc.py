"""
Dense node matching by zero-mean normalised cross-correlation.

For each node q on a regular lattice of the source grid, a (2h x 2h) template
centred on q is searched in the reference inside a window centred on
q + prior_offset with radius `radius`. The peak is refined by separable
parabolic interpolation. Returned `ref_xy` is the reference-grid position of
the template centre, so the displacement is ref_xy - src_xy.
"""

from typing import Dict, Optional

import cv2
import numpy as np


def match_nodes(src: np.ndarray, ref: np.ndarray, prior_offset, spacing: int = 48,
                half_tpl: int = 32, radius: int = 32,
                src_valid: Optional[np.ndarray] = None,
                ref_valid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    src = np.asarray(src, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if src_valid is None:
        src_valid = src > 0
    if ref_valid is None:
        ref_valid = ref > 0
    h, S = half_tpl, half_tpl + radius
    ox, oy = float(prior_offset[0]), float(prior_offset[1])
    rows = []
    for qy in range(h, src.shape[0] - h, spacing):
        for qx in range(h, src.shape[1] - h, spacing):
            ix, iy = int(round(qx + ox)), int(round(qy + oy))
            if iy - S < 0 or iy + S > ref.shape[0] or ix - S < 0 or ix + S > ref.shape[1]:
                continue
            if not ref_valid[iy - S:iy + S, ix - S:ix + S].all():
                continue
            if not src_valid[qy - h:qy + h, qx - h:qx + h].all():
                continue
            tpl = src[qy - h:qy + h, qx - h:qx + h]
            if tpl.std() < 1e-3:
                continue
            srch = ref[iy - S:iy + S, ix - S:ix + S]
            c = cv2.matchTemplate(srch, tpl, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (px, py) = cv2.minMaxLoc(c)
            cs = c.copy()
            cs[max(0, py - 4):py + 5, max(0, px - 4):px + 5] = -1.0
            second = float(cs.max())
            on_edge = not (0 < px < c.shape[1] - 1 and 0 < py < c.shape[0] - 1)
            dx = dy = 0.0
            if not on_edge:
                p = c[py - 1:py + 2, px - 1:px + 2]
                den_x = p[1, 0] - 2 * p[1, 1] + p[1, 2]
                den_y = p[0, 1] - 2 * p[1, 1] + p[2, 1]
                dx = 0.5 * (p[1, 0] - p[1, 2]) / den_x if den_x < 0 else 0.0
                dy = 0.5 * (p[0, 1] - p[2, 1]) / den_y if den_y < 0 else 0.0
            rows.append((qx, qy, ix - S + px + dx + h, iy - S + py + dy + h, peak, second, on_edge))
    a = np.array(rows, dtype=np.float64).reshape(-1, 7)
    return {"src_xy": a[:, 0:2], "ref_xy": a[:, 2:4], "peak": a[:, 4],
            "second": a[:, 5], "on_edge": a[:, 6].astype(bool)}


def accept(m: Dict[str, np.ndarray], peak_min: float = 0.5, uniq_max: float = 0.9) -> np.ndarray:
    """Pre-declared node acceptance: peak floor, uniqueness ratio, not on the search edge."""
    return (m["peak"] >= peak_min) & (m["second"] < uniq_max * m["peak"]) & (~m["on_edge"])
