"""
Held-out validation of a fitted registration model.

Two schemes, both reported:
  * spatial_blocks: K contiguous along-track blocks. The model is fitted
    (RANSAC, then least squares) on K-1 blocks and evaluated on the held-out
    block, so it must extrapolate across the gap. Pessimistic.
  * random_kfold: nodes assigned to K folds at random. Interpolation. Optimistic.

For each held-out node the residual is computed against a model that never saw
it. Mismatched nodes have large residuals by construction, so the summary gives
the fraction within 1, 2 and 3 px and the RMSE / median over nodes within
`trim` px (default 3 px). The trimming threshold is fixed in advance and does
not depend on the held-out data.
"""

from typing import Dict

import numpy as np

from . import models


def _summary(r: np.ndarray, trim: float) -> Dict:
    keep = r < trim
    return {
        "n_heldout": int(len(r)),
        "frac_within_1px": float((r < 1).mean()) if len(r) else None,
        "frac_within_2px": float((r < 2).mean()) if len(r) else None,
        "frac_within_3px": float((r < 3).mean()) if len(r) else None,
        "trim_px": trim,
        "n_within_trim": int(keep.sum()),
        "rmse_within_trim_px": float(np.sqrt((r[keep] ** 2).mean())) if keep.any() else None,
        "median_within_trim_px": float(np.median(r[keep])) if keep.any() else None,
        "p68_within_trim_px": float(np.percentile(r[keep], 68)) if keep.any() else None,
        "p95_within_trim_px": float(np.percentile(r[keep], 95)) if keep.any() else None,
    }


def heldout(src, ref, kind: str, thresh: float, scheme: str, k: int = 5, trim: float = 3.0,
            seed: int = 42, knot_px: float = None, stripe_px: float = 500.0, height_at=None) -> Dict:
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = len(src)
    if scheme == "spatial_blocks":
        order = np.argsort(src[:, 1], kind="stable")
        fold = np.empty(n, dtype=int)
        fold[order] = np.arange(n) * k // n
    elif scheme == "stripes":
        # interleaved along-track stripes of stripe_px rows: interpolation across short gaps
        fold = (np.floor((src[:, 1] - src[:, 1].min()) / stripe_px).astype(int)) % k
    elif scheme == "random_kfold":
        fold = np.random.default_rng(seed).permutation(np.arange(n) % k)
    else:
        raise ValueError(scheme)
    centre = src.mean(axis=0)
    scale = float(np.sqrt(((src - centre) ** 2).sum(axis=1).mean()))
    r_out = np.full(n, np.nan)
    per_fold = []
    for f in range(k):
        tr, te = fold != f, fold == f
        if kind == "pspline_relief":
            res = models.pspline_relief_trimmed(src[tr], ref[tr], thresh, knot_px, height_at,
                                                domain=(src[:, 1].min(), src[:, 1].max()),
                                                ucentre=float(src[:, 0].mean()), uscale=float(src[:, 0].std()))
        elif kind == "pspline":
            res = models.pspline_trimmed(src[tr], ref[tr], thresh, knot_px,
                                         domain=(src[:, 1].min(), src[:, 1].max()),
                                         ucentre=float(src[:, 0].mean()), uscale=float(src[:, 0].std()))
        else:
            res = models.ransac(src[tr], ref[tr], kind, thresh, seed=seed + f, centre=centre, scale=scale)
        if res is None:
            per_fold.append({"fold": f, "status": "fit_failed"})
            continue
        r_out[te] = models.residuals(res["model"], src[te], ref[te])
        per_fold.append({"fold": f, **_summary(r_out[te], trim)})
    ok = ~np.isnan(r_out)
    return {"scheme": scheme, "k": k, "model": kind, "ransac_thresh_px": thresh,
            "overall": _summary(r_out[ok], trim), "per_fold": per_fold,
            "heldout_residuals": r_out.tolist(), "fold": fold.tolist()}
