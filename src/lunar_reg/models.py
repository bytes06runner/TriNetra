"""
2D point-map models fitted by linear least squares, with a seeded RANSAC.

Models map source (x, y) to reference (x', y'). Coordinates are normalised
(centred, scaled to unit RMS) before fitting so polynomial terms stay well
conditioned; `predict` applies the same normalisation.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

MIN_SAMPLES = {"translation": 1, "similarity": 2, "affine": 3, "poly2": 6, "poly3": 10}


def _terms(u: np.ndarray, v: np.ndarray, kind: str) -> np.ndarray:
    one = np.ones_like(u)
    if kind == "affine":
        return np.column_stack([one, u, v])
    if kind == "poly2":
        return np.column_stack([one, u, v, u * u, u * v, v * v])
    if kind == "poly3":
        return np.column_stack([one, u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3])
    raise ValueError(kind)


@dataclass
class Model:
    kind: str
    centre: np.ndarray
    scale: float
    coef: np.ndarray  # layout depends on kind

    def predict(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        u = (xy[:, 0] - self.centre[0]) / self.scale
        v = (xy[:, 1] - self.centre[1]) / self.scale
        if self.kind == "translation":
            return xy + self.coef
        if self.kind == "similarity":
            a, b, tx, ty = self.coef
            return np.column_stack([xy[:, 0] + (a * u - b * v) * self.scale + tx,
                                    xy[:, 1] + (b * u + a * v) * self.scale + ty])
        T = _terms(u, v, self.kind)
        return xy + np.column_stack([T @ self.coef[:, 0], T @ self.coef[:, 1]])

    def params(self) -> Dict:
        return {"kind": self.kind, "centre": self.centre.tolist(), "scale": self.scale,
                "coef": np.asarray(self.coef).tolist()}


def fit(src: np.ndarray, ref: np.ndarray, kind: str, centre=None, scale=None) -> Model:
    """Least-squares fit of the displacement field d = ref - src."""
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if centre is None:
        centre = src.mean(axis=0)
    if scale is None:
        scale = float(np.sqrt(((src - centre) ** 2).sum(axis=1).mean())) or 1.0
    d = ref - src
    u = (src[:, 0] - centre[0]) / scale
    v = (src[:, 1] - centre[1]) / scale
    if kind == "translation":
        return Model(kind, np.asarray(centre), scale, d.mean(axis=0))
    if kind == "similarity":
        # d_x = (a u - b v) s + tx ; d_y = (b u + a v) s + ty  (a, b are deviations from identity)
        n = len(src)
        A = np.zeros((2 * n, 4))
        A[0::2] = np.column_stack([u * scale, -v * scale, np.ones(n), np.zeros(n)])
        A[1::2] = np.column_stack([v * scale, u * scale, np.zeros(n), np.ones(n)])
        sol = np.linalg.lstsq(A, d.reshape(-1), rcond=None)[0]
        return Model(kind, np.asarray(centre), scale, sol)
    T = _terms(u, v, kind)
    coef = np.linalg.lstsq(T, d, rcond=None)[0]
    return Model(kind, np.asarray(centre), scale, coef)


def residuals(model: Model, src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(model.predict(src) - np.asarray(ref, dtype=np.float64), axis=1)


def ransac(src: np.ndarray, ref: np.ndarray, kind: str, thresh: float, iters: int = 3000,
           seed: int = 42, centre=None, scale=None) -> Optional[Dict]:
    """Seeded RANSAC; the winning consensus set is refitted by least squares and
    the inlier set recomputed once against the refit."""
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n, k = len(src), MIN_SAMPLES[kind]
    if n < max(k, 2 * k):
        return None
    if centre is None:
        centre = src.mean(axis=0)
    if scale is None:
        scale = float(np.sqrt(((src - centre) ** 2).sum(axis=1).mean())) or 1.0
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, size=k, replace=False)
        try:
            m = fit(src[idx], ref[idx], kind, centre, scale)
        except np.linalg.LinAlgError:
            continue
        inl = residuals(m, src, ref) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < k:
        return None
    m = fit(src[best], ref[best], kind, centre, scale)
    r = residuals(m, src, ref)
    inl = r < thresh
    m = fit(src[inl], ref[inl], kind, centre, scale)
    r = residuals(m, src, ref)
    return {"model": m, "inliers": r < thresh, "residuals": r}


# ---------------------------------------------------------------------------
# Along-track penalised spline (P-spline) for pushbroom imagery.
#
# d(u, v) = sum_j B_j(v) * (c0_j + c1_j u + c2_j u^2), separately for dx and dy,
# where v is the along-track (row) coordinate and u the normalised cross-track
# coordinate. Cubic B-splines on uniform knots spaced `knot_px` apart over a
# fixed `domain` (row range) so that held-out folds share one basis. A second-
# difference penalty lambda * ||D2 c||^2 per cross-track term makes the fit
# interpolate linearly across gaps; lambda is chosen by GCV on the fitting data
# only. Robustness: iterative trimming (refit on residual < thresh, 6 rounds),
# not RANSAC, because the minimal sample is large.
# ---------------------------------------------------------------------------

@dataclass
class PSplineModel:
    kind: str
    knots: np.ndarray
    ucentre: float
    uscale: float
    coef: np.ndarray  # (n_basis * 3, 2)
    lam: float

    def _design(self, xy):
        from scipy.interpolate import BSpline
        v = np.clip(np.asarray(xy, dtype=np.float64)[:, 1], self.knots[3], self.knots[-4] - 1e-9)
        B = BSpline.design_matrix(v, self.knots, 3).toarray()
        u = (np.asarray(xy, dtype=np.float64)[:, 0] - self.ucentre) / self.uscale
        return np.hstack([B, B * u[:, None], B * (u * u)[:, None]])

    def predict(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return xy + self._design(xy) @ self.coef

    def params(self):
        return {"kind": self.kind, "knots": self.knots.tolist(), "ucentre": self.ucentre, "uscale": self.uscale,
                "lambda": self.lam, "coef": self.coef.tolist()}


def _pspline_fit(src, ref, knot_px, domain, ucentre, uscale, lams=10.0 ** np.arange(-3, 4)):
    from scipy.interpolate import BSpline  # noqa: F401
    v0, v1 = domain
    nseg = max(int(np.ceil((v1 - v0) / knot_px)), 1)
    inner = np.linspace(v0, v1, nseg + 1)
    knots = np.concatenate([[inner[0]] * 3, inner, [inner[-1]] * 3])
    proto = PSplineModel("pspline", knots, ucentre, uscale, None, 0.0)
    X = proto._design(src)
    d = np.asarray(ref, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    nb = len(knots) - 4
    D1 = np.diff(np.eye(nb), n=2, axis=0)
    D = np.kron(np.eye(3), D1)
    XtX, XtY, DtD = X.T @ X, X.T @ d, D.T @ D
    n = len(src)
    best = None
    for lam in lams:
        A = XtX + lam * DtD + 1e-9 * np.eye(XtX.shape[0])
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        c = Ainv @ XtY
        rss = float(((X @ c - d) ** 2).sum())
        tr = float(np.einsum("ij,ji->", X @ Ainv, X.T))
        gcv = n * rss / max(n - tr, 1.0) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, lam, c)
    return PSplineModel("pspline", knots, ucentre, uscale, best[2], float(best[1]))


def pspline_trimmed(src, ref, thresh, knot_px, domain=None, ucentre=None, uscale=None, rounds=6):
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if domain is None:
        domain = (src[:, 1].min(), src[:, 1].max())
    if ucentre is None:
        ucentre = float(src[:, 0].mean())
    if uscale is None:
        uscale = float(src[:, 0].std()) or 1.0
    keep = np.ones(len(src), bool)
    m = None
    for _ in range(rounds):
        if keep.sum() < 10:
            return None
        m = _pspline_fit(src[keep], ref[keep], knot_px, domain, ucentre, uscale)
        r = np.linalg.norm(m.predict(src) - ref, axis=1)
        new = r < (thresh if _ > 1 else 4 * thresh)  # first rounds generous, then the declared threshold
        if (new == keep).all():
            break
        keep = new
    r = np.linalg.norm(m.predict(src) - ref, axis=1)
    return {"model": m, "inliers": r < thresh, "residuals": r}



# ---------------------------------------------------------------------------
# P-spline + terrain relief parallax.
#
# d(u, v) = spline(u, v) + (h - h0) * p, with p a constant 2-vector (px of
# displacement per metre of height). h is the reference DTM height at the
# node's reference position, which for fitting is observed (ref xy). For
# prediction at an arbitrary source point the reference position is unknown,
# so h is looked up at x + d iteratively (3 rounds) through `height_at`, a
# callable mapping reference-frame xy -> height (NaN outside the DTM).
# The physically expected p is tan(emission) along the look direction
# divided by the ground sample distance; a fit that recovers the label's
# viewing angle is an independent check on the model.
# ---------------------------------------------------------------------------

@dataclass
class PSplineReliefModel:
    kind: str
    spline: PSplineModel
    p: np.ndarray
    h0: float
    height_at: object = None

    def predict(self, xy, heights=None):
        xy = np.asarray(xy, dtype=np.float64)
        base = self.spline.predict(xy)
        if heights is not None:
            return base + (np.asarray(heights) - self.h0)[:, None] * self.p[None, :]
        out = base
        for _ in range(3):
            h = self.height_at(out)
            h = np.where(np.isfinite(h), h, self.h0)
            out = base + (h - self.h0)[:, None] * self.p[None, :]
        return out

    def params(self):
        return {"kind": self.kind, "spline": self.spline.params(), "parallax_px_per_m": self.p.tolist(), "h0": self.h0}


def pspline_relief_trimmed(src, ref, thresh, knot_px, height_at, domain=None, ucentre=None, uscale=None, rounds=6):
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    h = height_at(ref)
    ok = np.isfinite(h)
    if ok.sum() < 20:
        return None
    if domain is None:
        domain = (src[:, 1].min(), src[:, 1].max())
    if ucentre is None:
        ucentre = float(src[:, 0].mean())
    if uscale is None:
        uscale = float(src[:, 0].std()) or 1.0
    h0 = float(np.nanmedian(h))
    keep = ok.copy()
    for it in range(rounds):
        if keep.sum() < 20:
            return None
        # alternate: spline on relief-corrected data, then p by least squares on spline residuals
        p = np.zeros(2) if it == 0 else p
        corr = ref - (np.where(ok, h, h0) - h0)[:, None] * p[None, :]
        sp = _pspline_fit(src[keep], corr[keep], knot_px, domain, ucentre, uscale)
        resid = ref - sp.predict(src)
        hc = np.where(ok, h, h0) - h0
        p = np.linalg.lstsq(hc[keep][:, None], resid[keep], rcond=None)[0].ravel()
        m = PSplineReliefModel("pspline_relief", sp, p, h0, height_at)
        r = np.linalg.norm(m.predict(src, heights=np.where(ok, h, h0)) - ref, axis=1)
        new = ok & (r < (thresh if it > 1 else 4 * thresh))
        if it > 2 and (new == keep).all():
            break
        keep = new
    # final joint refit: spline + p together on the kept set
    sp_proto = PSplineModel("pspline", sp.knots, ucentre, uscale, None, sp.lam)
    X = sp_proto._design(src)
    hc = np.where(ok, h, h0) - h0
    Xf = np.hstack([X, hc[:, None]])
    d = ref - src
    nb = len(sp.knots) - 4
    D = np.kron(np.eye(3), np.diff(np.eye(nb), n=2, axis=0))
    P = np.zeros((Xf.shape[1], Xf.shape[1]))
    P[:X.shape[1], :X.shape[1]] = sp.lam * (D.T @ D)
    A = Xf[keep].T @ Xf[keep] + P + 1e-9 * np.eye(Xf.shape[1])
    c = np.linalg.solve(A, Xf[keep].T @ d[keep])
    sp = PSplineModel("pspline", sp.knots, ucentre, uscale, c[:-1], sp.lam)
    m = PSplineReliefModel("pspline_relief", sp, c[-1], h0, height_at)
    r = np.linalg.norm(m.predict(src, heights=np.where(ok, h, h0)) - ref, axis=1)
    r[~ok] = np.inf
    return {"model": m, "inliers": r < thresh, "residuals": r}
