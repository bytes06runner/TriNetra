"""
tests/test_corroboration.py — Stage N independent phase-correlation estimator.

Checks the structural-independence contract, the shift sign convention, known-
truth recovery on non-circular windows, failure on noise, and that the frozen
Stage N results file still says what UPGRADE_N_CORROBORATION.md reports.
"""

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "src" / "corroboration" / "phase_correlation.py"
RESULTS_JSON = REPO_ROOT / "assets" / "real_cache" / "stage_n_corroboration_results.json"

# Load by path: the `src` package name is rebound by the LoFTR helper elsewhere in the suite.
_spec = importlib.util.spec_from_file_location("stage_n_pc_under_test", MODULE_PATH)
pcm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pcm)


def _texture(shape, seed=0, smooth=3.0):
    """Band-limited random terrain-like texture (no repo data needed)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal(shape))
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    f *= np.exp(-((fx ** 2 + fy ** 2) * (2 * np.pi * smooth) ** 2) / 2)
    img = np.real(np.fft.ifft2(f))
    return (img - img.min()) / (img.max() - img.min()) * 255.0


def test_module_imports_only_numpy_and_stdlib():
    """Independence contract: no OpenCV, scipy, or project imports."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are forbidden in the independent estimator"
            roots.add(node.module.split(".")[0])
    assert roots <= {"numpy", "typing"}, f"forbidden imports: {roots - {'numpy', 'typing'}}"


def test_module_shares_no_primary_pipeline_calls():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for token in ("matchTemplate", "estimateAffinePartial2D", "findHomography",
                  "compute_phase_congruency", "phaseCorrelate", "evaluate_flight_gate"):
        assert token not in src


@pytest.mark.parametrize("dx,dy", [(3.3, -7.6), (0.5, 0.5), (-20.25, 11.9)])
def test_circular_shift_recovery_and_sign(dx, dy):
    img = _texture((256, 256), seed=1)
    mov = pcm.fourier_shift(img, dx, dy)
    # unwindowed: a circular shift is exact for phase correlation
    r = pcm.estimate_shift(img, mov, window=False)
    assert abs(r["dx"] - dx) < 0.02 and abs(r["dy"] - dy) < 0.02
    back = pcm.estimate_shift(mov, img, window=False)
    assert abs(back["dx"] + dx) < 0.02 and abs(back["dy"] + dy) < 0.02
    # windowed (the setting used in Stage N): the Hann window does not move with
    # circularly shifted content, giving a known bias of up to ~0.13 px on this
    # very smooth texture
    w = pcm.estimate_shift(img, mov)
    assert abs(w["dx"] - dx) < 0.25 and abs(w["dy"] - dy) < 0.25


def test_noncircular_window_shift_recovery():
    """Two different windows of one larger texture: real edge effects, known truth."""
    big = _texture((600, 600), seed=2)
    ref = big[100:356, 100:356]
    for Dx, Dy in [(13, -9), (-31, 22), (0, 40)]:
        mov = big[100 + Dy:356 + Dy, 100 + Dx:356 + Dx]
        r = pcm.estimate_shift(ref, mov)
        # content at p in ref sits at p - D in the displaced window
        assert abs(r["dx"] + Dx) < 0.5 and abs(r["dy"] + Dy) < 0.5
        assert r["psr"] > 20.0


def test_noise_reference_is_not_significant():
    img = _texture((256, 256), seed=3)
    rng = np.random.default_rng(4)
    psr_noise = [pcm.estimate_shift(img, rng.uniform(0, 255, img.shape))["psr"] for _ in range(20)]
    psr_real = pcm.estimate_shift(img, pcm.fourier_shift(img, 5.0, -3.0))["psr"]
    assert max(psr_noise) < 12.0
    assert psr_real > 10 * max(psr_noise)


def test_lowpass_keeps_shift():
    img = _texture((256, 256), seed=5)
    r = pcm.estimate_shift(img, pcm.fourier_shift(img, 6.4, -2.2), lowpass_cyc_per_px=0.1)
    assert abs(r["dx"] - 6.4) < 0.1 and abs(r["dy"] + 2.2) < 0.1


def test_similarity_displacement_and_warp():
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0, 0, 1]])
    assert pcm.similarity_displacement(H, (10.0, 20.0)) == (5.0, -3.0)
    img = _texture((64, 64), seed=6)
    assert np.allclose(pcm.warp_linear_about_centre(img, np.eye(2))[1:-1, 1:-1], img[1:-1, 1:-1])


@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="Stage N results not generated")
def test_stage_n_frozen_findings():
    """Lock the reported Stage N findings to the results file."""
    d = json.loads(RESULTS_JSON.read_text())
    for hop in ("hop1", "hop2"):
        for v in d[hop]["variants"].values():
            assert v["verdict"]["significant"] is False
            assert v["verdict"]["agree_within_tol"] is False
    assert not any(c["significant"] for t in d["pitiscus"]["tiles"] for c in t["comparisons"].values())
    strip = d["pitiscus_strip_posthoc"]
    assert strip["whole_strip"]["vs_primary_bulk"]["significant"] is True
    assert strip["whole_strip"]["vs_primary_bulk"]["gap_px"] > 500
    assert all(s["vs_primary_bulk"]["significant"] for s in strip["segments"])
    ncc = strip["ncc_lag_check"]
    assert ncc["lags"]["phase_correlation"]["ncc_raw_mean"] > ncc["random_lag_raw_mean_max"]
    assert ncc["lags"]["stageK_applied_bulk"]["ncc_raw_mean"] < ncc["random_lag_raw_mean_max"]
    mi = d["matched_illumination"]
    assert mi["M2a_circular"]["pc_vs_primary"]["all_within_tol"] is True
    assert mi["M2b_noncircular"]["pc_vs_truth"]["all_within_tol"] is True
