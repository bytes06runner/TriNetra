"""
tests/test_provenance.py — Unit test for cache provenance assertions.
Verifies that assert_cache_provenance raises ValueError if arrays or metadata
leak across different cache source files.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_finetuned_checkpoint import assert_cache_provenance


def test_provenance_valid():
    """Test that matching arrays and metadata pass provenance check without error."""
    source_data = {
        'disp_ohrc': np.ones((100, 100), dtype=np.uint8),
        'disp_tmc': np.zeros((100, 100), dtype=np.uint8),
        'center_lat': -89.7207,
        'center_lon': 223.1257,
        'ohrc_gsd': 0.24,
        'tmc_gsd': 4.25,
        'ohrc_product': 'ch2_ohr_ncp_20241115T1525004388',
        'tmc_product': 'ch2_tmc_ncn_20231205T1906512971',
    }
    arrays = {
        'disp_ohrc': np.ones((100, 100), dtype=np.uint8),
        'disp_tmc': np.zeros((100, 100), dtype=np.uint8),
    }
    metadata = {
        'center_lat': -89.7207,
        'ohrc_res': 0.24,
        'tmc_res': 4.25,
        'ohrc_product_id': 'ch2_ohr_ncp_20241115T1525004388',
        'tmc_product_id': 'ch2_tmc_ncn_20231205T1906512971',
    }
    # Should not raise
    assert_cache_provenance(source_data, arrays, metadata)


def test_provenance_mismatched_latitude_raises():
    """Test that mismatched latitude (e.g. copying Shiv Shakti lat into Shackleton cache) raises ValueError."""
    source_data = {
        'disp_ohrc': np.ones((50, 50), dtype=np.uint8),
        'center_lat': -89.7207,
        'ohrc_gsd': 0.24,
    }
    arrays = {
        'disp_ohrc': np.ones((50, 50), dtype=np.uint8),
    }
    # Leaked latitude from Shiv Shakti Point (-69.58019)
    metadata = {
        'center_lat': -69.58019,
        'ohrc_res': 0.24,
    }
    with pytest.raises(ValueError, match='Provenance violation.*latitude'):
        assert_cache_provenance(source_data, arrays, metadata)


def test_provenance_mismatched_gsd_raises():
    """Test that mismatched GSD (e.g. 0.26 instead of 0.24) raises ValueError."""
    source_data = {
        'disp_ohrc': np.ones((50, 50), dtype=np.uint8),
        'center_lat': -89.7207,
        'ohrc_gsd': 0.24,
    }
    arrays = {
        'disp_ohrc': np.ones((50, 50), dtype=np.uint8),
    }
    metadata = {
        'center_lat': -89.7207,
        'ohrc_res': 0.26,  # wrong GSD
    }
    with pytest.raises(ValueError, match='Provenance violation.*OHRC GSD'):
        assert_cache_provenance(source_data, arrays, metadata)


def test_provenance_mismatched_array_raises():
    """Test that modified or mismatched array raises ValueError."""
    source_data = {
        'disp_ohrc': np.ones((50, 50), dtype=np.uint8),
        'center_lat': -89.7207,
        'ohrc_gsd': 0.24,
    }
    arrays = {
        'disp_ohrc': np.zeros((50, 50), dtype=np.uint8),  # wrong array
    }
    metadata = {
        'center_lat': -89.7207,
        'ohrc_res': 0.24,
    }
    with pytest.raises(ValueError, match="Provenance violation.*Array 'disp_ohrc'"):
        assert_cache_provenance(source_data, arrays, metadata)


def test_no_status_emojis_in_results_md():
    """J6a requirement: Verify that no non-ASCII status glyphs/emojis exist in results/*.md tables."""
    results_dir = REPO_ROOT / "results"
    md_files = list(results_dir.glob("*.md"))
    assert len(md_files) > 0, "No markdown files found in results/"

    forbidden_status_glyphs = {"✅", "❌", "⚠️", "🛑", "🔒", "📊", "📈", "🚀", "🟢", "🔴", "🟡"}

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                # Check markdown table rows for forbidden status glyphs
                if line.strip().startswith("|"):
                    for glyph in forbidden_status_glyphs:
                        assert glyph not in line, (
                            f"Non-ASCII status glyph '{glyph}' found in table row at {md_file.name}:{line_no}: {line.strip()}"
                        )


def test_cache_file_separation():
    """J7 requirement: Verify both Shiv Shakti and Shackleton cache files exist distinctly."""
    cache_dir = REPO_ROOT / "assets" / "real_cache"
    shiv_file = cache_dir / "real_flight_hop1_shivshakti.npz"
    shack_file = cache_dir / "real_flight_hop1_shackleton_sift.npz"

    assert shiv_file.exists(), f"Shiv Shakti cache file missing: {shiv_file}"
    assert shack_file.exists(), f"Shackleton SIFT cache file missing: {shack_file}"

    data_shiv = np.load(shiv_file, allow_pickle=True)
    data_shack = np.load(shack_file, allow_pickle=True)

    lat_shiv = float(data_shiv["target_lat"]) if "target_lat" in data_shiv else float(data_shiv["center_lat"])
    lat_shack = float(data_shack["target_lat"]) if "target_lat" in data_shack else float(data_shack["center_lat"])

    assert abs(lat_shiv - (-69.58019)) < 0.05, f"Expected Shiv Shakti lat ~ -69.58, got {lat_shiv}"
    assert abs(lat_shack - (-89.7207)) < 0.05, f"Expected Shackleton lat ~ -89.72, got {lat_shack}"


def test_forbidden_hub_and_scale_claims():
    """Verify that forbidden hub / false bridging phrases do not appear in reports or code."""
    forbidden_phrases = [
        "intermediate scale anchor",
        "via TMC-2",
        "scale bridge",
        "18.15x for Hop 1",
        "composed into the final correspondence",
        "114.6",
        "15.8°",
        "15.8 deg",
    ]
    # UPGRADE_U4_NAC.md documents the post-mortem audit and derivation of why 114.6° / 15.8° were stale
    files_to_check = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "app.py",
        REPO_ROOT / "src" / "trinetra" / "evaluate.py",
        *[f for f in (REPO_ROOT / "results").glob("*.md") if "UPGRADE_U4" not in f.name],
    ]
    for filepath in files_to_check:
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        for phrase in forbidden_phrases:
            assert phrase.lower() not in content.lower(), (
                f"Forbidden phrase '{phrase}' found in {filepath.relative_to(REPO_ROOT)}"
            )


def test_reported_pass_requires_recorded_delta_shuffle():
    """G3 requirement: Verify that any reported PASS requires a recorded Delta_shuffle >= +15.0%.
    Under the six-criterion spaceflight validity gate, any fit lacking a recorded Delta_shuffle
    or with Delta_shuffle < +15.0% must be GATED.
    """
    if "src" in sys.modules and hasattr(sys.modules["src"], "loftr"):
        del sys.modules["src"]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import src.module4_registration.registration as reg_mod
    evaluate_flight_gate = reg_mod.evaluate_flight_gate

    # 1. Calling evaluate_flight_gate with require_delta_shuffle=True fails if missing
    res_missing = evaluate_flight_gate(
        inliers=50, total_matches=100, inlier_ratio_pct=50.0, require_delta_shuffle=True
    )
    assert res_missing["is_gated"] is True
    assert res_missing["status"] == "GATED"
    assert res_missing["first_failing_criterion"] == "missing_delta_shuffle"

    # 2. Negative controls fail (e.g. Hop 1: -5.15%, Hop 2: -2.96%, U4 NAC: -5.17%)
    res_hop1 = evaluate_flight_gate(
        inliers=49, total_matches=217, inlier_ratio_pct=22.58, delta_shuffle=-5.15
    )
    assert res_hop1["is_gated"] is True
    assert res_hop1["status"] == "GATED"
    assert res_hop1["first_failing_criterion"] == "delta_shuffle"

    res_hop2 = evaluate_flight_gate(
        inliers=124, total_matches=307, inlier_ratio_pct=40.39, delta_shuffle=-2.96
    )
    assert res_hop2["is_gated"] is True
    assert res_hop2["status"] == "GATED"
    assert res_hop2["first_failing_criterion"] == "delta_shuffle"

    # 3. Only runs with Delta_shuffle >= +15.0% may pass
    res_pass = evaluate_flight_gate(
        inliers=50, total_matches=100, inlier_ratio_pct=50.0, delta_shuffle=18.5
    )
    assert res_pass["is_gated"] is False
    assert res_pass["status"] == "PASSED"



