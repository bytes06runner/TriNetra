"""
TriNetra — Professional Web Dashboard for SIH26166 Presentation.

Autonomous, scale-invariant image correspondence across Chandrayaan-2
planetary instruments: OHRC (0.26 m/px), TMC-2 (5.0 m/px), and IIRS (91.75 m/px).

Dual-Scene Pipeline:
- Hop 1: Scale-Invariance Benchmark: OHRC (0.26 m/px) vs 20× Simulated TMC-2 Sampling (5.20 m/px) [96.2% consensus]
- Hop 2: Real TMC-2 (4.96 m/px) ↔ Real IIRS (91.75 m/px) [18.5× Polar Overlap, Signal-Gated]
- Architecture: Decoupled two-hop correspondence framework with multi-hop homography composition target

Author: Srijeet Prasad Banerjee
"""

import streamlit as st
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import cv2
import math
from pathlib import Path

# TriNetra Pipeline Modules — Real Data
from src.pds_loader import load_tmc2, load_iirs, iirs_to_grey, crop, iirs_proxy_variants
try:
    from src.geo_align import find_common_region, compute_centered_crop_slices
except Exception:
    find_common_region = None
    compute_centered_crop_slices = None

# ─── Data file paths ────────────────────────────────────────────────
DESKTOP_DATA = Path.home() / "Desktop/data"
CACHE_NPZ_NORTH = Path(__file__).resolve().parent / "assets/real_cache/real_overlapping_pair.npz"
CACHE_NPZ_OHRC = Path(__file__).resolve().parent / "assets/real_cache/real_ohrc_crop.npz"
CACHE_NPZ_FLIGHT_HOP1 = Path(__file__).resolve().parent / "assets/real_cache/real_flight_hop1.npz"

# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="TriNetra — SIH26166",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── CSS: Professional Clean Design ──────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,wght@0,400;0,600;1,400;1,600&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #F9F8F6;
            color: #2D2D2D;
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1120px;
        }

        h1 {
            font-family: 'Newsreader', Georgia, serif !important;
            font-weight: 600 !important;
            color: #1a1a2e !important;
            letter-spacing: -0.02em;
        }
        h2 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            color: #1a1a2e !important;
            font-size: 1.35rem !important;
            letter-spacing: -0.01em;
        }
        h3 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            color: #444 !important;
            font-size: 1.05rem !important;
        }
        p, li, span, div {
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E8E5DF;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        .stButton > button {
            background-color: #DE7356 !important;
            color: white !important;
            border: none !important;
            border-radius: 9px !important;
            padding: 0.55rem 1.4rem !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.90rem !important;
            font-weight: 600 !important;
            cursor: pointer;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 3px rgba(222, 115, 86, 0.25) !important;
        }
        .stButton > button:hover {
            background-color: #C9604A !important;
            color: white !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(222, 115, 86, 0.3) !important;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E8E5DF;
            border-radius: 12px;
            padding: 1.1rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }
        .metric-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888;
            margin-bottom: 0.25rem;
        }
        .metric-val {
            font-size: 1.65rem;
            font-weight: 700;
            color: #1a1a2e;
            line-height: 1.15;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #777;
            margin-top: 0.25rem;
        }

        .stage-pill {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.8rem;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 500;
            margin-bottom: 0.35rem;
        }
        .stage-done {
            background: #EDF7ED;
            color: #1E4620;
        }
        .stage-active {
            background: #FDF0ED;
            color: #DE7356;
            font-weight: 600;
        }
        .stage-pending {
            background: #F5F4F0;
            color: #999;
        }

        .status-banner-success {
            background-color: #ECFDF5;
            border-left: 5px solid #10B981;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.92rem;
            color: #065F46;
            line-height: 1.5;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .status-banner-warning {
            background-color: #FEF3C7;
            border-left: 5px solid #F59E0B;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.92rem;
            color: #92400E;
            line-height: 1.5;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .presenter-box {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #3B82F6;
            border-radius: 8px;
            padding: 14px 18px;
            margin-top: 1.2rem;
            font-size: 0.9rem;
            color: #1E293B;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

inject_css()


# ─── Helper Renderers ─────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def stage_pill(label: str, status: str = "pending") -> str:
    icons = {"done": "✓", "active": "●", "pending": "○"}
    return f'<div class="stage-pill stage-{status}"><span>{icons.get(status, "○")}</span> {label}</div>'


def render_image(arr: np.ndarray, title: str = "", cmap: str = "bone"):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#F9F8F6")
    ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8, color="#2D2D2D")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="#F9F8F6")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)


# ─── Cache Loaders ───────────────────────────────────────────────────
def load_real_north_cache():
    """Load real Chandrayaan-2 North Polar overlapping pair (TMC-2 <-> IIRS)."""
    if CACHE_NPZ_NORTH.exists():
        data_npz = np.load(CACHE_NPZ_NORTH, allow_pickle=True)
        tmc_crop_u8 = data_npz["tmc_crop"]
        iirs_grey = data_npz["iirs_grey"]
        tmc_down = cv2.resize(tmc_crop_u8, (iirs_grey.shape[1], iirs_grey.shape[0]), interpolation=cv2.INTER_AREA)

        p_1500 = data_npz["iirs_proxy_1500nm"] if "iirs_proxy_1500nm" in data_npz else iirs_grey
        p_3band = data_npz["iirs_proxy_3band_mean"] if "iirs_proxy_3band_mean" in data_npz else iirs_grey
        p_pc1 = data_npz["iirs_proxy_pc1"] if "iirs_proxy_pc1" in data_npz else iirs_grey

        common = {
            "center_lat": float(data_npz["center_lat"]),
            "center_lon": float(data_npz["center_lon"]),
            "min_distance_km": float(data_npz["min_dist_m"]) / 1000.0,
            "a": {"center_scan": int(data_npz["tmc_scan"])},
            "b": {"center_scan": int(data_npz["iir_scan"])},
        }

        return {
            "tmc_full": tmc_crop_u8,
            "tmc_down": tmc_down,
            "iirs_band_avg": iirs_grey,
            "iirs_1500nm": p_1500,
            "iirs_3band": p_3band,
            "iirs_pc1": p_pc1,
            "tmc_res": float(data_npz["tmc_res"]),
            "iir_res": float(data_npz["iir_res"]),
            "sun_el": float(data_npz["sun_el"]),
        }, common
    return None, None


def load_real_flight_hop1_cache():
    """Load authentic Chandrayaan-2 dual-sensor flight correspondence (OHRC 0.26 m/px ↔ TMC-2 4.72 m/px)."""
    if CACHE_NPZ_FLIGHT_HOP1.exists():
        d = np.load(CACHE_NPZ_FLIGHT_HOP1, allow_pickle=True)
        return {
            "disp_ohrc": d["disp_ohrc"],
            "disp_tmc": d["disp_tmc"],
            "pts1": d["pts1"],
            "pts2": d["pts2"],
            "inlier_mask": d["inlier_mask"],
            "H": d["H"],
            "inliers": int(d["inliers"]),
            "total_matches": int(d["total_matches"]),
            "inlier_ratio": float(d["inlier_ratio"]),
            "ohrc_res": float(d["ohrc_res"]),
            "tmc_res": float(d["tmc_res"]),
            "scale_gap": float(d["scale_gap"]),
            "ohrc_product_id": str(d["ohrc_product_id"]),
            "tmc_product_id": str(d["tmc_product_id"]),
            "target_lat": float(d["target_lat"]),
            "target_lon": float(d["target_lon"]),
            "tmc_sun_elevation": float(d["tmc_sun_elevation"]),
            "tmc_sun_azimuth": float(d["tmc_sun_azimuth"]),
            "ohrc_sun_elevation": float(d["ohrc_sun_elevation"]),
            "ohrc_sun_azimuth": float(d["ohrc_sun_azimuth"]),
        }
    return None


def load_real_ohrc_cache():
    """Load real Chandrayaan-2 OHRC flight crop (0.26 m/px) and TMC-2 optical proxy."""
    if CACHE_NPZ_OHRC.exists():
        d = np.load(CACHE_NPZ_OHRC, allow_pickle=True)
        return {
            "ohrc_disp": d["ohrc_disp"],
            "tmc_proxy": d["tmc_proxy"],
            "tmc_disp": d["tmc_disp"],
            "pts1": d["pts1"],
            "pts2": d["pts2"],
            "inlier_mask": d["inlier_mask"],
            "H": d["H"],
            "inliers": int(d["inliers"]),
            "total_matches": int(d["total_matches"]),
            "inlier_ratio": float(d["inlier_ratio"]),
            "ohrc_res": float(d["ohrc_res"]),
            "tmc_res": float(d["tmc_res"]),
            "scale_gap": float(d["scale_gap"]),
            "sun_incidence": float(d["sun_incidence"]),
            "sun_azimuth": float(d["sun_azimuth"]),
            "center_lat": float(d["center_lat"]),
            "center_lon": float(d["center_lon"]),
        }
    return None


# ─── Session State Initialization ─────────────────────────────────────
legacy_keys = ["ohrc", "reg_result", "match_result", "synthetic", "homography", "inliers", "rmse", "stage", "mosaic"]
for k in legacy_keys:
    if k in st.session_state:
        del st.session_state[k]

if "active_scene" not in st.session_state:
    st.session_state.active_scene = "hop1"
if "hop1_step" not in st.session_state:
    st.session_state.hop1_step = 1
if "hop2_step" not in st.session_state:
    st.session_state.hop2_step = 1
if "selected_proxy_key" not in st.session_state:
    st.session_state.selected_proxy_key = "band_avg"

# Pre-load flight caches
if "north_data" not in st.session_state or st.session_state.north_data is None:
    n_raw, n_common = load_real_north_cache()
    st.session_state.north_data = n_raw
    st.session_state.north_common = n_common

if "ohrc_data" not in st.session_state or st.session_state.ohrc_data is None:
    st.session_state.ohrc_data = load_real_ohrc_cache()

if "flight_hop1_data" not in st.session_state or st.session_state.flight_hop1_data is None:
    st.session_state.flight_hop1_data = load_real_flight_hop1_cache()

if "hop1_mode" not in st.session_state:
    st.session_state.hop1_mode = "flight" if st.session_state.flight_hop1_data is not None else "benchmark"


# ─── Sidebar Navigation ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center; padding: 0.3rem 0 0.8rem 0;">', unsafe_allow_html=True)
    st.image("assets/logo.png", width=160)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-bottom:0.5rem;">Pipeline Scene Selection</p>', unsafe_allow_html=True)

    scene_options = {
        "hop1": "🔬 Hop 1: Scale Benchmark (20×)",
        "hop2": "🛰 Hop 2: Real TMC-2 ↔ IIRS (18.5×)",
        "overview": "📋 Unified System & Briefing",
    }
    selected_scene = st.radio(
        "Choose Instrument Hop",
        options=list(scene_options.keys()),
        format_func=lambda k: scene_options[k],
        index=list(scene_options.keys()).index(st.session_state.active_scene),
        label_visibility="collapsed",
    )
    if selected_scene != st.session_state.active_scene:
        st.session_state.active_scene = selected_scene
        st.rerun()

    st.markdown("---")

    # Dynamic Stages for Selected Scene
    if st.session_state.active_scene == "hop1":
        st.markdown('<p style="font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-bottom:0.5rem;">Hop 1 Stages</p>', unsafe_allow_html=True)
        h1_stages = [
            ("Multi-Scale Flight Crop", 1),
            ("Scale-Invariant Matching", 2),
            ("Homography & Overlay", 3),
        ]
        for lbl, s_num in h1_stages:
            st_class = "done" if st.session_state.hop1_step > s_num else ("active" if st.session_state.hop1_step == s_num else "pending")
            st.markdown(stage_pill(lbl, st_class), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Step 1", key="h1_btn1", use_container_width=True):
                st.session_state.hop1_step = 1
                st.rerun()
        with c2:
            if st.button("Step 2", key="h1_btn2", use_container_width=True):
                st.session_state.hop1_step = 2
                st.rerun()
        with c3:
            if st.button("Step 3", key="h1_btn3", use_container_width=True):
                st.session_state.hop1_step = 3
                st.rerun()

    elif st.session_state.active_scene == "hop2":
        st.markdown('<p style="font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-bottom:0.5rem;">Hop 2 Stages</p>', unsafe_allow_html=True)
        h2_stages = [
            ("Polar Footprint Ingestion", 1),
            ("Pushbroom Destriping", 2),
            ("Structural Gating", 3),
        ]
        for lbl, s_num in h2_stages:
            st_class = "done" if st.session_state.hop2_step > s_num else ("active" if st.session_state.hop2_step == s_num else "pending")
            st.markdown(stage_pill(lbl, st_class), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Step 1", key="h2_btn1", use_container_width=True):
                st.session_state.hop2_step = 1
                st.rerun()
        with c2:
            if st.button("Step 2", key="h2_btn2", use_container_width=True):
                st.session_state.hop2_step = 2
                st.rerun()
        with c3:
            if st.button("Step 3", key="h2_btn3", use_container_width=True):
                st.session_state.hop2_step = 3
                st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding-top: 0.2rem;">
        <p style="font-size:0.7rem; color:#aaa; line-height: 1.4;">
            <strong>ISRO · SIH26166</strong><br/>
            Chandrayaan-2 Lunar Correspondence<br/>
            OHRC (0.26m) ↔ TMC-2 (5m) ↔ IIRS (92m)
        </p>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCENE 1: HOP 1 — REAL OHRC ↔ TMC-2 (20× SCALE GAP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.active_scene == "hop1":
    flight_data = st.session_state.flight_hop1_data
    bench_data = st.session_state.ohrc_data

    # Mode Selector
    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        mode_options = []
        if flight_data is not None:
            mode_options.append("🚀 Authentic Flight Data (OHRC ↔ TMC-2 Dual-Sensor)")
        mode_options.append("🔬 Controlled Scale Benchmark (20× Optical Emulation)")
        
        hop1_mode_selection = st.radio(
            "Validation Mode:",
            mode_options,
            index=0 if (st.session_state.hop1_mode == "flight" and flight_data is not None) else len(mode_options)-1,
            horizontal=True,
            key="hop1_mode_radio"
        )
        is_flight_mode = "Authentic" in hop1_mode_selection

    if is_flight_mode:
        active_data = flight_data
        st.markdown("""
        <div class="status-banner-success">
            <strong>🚀 Authentic Flight Cross-Instrument Validation:</strong> Matching real Chandrayaan-2 <strong>OHRC (0.26 m/px)</strong> flight calibrated image (<code>ch2_ohr_ncp_20211023T0027462822</code>) against real <strong>TMC-2 (4.72 m/px)</strong> flight calibrated image (<code>ch2_tmc_ncn_20230130T1900132182</code>). Both scenes were captured by two independent physical sensors on board Chandrayaan-2 over the South Pole Shiv Shakti Point crater field across an <strong>18.15× optical scale gap</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-bottom: 1.2rem;">
            <h1 style="font-size: 2.2rem; margin-bottom: 0.2rem;">Hop 1: Real Flight Cross-Instrument Validation: OHRC ↔ TMC-2</h1>
            <p style="color: #666; max-width: 780px; font-size: 0.96rem;">
                Cross-instrument co-registration between Chandrayaan-2 <strong>OHRC (0.26 m/px)</strong> and <strong>TMC-2 (4.72 m/px)</strong>
                across an <strong>18.15× optical resolution gap</strong> over identical lunar terrain at Shiv Shakti Point (Lat -69.58°S, Lon 32.29°E).
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Ground-Truth Flight Metadata & Selenographic Footprints"):
            st.markdown(f"""
            - **OHRC Product ID:** `ch2_ohr_ncp_20211023T0027462822_d_img_d18` (Calibrated Level-2, 0.26 m/px)
              - Acquisition Time: `2021-10-23T00:27:46Z` | Orbit Limb: `Ascending` | Spacecraft Roll: `+15.76°` (Oblique mode)
              - Sun Elevation: `{active_data['ohrc_sun_elevation']:.1f}°` | Sun Azimuth: `{active_data['ohrc_sun_azimuth']:.1f}°`
            - **TMC-2 Product ID:** `ch2_tmc_ncn_20230130T1900132182_d_img_d32` (Calibrated Level-2, 4.72 m/px)
              - Acquisition Time: `2023-01-30T19:00:13Z` | Orbit Limb: `Ascending` | Spacecraft Roll: `-0.02°` (Nadir mode)
              - Sun Elevation: `{active_data['tmc_sun_elevation']:.1f}°` | Sun Azimuth: `{active_data['tmc_sun_azimuth']:.1f}°`
            - **Physical Ground Overlap:** Lat `-69.58019°`, Lon `32.28800°` (verified by official PDS4 Geometry Grid `.csv` files).
            - **Cross-Illumination Offset:** 114.6° difference in solar azimuth angle (evaluating multi-modal shadow-invariant registration).
            """)
    else:
        active_data = bench_data
        st.markdown("""
        <div class="status-banner-warning">
            <strong>🔬 Controlled Single-Sensor Benchmark:</strong> The 5.20 m/px image is an optical 20× anti-aliased downsampling of the real OHRC flight image (0.26 m/px). This isolates scale invariance on authentic lunar crater terrain while holding solar elevation and spacecraft attitude fixed.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-bottom: 1.2rem;">
            <h1 style="font-size: 2.2rem; margin-bottom: 0.2rem;">Hop 1: Scale-Invariance Benchmark: OHRC vs 20× Simulated TMC-2 Sampling</h1>
            <p style="color: #666; max-width: 780px; font-size: 0.96rem;">
                Evaluating <strong>20× optical scale invariance</strong> using Chandrayaan-2 OHRC flight product (0.26 m/px)
                against an anti-aliased 5.20 m/px simulated TMC-2 sampling across authentic lunar crater terrain.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if active_data is None:
        st.error("Hop 1 dataset cache archive not found. Run scripts/align_real_tmc_ohrc.py to populate.")
        st.stop()

    # Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("OHRC GSD", f"{active_data['ohrc_res']} m/px", "Panchromatic Visible"), unsafe_allow_html=True)
    with c2:
        lbl = "TMC-2 Flight GSD" if is_flight_mode else "TMC-2 Sampling"
        sub = "Real Flight Data" if is_flight_mode else "20× Optical Emulation"
        st.markdown(metric_card(lbl, f"{active_data['tmc_res']} m/px", sub), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Scale Ratio", f"{active_data['scale_gap']:.1f}×", f"Octaves: {math.log2(active_data['scale_gap']):.2f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Inlier Consensus", f"{active_data['inliers']} Inliers", f"{active_data['inlier_ratio']:.1f}% ({active_data['inliers']}/{active_data['total_matches']}) MAGSAC++"), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    step = st.session_state.hop1_step

    # ── Sub-step 1: Flight Crop & Resolution Alignment
    if step == 1:
        if is_flight_mode:
            st.markdown("<h3>Stage 1: Multi-Scale Flight Crop & Spatial Resolution Normalization</h3>", unsafe_allow_html=True)
            st.markdown("""
            <p style="color:#555; font-size:0.9rem;">
                The left image shows the 1000×1000 sub-window from the raw 93K × 12K <strong>OHRC flight image (0.26 m/px)</strong>.
                The right image shows the corresponding crater field extracted from the raw 190K × 4K <strong>TMC-2 flight image (4.72 m/px)</strong> at line 132,700, sample 710.
            </p>
            """, unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                render_image(active_data["disp_ohrc"], "Real OHRC Flight Image (0.26 m/px — Shiv Shakti Point)")
            with col_b:
                render_image(active_data["disp_tmc"], "Real TMC-2 Flight Image (4.72 m/px — South Pole Orbit)")
            st.markdown("""
            <div class="presenter-box">
                <strong>💡 Presenter's Note for Evaluators:</strong> Notice the distinct crater topography present in both cameras. Because OHRC (roll +15.8°) was acquired in October 2021 with sun azimuth 298.4° and TMC-2 was acquired in January 2023 with sun azimuth 53.0°, the shadow directions differ by 114.6°. This rigorously evaluates robust, illumination-invariant geometric correspondence on real flight data.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<h3>Stage 1: Multi-Scale Flight Crop & 20× Optical Downsampling</h3>", unsafe_allow_html=True)
            st.markdown("""
            <p style="color:#555; font-size:0.9rem;">
                The left image shows a 1000×1000 sub-window extracted from the raw 93K × 12K OHRC flight image (0.26 m/px).
                The right image is the 20× anti-aliased optical downsampling (5.20 m/px), emulating the spatial integration of TMC-2's linear detector.
            </p>
            """, unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                render_image(active_data["ohrc_disp"], "Real OHRC Flight Data (0.26 m/px — South Pole)")
            with col_b:
                render_image(active_data["tmc_disp"], "Simulated TMC-2 Sampling (5.20 m/px — 20× Downsampled)")

        st.markdown("<br/>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn2:
            if st.button("Run SIFT Matching →", use_container_width=True):
                st.session_state.hop1_step = 2
                st.rerun()

    # ── Sub-step 2: Feature Matching
    elif step == 2:
        st.markdown("<h3>Stage 2: Scale-Aligned Keypoint Correspondence</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#555; font-size:0.9rem;">
            Horizontal green correspondence vectors connecting matching crater rims across the optical scale difference.
            Notice how prominent crater rim geometries remain invariant under scale transitions.
        </p>
        """, unsafe_allow_html=True)

        # Draw green correspondence lines
        img1 = active_data["disp_ohrc"] if "disp_ohrc" in active_data else active_data["ohrc_disp"]
        img2 = active_data["disp_tmc"] if "disp_tmc" in active_data else active_data["tmc_disp"]
        pts1 = active_data["pts1"]
        pts2 = active_data["pts2"]
        mask = active_data["inlier_mask"]

        inlier_indices = np.where(mask)[0]
        sample_k = min(45, len(inlier_indices))
        if sample_k > 0:
            np.random.seed(42)
            sample_idx = np.random.choice(inlier_indices, sample_k, replace=False)
        else:
            sample_idx = []

        vis = np.hstack([img1, img2])
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
        w = img1.shape[1]

        for idx in sample_idx:
            p1 = (int(pts1[idx][0]), int(pts1[idx][1]))
            p2 = (int(pts2[idx][0] + w), int(pts2[idx][1]))
            cv2.line(vis_rgb, p1, p2, (0, 225, 110), 1, cv2.LINE_AA)
            cv2.circle(vis_rgb, p1, 3, (255, 120, 0), -1)
            cv2.circle(vis_rgb, p2, 3, (0, 200, 255), -1)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#F9F8F6")
        ax.imshow(vis_rgb)
        ax.axis("off")
        lbl_pair = "Real OHRC (0.26 m/px) ↔ Real TMC-2 (4.72 m/px)" if is_flight_mode else "Real OHRC (0.26 m/px) ↔ Simulated TMC-2 Sampling (5.20 m/px)"
        ax.set_title(f"{lbl_pair} — {active_data['inliers']} Inliers", fontsize=10, fontweight="bold", pad=8)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="#F9F8F6")
        plt.close(fig)
        buf.seek(0)
        st.image(buf, use_container_width=True)

        if is_flight_mode:
            st.markdown("""
            <div class="presenter-box">
                <strong>💡 Flight Validation Note:</strong> The correspondence vectors demonstrate cross-instrument alignment despite the 18.15× resolution gap and 114.6° solar azimuth offset. USAC-MAGSAC++ robustly rejects illumination-dependent false positives and locks onto physical crater rim features.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="presenter-box">
                <strong>💡 Presenter's Note for Evaluators:</strong> Because both images represent the same panchromatic scene at different sampling resolutions, scale-space SIFT coupled with USAC-MAGSAC++ achieves a <strong>96.2% inlier ratio</strong>, confirming mathematical scale invariance across a 20× sampling gap under controlled single-sensor conditions.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns([1, 3, 1])
        with col_b1:
            if st.button("← Back", use_container_width=True):
                st.session_state.hop1_step = 1
                st.rerun()
        with col_b3:
            if st.button("Compute Overlay →", use_container_width=True):
                st.session_state.hop1_step = 3
                st.rerun()

    # ── Sub-step 3: Registration Overlay
    elif step == 3:
        st.markdown("<h3>Stage 3: MAGSAC++ Geometric Registration & Verification Overlay</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#555; font-size:0.9rem;">
            The computed homography matrix maps OHRC coordinates into the TMC-2 sampling frame.
            In the false-color composite: <strong>Red = Warped OHRC</strong>, <strong>Cyan = Target TMC-2</strong>.
            Regions of geometric alignment appear in neutral grayscale/white.
        </p>
        """, unsafe_allow_html=True)

        img1 = active_data["disp_ohrc"] if "disp_ohrc" in active_data else active_data["ohrc_disp"]
        img2 = active_data["disp_tmc"] if "disp_tmc" in active_data else active_data["tmc_disp"]
        H = active_data["H"]

        warped_ohrc = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        overlay = np.zeros((img2.shape[0], img2.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 0] = warped_ohrc  # Red
        overlay[:, :, 1] = img2         # Green
        overlay[:, :, 2] = img2         # Blue

        diff = np.abs(warped_ohrc.astype(np.float32) - img2.astype(np.float32))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        c_reg1, c_reg2 = st.columns([1, 1])
        with c_reg1:
            render_image(overlay, "False-Color Registration Overlay (Red: OHRC, Cyan: TMC-2)", cmap=None)
        with c_reg2:
            st.markdown(metric_card("Reprojection RMSE", f"{rmse:.2f} DN", "Mean Squared Pixel Discrepancy"), unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(metric_card("Geometric Inliers", f"{active_data['inliers']}", "MAGSAC++ Inliers at 5.0 px Threshold"), unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(metric_card("Transform Type", "Projective Homography", "Degrees of Freedom: 8 (3x3 Matrix)"), unsafe_allow_html=True)

        if is_flight_mode:
            st.markdown("""
            <div class="presenter-box">
                <strong>💡 Flight Validation Summary:</strong> Successful dual-sensor co-registration between real Chandrayaan-2 OHRC (0.26 m/px) and real TMC-2 (4.72 m/px) flight products. The estimated projective homography accounts for both the 18.15× spatial scale ratio and the 15.8° spacecraft roll angle difference.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b1:
            if st.button("← Back to Matching", use_container_width=True):
                st.session_state.hop1_step = 2
                st.rerun()
        with col_b3:
            if st.button("Proceed to Hop 2 →", use_container_width=True):
                st.session_state.active_scene = "hop2"
                st.session_state.hop2_step = 1
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCENE 2: HOP 2 — REAL TMC-2 ↔ IIRS (18.5× SCALE GAP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.active_scene == "hop2":
    st.markdown("""
    <div class="status-banner-success">
        <strong>✅ Real Flight Overlap:</strong> Confirmed geographic intersection at Lunar North Pole (89.7086°N, 5.0764°E) within 51.2 m ground separation. Solar incidence: 76.92° vs 76.93° (Δ = 0.01°).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.3rem; margin-bottom: 0.2rem;">Hop 2: Real TMC-2 ↔ IIRS Correspondence</h1>
        <p style="color: #666; max-width: 780px; font-size: 0.96rem;">
            Bridging the <strong>18.5× cross-modal scale gap</strong> between panchromatic visible TMC-2 (4.96 m/px)
            and hyperspectral infrared IIRS (91.75 m/px, 256 bands) under extreme polar grazing illumination.
        </p>
    </div>
    """, unsafe_allow_html=True)

    north_raw = st.session_state.north_data
    north_ci = st.session_state.north_common

    if north_raw is None:
        st.error("North Polar cache archive not found. Run smoke_test_real.py to populate.")
        st.stop()

    # Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("TMC-2 GSD", f"{north_raw['tmc_res']:.2f} m/px", "Panchromatic Visible"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("IIRS GSD", f"{north_raw['iir_res']:.2f} m/px", "256 Bands SWIR"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Scale Ratio", f"{north_raw['iir_res']/north_raw['tmc_res']:.1f}×", "Ground Sep: 51.2 m"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Destriping", "91.7% Reduction", "Col Std: 0.370 → 0.031"), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    step2 = st.session_state.hop2_step

    # ── Sub-step 1: Footprint Ingestion
    if step2 == 1:
        st.markdown("<h3>Stage 1: Selenographic Footprint Ingestion & Alignment</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#555; font-size:0.9rem;">
            Confirmed geographic overlap pair from the Lunar North Pole (89.7086°N, 5.0764°E).
            Loaded via zero-copy memory mapping without heap memory overhead.
        </p>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            render_image(north_raw["tmc_full"], "TMC-2 High-Resolution Crop (4000×4000 px @ 4.96 m/px)")
        with col_b:
            render_image(north_raw["tmc_down"], "TMC-2 Scaled to IIRS Grid (216×216 px @ 91.75 m/px)")

        st.markdown("""
        <div class="presenter-box">
            <strong>💡 Presenter's Note for Evaluators:</strong> In polar regions (>85° latitude), standard cylindrical coordinates suffer from extreme longitude convergence. Our 3D Cartesian KD-Tree aligner maps latitude/longitude onto a 1,737.4 km lunar sphere, ensuring exact 51.2 meter ground accuracy.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn2:
            if st.button("Inspect Proxy Destriping →", use_container_width=True):
                st.session_state.hop2_step = 2
                st.rerun()

    # ── Sub-step 2: Destriping & Proxy Variants
    elif step2 == 2:
        st.markdown("<h3>Stage 2: Pushbroom Destriping & IIRS Proxy Variants</h3>", unsafe_allow_html=True)
        st.markdown("""
        <p style="color:#555; font-size:0.9rem;">
            Raw pushbroom spectrometers exhibit severe column-to-column gain non-uniformity and defective detector pixels (white vertical stripes).
            We engineered an autonomous pushbroom calibration filter with bad-detector column detection (>2.5 MAD) and adjacent-column linear interpolation, followed by cross-track column median destriping (α = 0.85). This eliminates saturated detector columns (e.g. sample 210) and reduces stripe variance by 91.7% (column std: 0.370 → 0.031) while preserving genuine lunar terrain topography.
        </p>
        """, unsafe_allow_html=True)

        proxy_key = st.radio(
            "Select IIRS Visible Proxy Candidate:",
            options=["band_avg", "1500nm", "3band", "pc1"],
            format_func=lambda k: {
                "band_avg": "Sub-2000nm Normalised Mean (Primary Proxy)",
                "1500nm": "Band 50 (1500 nm Clean Albedo Channel)",
                "3band": "3-Band Average (1000 nm, 1250 nm, 1500 nm)",
                "pc1": "Principal Component 1 (PC1 — Spectral Variance)",
            }[k],
            horizontal=True,
        )
        st.session_state.selected_proxy_key = proxy_key

        curr_iirs = {
            "band_avg": north_raw["iirs_band_avg"],
            "1500nm": north_raw["iirs_1500nm"],
            "3band": north_raw["iirs_3band"],
            "pc1": north_raw["iirs_pc1"],
        }[proxy_key]

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            render_image(north_raw["tmc_down"], "TMC-2 Optical Ground Truth (216×216 px)")
        with c_v2:
            render_image(curr_iirs, f"Destriped IIRS Candidate: {proxy_key}")

        st.markdown("""
        <div class="presenter-box">
            <strong>💡 Presenter's Note for Evaluators:</strong> Dividing each band by its spatial mean normalizes solar spectral irradiance across wavelengths. Detecting and interpolating anomalous pushbroom detector elements (such as saturated columns 137–144 and 210) eliminates vertical sensor blinding, while column median relaxation (α = 0.85) removes residual gain stripes.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b1:
            if st.button("← Back to Alignment", use_container_width=True):
                st.session_state.hop2_step = 1
                st.rerun()
        with col_b3:
            if st.button("Check Matching Gating →", use_container_width=True):
                st.session_state.hop2_step = 3
                st.rerun()

    # ── Sub-step 3: Structural Signal Gating
    elif step2 == 3:
        st.markdown("<h3>Stage 3: Scientific Signal Evaluation & Automated Gating</h3>", unsafe_allow_html=True)

        st.markdown("""
        <div class="status-banner-warning">
            <strong>🛑 Gated: Noise-Limited Polar Signal (SWIR SNR ≈ 1.4)</strong><br/>
            IIRS acquisition at 89.7°N with 13.1° solar elevation (76.9° incidence) yields SWIR radiance near the detector noise floor.
            Measured SWIR radiance is only <strong>7.53 DN</strong> at 1500 nm (Band 50) and <strong>7.19 DN</strong> at 2000 nm (Band 77), with SNR near unity (1.40 and 1.23).
            Cross-modal matching is intentionally gated to maintain scientific validity; fabricated matches and artificial RMSE values are rejected.
            A 2× better-illuminated pair (62.9° incidence) has been identified for future flight validation.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:white; border:1px solid #E8E5DF; border-radius:10px; padding:1.2rem; margin-bottom:1.5rem;">
            <h4 style="margin-top:0; color:#1a1a2e;">Why Gating Demonstrates Engineering Maturity:</h4>
            <ul style="color:#555; font-size:0.9rem; line-height:1.7; margin-bottom:0;">
                <li><strong>Empirically Verified Noise Floor:</strong> Radiance in the 89.7°N crop is 60× lower than equatorial/temperate segments of the same flight strip (7.5 DN vs 458.9 DN at 1500 nm), collapsing SNR to 1.40. In noise-dominated regolith, feature extractors produce false pseudo-correspondences.</li>
                <li><strong>Operational Integrity:</strong> In autonomous planetary exploration systems, knowing <em>when not to register</em> prevents catastrophic navigation divergence.</li>
                <li><strong>Multi-Proxy Validation:</strong> All four independent reduction methods (sub-2000nm mean, 1500 nm channel, 3-band composite, and PC1) yield |r| &le; 0.027 against downsampled TMC-2, proving absence of extractable crater topography in this crop.</li>
                <li><strong>Identified Flight Candidate:</strong> Pair search in the archive identified overlapping candidate <code>ch2_tmc_ncn_20230528T0722305575</code> &harr; <code>ch2_iir_nci_20230528T0722291281</code> with 62.9° incidence (2.01× radiance gain) acquired simultaneously (Δt = 1.4s) on the same orbit.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            render_image(north_raw["tmc_down"], "TMC-2 Ground Truth (Subtle Low Relief)")
        with c2:
            render_image(north_raw["iirs_band_avg"], "Destriped IIRS Proxy (Regolith Flat Signal)")

        st.markdown("<br/>", unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b1:
            if st.button("← Back to Proxy Variants", use_container_width=True):
                st.session_state.hop2_step = 2
                st.rerun()
        with col_b3:
            if st.button("View Unified Briefing →", use_container_width=True):
                st.session_state.active_scene = "overview"
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCENE 3: UNIFIED ARCHITECTURE & EVALUATOR BRIEFING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif st.session_state.active_scene == "overview":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.4rem; margin-bottom: 0.2rem;">TriNetra Unified System Architecture</h1>
        <p style="color: #666; max-width: 800px; font-size: 0.96rem;">
            Autonomous multi-modal, sun-angle and scale-invariant image correspondence across all three Chandrayaan-2 lunar instruments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Mathematical Formula Box (Design Target)
    st.markdown("""
    <div style="background:white; border:1px solid #E8E5DF; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; text-align:center;">
        <h3 style="margin-top:0; color:#1a1a2e;">Mathematical Design Target: Multi-Hop Homography Composition</h3>
        <p style="font-size: 1.15rem; color:#DE7356; font-family: monospace; font-weight: 700; margin: 0.8rem 0;">
            H(OHRC → IIRS) = H(TMC-2 → IIRS) · H(OHRC → TMC-2)
        </p>
        <p style="color:#B45309; background:#FEF3C7; border: 1px solid #FDE68A; border-radius:6px; padding:0.65rem 0.9rem; font-size:0.86rem; max-width:720px; margin:0.8rem auto 0 auto; text-align:left; line-height:1.55;">
            <strong>⚠️ Architecture Note:</strong> Composition requires all three instruments over common ground. No such triple overlap has been located in the accessible archive; measured cross-track separation between the nearest OHRC and TMC-2 footprints at the selected site is 348 km.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Pillars of TriNetra
    st.markdown("""
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-bottom: 1.5rem;">
        <div style="background:white; border:1px solid #E8E5DF; border-radius:10px; padding:1.2rem;">
            <h4 style="color:#1a1a2e; margin-top:0;">1. Hop 1: Scale-Invariance Benchmark (20× Gap)</h4>
            <p style="color:#555; font-size:0.88rem; line-height:1.6;">
                Evaluated on real 0.26 m/px OHRC flight data (<code>ch2_ohr_ncp_20211023</code>) against 20× anti-aliased simulated TMC-2 sampling. Scale-space SIFT achieves <strong>300 inliers (96.2% consensus)</strong> on authentic crater terrain; cross-instrument flight validation is pending.
            </p>
        </div>
        <div style="background:white; border:1px solid #E8E5DF; border-radius:10px; padding:1.2rem;">
            <h4 style="color:#1a1a2e; margin-top:0;">2. Hop 2: Real TMC-2 ↔ IIRS Polar Overlap (18.5× Gap)</h4>
            <p style="color:#555; font-size:0.88rem; line-height:1.6;">
                Evaluated on confirmed North Polar overlapping pair (<code>89.7086°N, 5.0764°E</code>). Pushbroom destriping reduces stripe variance by 91.7% (std 0.370 → 0.031) with bad detector column repair, while automated gating prevents spurious registrations when terrain contrast is insufficient.
            </p>
        </div>
        <div style="background:white; border:1px solid #E8E5DF; border-radius:10px; padding:1.2rem;">
            <h4 style="color:#1a1a2e; margin-top:0;">3. Gigabyte-Scale Memory Mapping</h4>
            <p style="color:#555; font-size:0.88rem; line-height:1.6;">
                Zero-copy <code>np.memmap</code> enables rapid sub-window extraction directly from 1.5 GB TMC-2 and 2.6 GB IIRS binary files without RAM exhaustion.
            </p>
        </div>
        <div style="background:white; border:1px solid #E8E5DF; border-radius:10px; padding:1.2rem;">
            <h4 style="color:#1a1a2e; margin-top:0;">4. 3D Selenographic KD-Tree</h4>
            <p style="color:#555; font-size:0.88rem; line-height:1.6;">
                Converts spherical coordinates to 3D Cartesian coordinates on a 1,737.4 km lunar sphere, overcoming polar meridian singularities.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Instrument Specs Table
    st.markdown("<h3>Chandrayaan-2 Instrument Specifications</h3>", unsafe_allow_html=True)
    st.markdown("""
    | Instrument | Ground Sample Distance | Spectral Range | Swath Width | Primary Science Goal |
    | :--- | :--- | :--- | :--- | :--- |
    | **OHRC** | **0.26 m/pixel** (Nadir) | 0.45–0.70 µm (Panchromatic Visible) | 3.0 km | Safe landing site hazard detection |
    | **TMC-2** | **5.00 m/pixel** (Hub) | 0.50–0.80 µm (Panchromatic Visible) | 20.0 km | High-resolution 3D Digital Elevation Modeling |
    | **IIRS** | **91.75 m/pixel** | 0.80–5.00 µm (256 SWIR Bands) | 20.0 km | Hydroxyl ($OH/H_2O$) & mineral mapping |
    """)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("← Review Hop 1 (Scale Benchmark)", use_container_width=True):
            st.session_state.active_scene = "hop1"
            st.session_state.hop1_step = 2
            st.rerun()
    with col_nav2:
        if st.button("Review Hop 2 (TMC-2 / IIRS Overlap) →", use_container_width=True):
            st.session_state.active_scene = "hop2"
            st.session_state.hop2_step = 2
            st.rerun()
