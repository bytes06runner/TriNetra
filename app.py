"""
TriNetra — Professional Web Dashboard for SIH26166 Presentation.

A minimalist, Anthropic-inspired interactive pipeline demo with smooth
micro-animations, responsive layout, and real pipeline execution.

Author: Srijeet Prasad Banerjee
"""

import streamlit as st
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path

# TriNetra Pipeline Modules
from scripts.generate_mock_data import MockLunarDataGenerator
from src.module1_preprocessing.preprocessor import OHRCPreprocessor, TMC2Preprocessor
from src.module1_preprocessing.iirs_pca import IIRSPreprocessor
from src.module2_matching.hub_matcher import HubAndSpokeMatcher
from src.module3_crater_verification.structural_matcher import StructuralMatcher
from src.module4_registration.registration import GeometricRegistrar
from src.module5_confidence.visualizer import ExplainabilityVisualizer


# ─── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="TriNetra — SIH26166",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── CSS: Full Professional Redesign ─────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
        /* ═══════════════════════════════════════════════════════════
           FONTS — Import Inter (closest open-source to Claude Sans)
           ═══════════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,wght@0,400;0,600;1,400;1,600&display=swap');

        /* ═══════════════════════════════════════════════════════════
           GLOBAL RESET
           ═══════════════════════════════════════════════════════════ */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #F9F8F6;
            color: #2D2D2D;
        }

        /* Remove default Streamlit padding bloat */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        /* ═══════════════════════════════════════════════════════════
           TYPOGRAPHY
           ═══════════════════════════════════════════════════════════ */
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
            font-size: 1.5rem !important;
            letter-spacing: -0.01em;
        }
        h3 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            color: #444 !important;
            font-size: 1.15rem !important;
        }
        p, li, span, div {
            font-family: 'Inter', sans-serif;
            line-height: 1.65;
        }

        /* ═══════════════════════════════════════════════════════════
           SIDEBAR
           ═══════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E8E5DF;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        /* ═══════════════════════════════════════════════════════════
           BUTTONS — Peach accent with micro-animation
           ═══════════════════════════════════════════════════════════ */
        .stButton > button {
            background-color: #DE7356 !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.7rem 2.2rem !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 1px 3px rgba(222, 115, 86, 0.3) !important;
        }
        .stButton > button:hover {
            background-color: #C9604A !important;
            color: white !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(222, 115, 86, 0.35) !important;
        }
        .stButton > button:active {
            transform: translateY(0px) scale(0.98) !important;
            box-shadow: 0 1px 4px rgba(222, 115, 86, 0.25) !important;
        }

        /* Secondary button style */
        .secondary-btn .stButton > button {
            background-color: transparent !important;
            color: #DE7356 !important;
            border: 1.5px solid #DE7356 !important;
            box-shadow: none !important;
        }
        .secondary-btn .stButton > button:hover {
            background-color: rgba(222, 115, 86, 0.08) !important;
            color: #C9604A !important;
            box-shadow: none !important;
        }

        /* ═══════════════════════════════════════════════════════════
           CARDS
           ═══════════════════════════════════════════════════════════ */
        .metric-card {
            background: white;
            border: 1px solid #E8E5DF;
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .metric-card:hover {
            border-color: #DE7356;
            box-shadow: 0 4px 16px rgba(222, 115, 86, 0.1);
            transform: translateY(-2px);
        }
        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            font-family: 'Newsreader', serif;
            font-size: 1.8rem;
            font-weight: 600;
            color: #1a1a2e;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #999;
            margin-top: 0.2rem;
        }

        /* ═══════════════════════════════════════════════════════════
           STAGE PILLS (sidebar)
           ═══════════════════════════════════════════════════════════ */
        .stage-pill {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.55rem 0.8rem;
            border-radius: 10px;
            margin-bottom: 0.35rem;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.25s ease;
        }
        .stage-done {
            background: rgba(76, 175, 80, 0.08);
            color: #2E7D32;
        }
        .stage-active {
            background: rgba(222, 115, 86, 0.12);
            color: #DE7356;
            font-weight: 600;
            border-left: 3px solid #DE7356;
        }
        .stage-pending {
            color: #BBB;
        }

        /* ═══════════════════════════════════════════════════════════
           SUCCESS / ERROR ALERTS
           ═══════════════════════════════════════════════════════════ */
        .stSuccess {
            border-radius: 12px !important;
            animation: slideIn 0.4s ease-out;
        }
        .stAlert {
            border-radius: 12px !important;
            animation: slideIn 0.4s ease-out;
        }

        /* ═══════════════════════════════════════════════════════════
           IMAGES
           ═══════════════════════════════════════════════════════════ */
        [data-testid="stImage"] img {
            border-radius: 12px;
            border: 1px solid #E8E5DF;
            transition: all 0.3s ease;
        }
        [data-testid="stImage"] img:hover {
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            transform: scale(1.01);
        }

        /* ═══════════════════════════════════════════════════════════
           SPINNER
           ═══════════════════════════════════════════════════════════ */
        .stSpinner > div {
            border-color: #DE7356 !important;
        }

        /* ═══════════════════════════════════════════════════════════
           DIVIDERS
           ═══════════════════════════════════════════════════════════ */
        hr {
            border: none;
            border-top: 1px solid #E8E5DF;
            margin: 2rem 0;
        }

        /* ═══════════════════════════════════════════════════════════
           ANIMATIONS
           ═══════════════════════════════════════════════════════════ */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-10px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .animate-in {
            animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .animate-in-delay {
            animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.15s both;
        }
        .animate-in-delay2 {
            animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.3s both;
        }
        .animate-fade {
            animation: fadeIn 0.8s ease both;
        }

        /* Hide Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ─── Helper Functions ─────────────────────────────────────────────────
def render_image(img: np.ndarray, title: str):
    """Render a normalized grayscale image with title."""
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img, 0, 1)
    st.image(img, caption=title, use_container_width=True)


def metric_card(label: str, value: str, sub: str = "") -> str:
    """Return HTML for a styled metric card."""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """


def stage_pill(label: str, status: str) -> str:
    """Return HTML for a sidebar stage pill."""
    icons = {"done": "✓", "active": "●", "pending": "○"}
    css_class = f"stage-{status}"
    icon = icons.get(status, "○")
    return f'<div class="stage-pill {css_class}">{icon}&nbsp;&nbsp;{label}</div>'


def animated_section(html: str, delay: str = ""):
    """Wrap content in an animated container."""
    cls = "animate-in" if not delay else delay
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────
defaults = {
    "step": 0,
    "raw_data": None,
    "prep_data": None,
    "match_result": None,
    "reg_result": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

current_step = st.session_state.step


# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    # Logos
    st.markdown('<div class="animate-in" style="text-align:center; padding: 0.5rem 0 1rem 0;">', unsafe_allow_html=True)
    st.image("assets/logo.png", width=180)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline stages
    st.markdown('<p style="font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#999; margin-bottom:0.8rem;">Pipeline Stages</p>', unsafe_allow_html=True)

    stages = [
        ("Data Ingestion", 1),
        ("Preprocessing", 2),
        ("Structural Matching", 3),
        ("Geometric Registration", 4),
        ("Visualisation", 5),
    ]

    for label, step_num in stages:
        if current_step >= step_num:
            st.markdown(stage_pill(label, "done"), unsafe_allow_html=True)
        elif current_step == step_num - 1:
            st.markdown(stage_pill(label, "active"), unsafe_allow_html=True)
        else:
            st.markdown(stage_pill(label, "pending"), unsafe_allow_html=True)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style="text-align:center; padding-top: 0.5rem;">
        <p style="font-size:0.7rem; color:#aaa; letter-spacing:0.05em;">
            ISRO · SIH26166<br/>
            Chandrayaan-2 Optical Correspondence
        </p>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0 — HERO LANDING PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if current_step == 0:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 1rem 0;">
        <div class="animate-in">
            <h1 style="font-size: 3.2rem; margin-bottom: 0.2rem; font-style: italic;">TriNetra</h1>
        </div>
        <div class="animate-in-delay">
            <p style="font-family: 'Newsreader', serif; font-style: italic; font-size: 1.15rem; color: #777; margin-bottom: 2rem;">
                Precision · Alignment · Discovery
            </p>
        </div>
        <div class="animate-in-delay2">
            <p style="max-width: 580px; margin: 0 auto 1rem auto; font-size: 1rem; color: #555; line-height: 1.7;">
                Autonomous, scale-invariant image correspondence across
                <strong>OHRC</strong>, <strong>TMC-2</strong>, and <strong>IIRS</strong>
                lunar instruments aboard Chandrayaan-2.
            </p>
            <p style="max-width: 520px; margin: 0 auto 0.5rem auto; font-size: 0.85rem; color: #999;">
                Bridging a 320× resolution gap across visible and infrared modalities,<br/>
                robust to extreme polar shadow conditions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Instrument spec cards
    st.markdown('<div class="animate-in-delay2">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("OHRC", "0.26 m/px", "Panchromatic · 93K × 12K"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("TMC-2", "5.97 m/px", "Panchromatic · 190K × 4K"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("IIRS", "68.38 m/px", "256 Bands · Hyperspectral IR"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # CTA Button
    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        if st.button("Initialize Pipeline →", use_container_width=True):
            with st.spinner("Generating physically consistent lunar terrain…"):
                gen = MockLunarDataGenerator(seed=42)
                st.session_state.raw_data = gen.generate_all()
                st.session_state.step = 1
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — DATA INGESTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif current_step == 1:
    st.markdown("""
    <div class="animate-in">
        <h1>Data Ingestion</h1>
        <p style="color:#666; max-width:700px;">
            Raw lunar surface data loaded. Three instruments, three vastly different scales
            and modalities — all imaging the same terrain.
        </p>
    </div>
    """, unsafe_allow_html=True)

    raw = st.session_state.raw_data

    # Instrument images
    st.markdown('<div class="animate-in-delay">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_image(raw["ohrc"]["image"], "OHRC — 0.25 m/px  ·  Visible")
        st.markdown(metric_card("Scale", "1×", "Highest resolution"), unsafe_allow_html=True)
    with c2:
        render_image(raw["tmc2"]["image"], "TMC-2 — 5 m/px  ·  Visible")
        st.markdown(metric_card("Scale", "20×", "Hub instrument"), unsafe_allow_html=True)
    with c3:
        iirs_avg = np.mean(raw["iirs"]["cube"], axis=2)
        render_image(iirs_avg, "IIRS — 80 m/px  ·  Mean Hyperspectral Band")
        st.markdown(metric_card("Scale", "320×", "256 spectral bands"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("Run Preprocessing →", use_container_width=True):
            with st.spinner("Applying shadow-aware CLAHE, percentile stretching, and PCA…"):
                prep_ohrc = OHRCPreprocessor().preprocess(raw["ohrc"]["image"])
                prep_tmc2 = TMC2Preprocessor().preprocess(raw["tmc2"]["image"])
                prep_iirs = IIRSPreprocessor().preprocess(raw["iirs"]["cube"])
                st.session_state.prep_data = {
                    "ohrc": prep_ohrc,
                    "tmc2": prep_tmc2,
                    "iirs": prep_iirs,
                }
                st.session_state.step = 2
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — PREPROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif current_step == 2:
    st.markdown("""
    <div class="animate-in">
        <h1>Radiometric Preprocessing</h1>
        <p style="color:#666; max-width:700px;">
            Extreme lunar shadows lifted via adaptive CLAHE. The IIRS 256-band
            hyperspectral cube has been collapsed to a 2D proxy via PCA.
        </p>
    </div>
    """, unsafe_allow_html=True)

    prep = st.session_state.prep_data

    st.markdown('<div class="animate-in-delay">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        render_image(prep["ohrc"].image, "OHRC — Shadow-Aware CLAHE")
    with c2:
        render_image(prep["tmc2"].image, "TMC-2 — Percentile Stretch")
    with c3:
        render_image(prep["iirs"].image, "IIRS — PCA 1st Component")
    st.markdown('</div>', unsafe_allow_html=True)

    # Quality metrics
    st.markdown('<div class="animate-in-delay2">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card(
            "OHRC Confidence",
            f"{prep['ohrc'].confidence:.0%}",
            f"Shadow: {prep['ohrc'].quality_metrics.get('shadow_fraction', 0):.0%}"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(
            "TMC-2 Confidence",
            f"{prep['tmc2'].confidence:.0%}",
            f"Dynamic Range: {prep['tmc2'].quality_metrics.get('dynamic_range', 0):.2f}"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(
            "IIRS Confidence",
            f"{prep['iirs'].confidence:.0%}",
            f"Entropy: {prep['iirs'].quality_metrics.get('entropy', 0):.2f}"
        ), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("Run Structural Matching →", use_container_width=True):
            with st.spinner("Extracting Hessian ridges and executing Hub-and-Spoke matching…"):
                from src.module2_matching.orb_fallback_matcher import ORBFallbackMatcher
                base_matcher = ORBFallbackMatcher(upsample_low_res=4.0)
                matcher = StructuralMatcher(base_matcher=base_matcher)
                hub = HubAndSpokeMatcher(hop1_matcher=matcher, hop2_matcher=matcher)

                res = hub.match(
                    src_image=prep["ohrc"].image,
                    dst_image=prep["iirs"].image,
                    src_instrument="OHRC",
                    dst_instrument="IIRS",
                    tmc2_image=prep["tmc2"].image,
                )
                st.session_state.match_result = res
                st.session_state.step = 3
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — STRUCTURAL MATCHING RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif current_step == 3:
    res = st.session_state.match_result

    st.markdown("""
    <div class="animate-in">
        <h1>Hub-and-Spoke Structural Matching</h1>
        <p style="color:#666; max-width:700px;">
            OHRC ↔ IIRS correspondence via TMC-2 hub.
            Illumination-invariant Hessian ridges ensure shadow-robust matching.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if res.num_matches == 0:
        st.error("Matching returned 0 keypoints. Please restart the pipeline.")
    else:
        # Stats
        st.markdown('<div class="animate-in-delay">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card("Keypoints Found", str(res.num_matches), "Structurally verified"), unsafe_allow_html=True)
        with c2:
            avg_conf = float(np.mean(res.confidences)) if len(res.confidences) > 0 else 0
            st.markdown(metric_card("Avg Confidence", f"{avg_conf:.0%}", "Per-match descriptor score"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Scale Gap Bridged", "320×", "OHRC 0.25m → IIRS 80m"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Match visualisation
        prep = st.session_state.prep_data
        match_path = "assets/matches.png"
        ExplainabilityVisualizer.plot_matches(
            prep["ohrc"].image, prep["iirs"].image, res, save_path=match_path
        )
        st.markdown('<div class="animate-in-delay2">', unsafe_allow_html=True)
        st.image(match_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        if st.button("Run Geometric Registration →", use_container_width=True):
            with st.spinner("Computing MAGSAC++ homography…"):
                registrar = GeometricRegistrar(transform_type="homography")
                reg_res = registrar.register(res)
                st.session_state.reg_result = reg_res
                st.session_state.step = 4
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4 — GEOMETRIC REGISTRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif current_step == 4:
    reg_res = st.session_state.reg_result

    st.markdown("""
    <div class="animate-in">
        <h1>Geometric Registration</h1>
        <p style="color:#666; max-width:700px;">
            MAGSAC++ robust estimation with outlier rejection.
            The computed homography maps OHRC pixel coordinates into the IIRS frame.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not reg_res.success:
        st.error("Registration failed — too many geometric outliers.")
    else:
        # Stats
        st.markdown('<div class="animate-in-delay">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card("Inliers", str(reg_res.num_inliers), "Geometrically consistent"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("RMSE", f"{reg_res.rmse:.2f} px", "Reprojection error"), unsafe_allow_html=True)
        with c3:
            inlier_ratio = reg_res.num_inliers / max(1, st.session_state.match_result.num_matches)
            st.markdown(metric_card("Inlier Ratio", f"{inlier_ratio:.0%}", "Matches passing MAGSAC++"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Overlay visualisation
        prep = st.session_state.prep_data
        overlay_path = "assets/overlay.png"
        ExplainabilityVisualizer.plot_registration_overlay(
            prep["ohrc"].image, prep["iirs"].image, reg_res, save_path=overlay_path
        )
        st.markdown('<div class="animate-in-delay2">', unsafe_allow_html=True)
        st.markdown("### Pixel-Perfect Overlay")
        st.markdown("""
        <p style="color:#888; font-size:0.9rem;">
            Red channel = Warped OHRC&nbsp;&nbsp;·&nbsp;&nbsp;Cyan = Target IIRS&nbsp;&nbsp;·&nbsp;&nbsp;
            Grayscale/white = perfect alignment
        </p>
        """, unsafe_allow_html=True)
        st.image(overlay_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Restart
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("↻  Restart Pipeline", use_container_width=True):
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
