"""
TriNetra Web Application for SIH26166 Presentation.
Built with Streamlit for a smooth, minimalist UI matching the requested aesthetic.
"""

import streamlit as st
import time
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TriNetra Pipeline Modules
from scripts.generate_mock_data import MockLunarDataGenerator
from src.module1_preprocessing.preprocessor import OHRCPreprocessor, TMC2Preprocessor
from src.module1_preprocessing.iirs_pca import IIRSPreprocessor
from src.module2_matching.hub_matcher import HubAndSpokeMatcher
from src.module3_crater_verification.structural_matcher import StructuralMatcher
from src.module4_registration.registration import GeometricRegistrar
from src.module5_confidence.visualizer import ExplainabilityVisualizer

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TriNetra Pipeline",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR CLAUDE-LIKE MINIMALIST AESTHETIC ---
st.markdown("""
<style>
    /* Clean headers */
    h1, h2, h3, h4 {
        color: #152B4D;
        font-family: "Georgia", "Times New Roman", serif;
        font-weight: 500;
    }
    
    /* Button styling (matches the Peach #DE7356) */
    .stButton > button {
        background-color: #DE7356;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #c96549;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Custom divider */
    hr {
        border-top: 1px solid #E5E2DC;
    }
    
    /* Image styling */
    img {
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Minimalist boxes */
    .css-1r6slb0, .st-c5, .st-c6 {
        border-radius: 8px;
        border: 1px solid #E5E2DC;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_image_normalized(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.float32 or img.dtype == np.float64:
        return np.clip(img, 0, 1)
    return img

def render_image(img, title):
    st.image(get_image_normalized(img), caption=title, use_container_width=True)


# --- STATE INITIALIZATION ---
if "step" not in st.session_state:
    st.session_state.step = 0
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "prep_data" not in st.session_state:
    st.session_state.prep_data = None
if "match_result" not in st.session_state:
    st.session_state.match_result = None
if "reg_result" not in st.session_state:
    st.session_state.reg_result = None


# --- SIDEBAR NAV ---
with st.sidebar:
    st.image("assets/logo.png", width=250)
    st.markdown("### Pipeline Stages")
    
    stages = [
        "1. Data Ingestion",
        "2. Preprocessing",
        "3. Structural Matching",
        "4. Geometric Registration",
    ]
    
    for i, stage in enumerate(stages):
        if i < st.session_state.step:
            st.markdown(f"✅ **{stage}**")
        elif i == st.session_state.step:
            st.markdown(f"🟢 **{stage}**")
        else:
            st.markdown(f"⚪ *{stage}*")
            
    st.markdown("---")
    st.markdown("*ISRO SIH26166 Prototype*")


# --- MAIN CONTENT ---

# Hero Section (only on step 0)
if st.session_state.step == 0:
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0;'>TriNetra</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-style: italic; color: #555;'>Precision. Alignment. Discovery.</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; max-width: 600px; margin: 0 auto 2rem auto;'>Autonomous, scale-invariant image correspondence across OHRC, TMC-2, and IIRS lunar datasets.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Initialize & Generate Lunar Data", use_container_width=True):
            with st.spinner("Generating physically consistent mock lunar surface..."):
                gen = MockLunarDataGenerator(seed=np.random.randint(0, 10000))
                st.session_state.raw_data = gen.generate_all()
                st.session_state.step = 1
                st.rerun()

# Step 1: Data Ingestion View
elif st.session_state.step == 1:
    st.header("Step 1: Raw Data Ingestion")
    st.markdown("Geometrically consistent mock data loaded. Note the extreme scale gaps and radiometric differences.")
    
    raw = st.session_state.raw_data
    c1, c2, c3 = st.columns(3)
    with c1:
        render_image(raw['ohrc']['image'], "OHRC (0.25 m/px)")
    with c2:
        render_image(raw['tmc2']['image'], "TMC-2 (5 m/px)")
    with c3:
        iirs_avg = np.mean(raw['iirs']['cube'], axis=2)
        render_image(iirs_avg, "IIRS Hyperspectral (80 m/px) - Mean Band")
        
    st.markdown("---")
    if st.button("Run Preprocessing Module"):
        with st.spinner("Applying CLAHE, percentile stretching, and PCA..."):
            prep_ohrc = OHRCPreprocessor().preprocess(raw['ohrc']['image'])
            prep_tmc2 = TMC2Preprocessor().preprocess(raw['tmc2']['image'])
            prep_iirs = IIRSPreprocessor().preprocess(raw['iirs']['cube'])
            
            st.session_state.prep_data = {
                'ohrc': prep_ohrc,
                'tmc2': prep_tmc2,
                'iirs': prep_iirs
            }
            st.session_state.step = 2
            st.rerun()

# Step 2: Preprocessing View
elif st.session_state.step == 2:
    st.header("Step 2: Radiometric Preprocessing")
    st.markdown("Shadows lifted and contrast normalized. IIRS cube reduced via PCA.")
    
    prep = st.session_state.prep_data
    c1, c2, c3 = st.columns(3)
    with c1:
        render_image(prep['ohrc'].image, "OHRC (CLAHE Enhanced)")
    with c2:
        render_image(prep['tmc2'].image, "TMC-2 (Percentile Stretch)")
    with c3:
        render_image(prep['iirs'].image, "IIRS (PCA 1st Component)")
        
    st.markdown("---")
    if st.button("Run Structural Hub Matching"):
        with st.spinner("Extracting Hessian ridges and executing Hub-and-Spoke matching..."):
            from src.module2_matching.orb_fallback_matcher import ORBFallbackMatcher
            # Use structural matcher to bypass radiometric gaps
            base_matcher = ORBFallbackMatcher()
            matcher = StructuralMatcher(base_matcher=base_matcher)
            hub = HubAndSpokeMatcher(base_matcher=matcher)
            
            # Match OHRC to IIRS via TMC-2
            res = hub.match(prep['ohrc'], prep['iirs'], prep['tmc2'])
            st.session_state.match_result = res
            st.session_state.step = 3
            st.rerun()

# Step 3: Structural Matching View
elif st.session_state.step == 3:
    st.header("Step 3: Hub-and-Spoke Structural Matching")
    res = st.session_state.match_result
    
    if res.num_matches == 0:
        st.error("Matching failed to find enough keypoints.")
    else:
        st.success(f"Successfully bridged the 320x scale gap! Found {res.num_matches} structurally verified keypoints between OHRC and IIRS (anchored by TMC-2).")
        
        # Plot matches
        match_plot_path = "assets/matches.png"
        prep = st.session_state.prep_data
        ExplainabilityVisualizer.plot_matches(
            prep['ohrc'].image, prep['iirs'].image, res, save_path=match_plot_path
        )
        
        st.image(match_plot_path, use_container_width=True)
        
    st.markdown("---")
    if st.button("Run Geometric Registration"):
        with st.spinner("Computing MAGSAC++ transform..."):
            registrar = GeometricRegistrar(transform_type="affine")
            reg_res = registrar.register(res)
            st.session_state.reg_result = reg_res
            st.session_state.step = 4
            st.rerun()

# Step 4: Geometric Registration View
elif st.session_state.step == 4:
    st.header("Step 4: Geometric Registration (MAGSAC++)")
    reg_res = st.session_state.reg_result
    
    if not reg_res.success:
        st.error("Registration failed. Too many geometric outliers.")
    else:
        st.success(f"MAGSAC++ converged with {reg_res.num_inliers} inliers. RMSE: {reg_res.rmse:.2f}px.")
        
        overlay_plot_path = "assets/overlay.png"
        prep = st.session_state.prep_data
        ExplainabilityVisualizer.plot_registration_overlay(
            prep['ohrc'].image, prep['iirs'].image, reg_res, save_path=overlay_plot_path
        )
        
        st.markdown("### Explainability Diagnostic: Pixel-Perfect Overlay")
        st.markdown("*Red channel = Warped OHRC. Cyan channels = Target IIRS. Where they align perfectly, the image becomes grayscale/white.*")
        st.image(overlay_plot_path, use_container_width=True)
        
    st.markdown("---")
    if st.button("Restart Pipeline"):
        st.session_state.step = 0
        st.rerun()
