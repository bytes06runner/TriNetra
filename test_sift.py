import numpy as np
import cv2
from scripts.generate_mock_data import MockLunarDataGenerator
from src.module1_preprocessing.preprocessor import OHRCPreprocessor, TMC2Preprocessor
from src.module1_preprocessing.iirs_pca import IIRSPreprocessor
from src.module2_matching.scale_handler import ScaleAligner

gen = MockLunarDataGenerator(seed=42)
raw = gen.generate_all()

prep_ohrc = OHRCPreprocessor().preprocess(raw['ohrc']['image']).image
prep_tmc2 = TMC2Preprocessor().preprocess(raw['tmc2']['image']).image

print(f"OHRC: {prep_ohrc.shape}, TMC2: {prep_tmc2.shape}")

# Scale alignment
scale_ratio = 0.05
aligner = ScaleAligner(upsample_factor=1.0)
aligned = aligner.align(prep_ohrc, prep_tmc2, scale_ratio)
aligned_src = aligned["aligned_src"]
aligned_dst = aligned["aligned_dst"]

# Convert to uint8
def to_u8(img): return (np.clip(img, 0, 1) * 255).astype(np.uint8)
src_u8 = to_u8(aligned_src)
dst_u8 = to_u8(aligned_dst)

sift = cv2.SIFT_create()
kp_src, des_src = sift.detectAndCompute(src_u8, None)
kp_dst, des_dst = sift.detectAndCompute(dst_u8, None)

print(f"SIFT KPs - Src: {len(kp_src)}, Dst: {len(kp_dst)}")

if des_src is not None and des_dst is not None:
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des_src, des_dst, k=2)
    
    good = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(f"SIFT good matches: {len(good)}")
