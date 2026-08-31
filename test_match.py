import numpy as np
from scripts.generate_mock_data import MockLunarDataGenerator
from src.module1_preprocessing.preprocessor import OHRCPreprocessor, TMC2Preprocessor
from src.module1_preprocessing.iirs_pca import IIRSPreprocessor
from src.module2_matching.hub_matcher import HubAndSpokeMatcher
from src.module3_crater_verification.structural_matcher import StructuralMatcher
from src.module2_matching.orb_fallback_matcher import ORBFallbackMatcher

print("Generating mock data...")
gen = MockLunarDataGenerator(seed=42)
raw = gen.generate_all()

print("Preprocessing...")
prep_ohrc = OHRCPreprocessor().preprocess(raw['ohrc']['image'])
prep_tmc2 = TMC2Preprocessor().preprocess(raw['tmc2']['image'])
prep_iirs = IIRSPreprocessor().preprocess(raw['iirs']['cube'])

print("Matching...")
base_matcher = ORBFallbackMatcher(upsample_low_res=4.0)
matcher = StructuralMatcher(base_matcher=base_matcher)
hub = HubAndSpokeMatcher(hop1_matcher=matcher, hop2_matcher=matcher)

res = hub.match(
    src_image=prep_ohrc.image, 
    dst_image=prep_iirs.image, 
    src_instrument="OHRC", 
    dst_instrument="IIRS", 
    tmc2_image=prep_tmc2.image
)

print(f"Final matches: {res.num_matches}")
from src.module4_registration.registration import GeometricRegistrar
registrar = GeometricRegistrar(transform_type="affine")
reg_res = registrar.register(res)
print(f"Registration success: {reg_res.success}, inliers: {reg_res.num_inliers}")
