import os
import glob
import matplotlib.pyplot as plt
from src.pds_parser import PDS4Parser
from src.data_loader_v2 import MemmapLoader
from src.module1_preprocessing_v2 import preprocess_ohrc, preprocess_iirs_band, preprocess_synthetic_tmc2, synthesize_tmc2

def main():
    base_dir = "/Users/srijeetprasadbanerjee/.gemini/antigravity/scratch/sih26166-lunar-correspondence"
    data_dir = os.path.join(base_dir, "data")
    
    ohrc_xmls = glob.glob(os.path.join(data_dir, "ch2_ohr*", "data", "**", "*.xml"), recursive=True)
    iirs_xmls = glob.glob(os.path.join(data_dir, "ch2_iir*", "data", "**", "*.xml"), recursive=True)
    
    if not ohrc_xmls or not iirs_xmls:
        print("Could not find data files.")
        return
        
    ohrc_xml_path = ohrc_xmls[0]
    iirs_xml_path = iirs_xmls[0]
    
    ohrc_img_path = ohrc_xml_path.replace(".xml", ".img")
    iirs_img_path = iirs_xml_path.replace(".xml", ".qub")
    
    print("Parsing XML labels...")
    ohrc_meta = PDS4Parser(ohrc_xml_path).parse()
    iirs_meta = PDS4Parser(iirs_xml_path).parse()
    
    print("\n=== OHRC Metadata ===")
    for k, v in ohrc_meta.__dict__.items():
        print(f"{k}: {v}")
    
    print("\n=== IIRS Metadata ===")
    for k, v in iirs_meta.__dict__.items():
        print(f"{k}: {v}")
        
    print("\nLoading data...")
    ohrc_loader = MemmapLoader(ohrc_img_path, ohrc_xml_path)
    ohrc_patch = ohrc_loader.extract_patch(size=4000)
    
    iirs_loader = MemmapLoader(iirs_img_path, iirs_xml_path)
    iirs_band = iirs_loader.extract_band(band_index=34) # Band 35 is index 34
    
    print("Synthesizing TMC-2...")
    synth_tmc2 = synthesize_tmc2(ohrc_patch, ohrc_meta.pixel_resolution_m, tmc2_target_gsd=5.0)
    
    print("Preprocessing...")
    ohrc_prep = preprocess_ohrc(ohrc_patch)
    tmc2_prep = preprocess_synthetic_tmc2(ohrc_patch, ohrc_meta.pixel_resolution_m, tmc2_target_gsd=5.0)
    iirs_prep = preprocess_iirs_band(iirs_band)
    
    print("\n=== Preprocessing Metrics ===")
    print(f"OHRC: dynamic_range={ohrc_prep.dynamic_range:.2f}, shadow_fraction={ohrc_prep.shadow_fraction:.4f}")
    print(f"TMC-2: dynamic_range={tmc2_prep.dynamic_range:.2f}, shadow_fraction={tmc2_prep.shadow_fraction:.4f}")
    print(f"IIRS: dynamic_range={iirs_prep.dynamic_range:.2f}, shadow_fraction={iirs_prep.shadow_fraction:.4f}")
    
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Raw
    ax = axes[0, 0]
    ax.imshow(ohrc_patch, cmap='gray')
    ax.set_title(f"Raw OHRC\nGSD: {ohrc_meta.pixel_resolution_m:.2f}m\nSun Elev: {ohrc_meta.sun_elevation_deg:.2f}°")
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(synth_tmc2, cmap='gray')
    ax.set_title(f"Raw Synth TMC-2\nGSD: 5.0m\nSun Elev: {ohrc_meta.sun_elevation_deg:.2f}°")
    ax.axis('off')
    
    ax = axes[0, 2]
    ax.imshow(iirs_band, cmap='gray')
    ax.set_title(f"Raw IIRS Band 35\nGSD: {iirs_meta.pixel_resolution_m:.2f}m\nSun Elev: {iirs_meta.sun_elevation_deg:.2f}°")
    ax.axis('off')
    
    # Row 2: Preprocessed
    ax = axes[1, 0]
    ax.imshow(ohrc_prep.image, cmap='gray')
    ax.set_title("Preprocessed OHRC")
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(tmc2_prep.image, cmap='gray')
    ax.set_title("Preprocessed Synth TMC-2")
    ax.axis('off')
    
    ax = axes[1, 2]
    ax.imshow(iirs_prep.image, cmap='gray')
    ax.set_title("Preprocessed IIRS Band 35")
    ax.axis('off')
    
    plt.tight_layout()
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "phase1_v2_verification.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved verification plot to {out_file}")

if __name__ == "__main__":
    main()
