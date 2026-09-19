"""Chandrayaan-2 to lunar reference registration (Stage O onward).

Modules:
    pitiscus_grid  build the TMC-2 / LROC NAC common map grid at Pitiscus
    dense_ncc      dense normalised cross-correlation node matching
    models         geometric models and seeded RANSAC
    validate       spatial-block and random k-fold held-out validation
    products       registered GeoTIFF, tie-point CSV / GeoJSON writers
"""
