"""
Registered product and tie-point writers. All writers refuse to overwrite.
"""

import json
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np


def _fresh(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def inverse_map(model, shape, iters: int = 8):
    """Source-grid coordinates q with model(q) = p for every reference-grid pixel p,
    by fixed-point iteration q <- p - d(q). Returns (qx, qy) float32 maps and the
    maximum final update (convergence check)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    p = np.column_stack([xx.ravel(), yy.ravel()])
    q = p.copy()
    step = np.inf
    for _ in range(iters):
        d = model.predict(q) - q
        q_new = p - d
        step = float(np.abs(q_new - q).max())
        q = q_new
    return q[:, 0].reshape(h, w).astype(np.float32), q[:, 1].reshape(h, w).astype(np.float32), step


def write_registered_geotiff(path, src_grid: np.ndarray, model, crs_wkt: str,
                             x0: float, y0: float, gsd: float, tags: Dict) -> Dict:
    import rasterio
    from rasterio.transform import Affine

    path = _fresh(path)
    qx, qy, step = inverse_map(model, src_grid.shape)
    reg = cv2.remap(src_grid.astype(np.float32), qx, qy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # grid pixel i is the map point x0 + i*gsd, i.e. the pixel centre
    transform = Affine(gsd, 0.0, x0 - gsd / 2.0, 0.0, -gsd, y0 + gsd / 2.0)
    with rasterio.open(path, "w", driver="GTiff", height=reg.shape[0], width=reg.shape[1], count=1,
                       dtype="float32", crs=crs_wkt, transform=transform, nodata=0.0,
                       compress="deflate", tiled=True) as dst:
        dst.write(reg, 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return {"path": str(path), "inverse_map_final_step_px": step, "registered": reg}


def write_csv(path, header_lines: Sequence[str], columns: Dict[str, np.ndarray], fmt: Dict[str, str]):
    path = _fresh(path)
    names = list(columns)
    n = len(columns[names[0]])
    with open(path, "x", encoding="utf-8") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write(",".join(names) + "\n")
        for i in range(n):
            f.write(",".join(format(columns[k][i], fmt.get(k, "")) for k in names) + "\n")
    return str(path)


def write_geojson(path, lon, lat, props: Dict[str, np.ndarray], comment: str):
    path = _fresh(path)
    feats = []
    for i in range(len(lon)):
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [round(float(lon[i]), 8), round(float(lat[i]), 8)]},
                      "properties": {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in props.items()}})
    with open(path, "x", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "comment": comment, "features": feats}, f)
    return str(path)
