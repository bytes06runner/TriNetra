#!/usr/bin/env python3
"""
find_bright_pair.py — Identify best-illuminated overlapping TMC-2 / IIRS flight pairs.

Ranks candidate overlapping pairs by lowest combined solar incidence angle
(highest solar elevation / maximum scene radiance) rather than azimuth matching.

SIH26166 — Chandrayaan-2 Multi-modal Lunar Image Correspondence
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

# Spherical geometry helpers (Selenographic Moon sphere R=1737.4 km)
R_MOON = 1737.4  # km

CORNER_KEYS = [
    ("upper_left_latitude", "upper_left_longitude"),
    ("upper_right_latitude", "upper_right_longitude"),
    ("lower_right_latitude", "lower_right_longitude"),
    ("lower_left_latitude", "lower_left_longitude"),
]


def gc_dist_bearing(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    p0, p1 = math.radians(lat0), math.radians(lat)
    dl = math.radians(lon - lon0)
    sin_d = math.sqrt(
        (math.cos(p1) * math.sin(dl)) ** 2
        + (math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl)) ** 2
    )
    cos_d = math.sin(p0) * math.sin(p1) + math.cos(p0) * math.cos(p1) * math.cos(dl)
    d = math.atan2(sin_d, cos_d) * R_MOON
    brg = math.atan2(
        math.sin(dl) * math.cos(p1),
        math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl),
    )
    return d, brg


def project(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    d, brg = gc_dist_bearing(lat0, lon0, lat, lon)
    return (d * math.sin(brg), d * math.cos(brg))


def _orient(p, q, r):
    v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    return 0 if abs(v) < 1e-12 else (1 if v > 0 else 2)


def _on_seg(p, q, r):
    return (
        min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
        and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
    )


def seg_cross(p1, q1, p2, q2):
    o1, o2 = _orient(p1, q1, p2), _orient(p1, q1, q2)
    o3, o4 = _orient(p2, q2, p1), _orient(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_seg(p1, p2, q1):
        return True
    if o2 == 0 and _on_seg(p1, q2, q1):
        return True
    if o3 == 0 and _on_seg(p2, p1, q2):
        return True
    if o4 == 0 and _on_seg(p2, q1, q2):
        return True
    return False


def pt_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def polys_overlap(a, b):
    for i in range(len(a)):
        for j in range(len(b)):
            if seg_cross(a[i], a[(i + 1) % len(a)], b[j], b[(j + 1) % len(b)]):
                return True
    return pt_in_poly(a[0], b) or pt_in_poly(b[0], a)


def centroid(poly):
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def inflate(poly, pad):
    if pad <= 0:
        return poly
    cx, cy = centroid(poly)
    out = []
    for x, y in poly:
        d = math.hypot(x - cx, y - cy)
        s = 1.0 if d < 1e-9 else (d + pad) / d
        out.append((cx + (x - cx) * s, cy + (y - cy) * s))
    return out


def load_catalog(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    rows = list(csv.DictReader(open(path)))
    valid = []
    for r in rows:
        try:
            r["_corners"] = [(float(r[a]), float(r[b])) for a, b in CORNER_KEYS]
            lats = [c[0] for c in r["_corners"]]
            r["_lat_c"] = sum(lats) / 4.0
            r["_inc"] = float(r["solar_incidence"]) if r.get("solar_incidence") else None
            r["_az"] = float(r["sun_azimuth"]) if r.get("sun_azimuth") else None
            r["_mb"] = float(r["file_size_mb"]) if r.get("file_size_mb") else None
            valid.append(r)
        except (KeyError, TypeError, ValueError):
            continue
    return valid


def az_diff(az1: float | None, az2: float | None) -> float | None:
    if az1 is None or az2 is None:
        return None
    d = abs(az1 - az2) % 360.0
    return min(d, 360.0 - d)


def find_bright_pairs(
    tmc_csv: Path,
    iirs_csv: Path,
    pad_km: float = 15.0,
    max_gb: float = 6.0,
    top_n: int = 10,
) -> list[dict]:
    tmc_list = load_catalog(tmc_csv)
    iirs_list = load_catalog(iirs_csv)

    max_mb = max_gb * 1024.0 if max_gb > 0 else float("inf")

    tmc_cand = [r for r in tmc_list if (r["_mb"] or 0) <= max_mb and r["_inc"] is not None]
    iir_cand = [r for r in iirs_list if (r["_mb"] or 0) <= max_mb and r["_inc"] is not None]

    matches = []
    for t in tmc_cand:
        lat0, lon0 = t["_corners"][0]
        t_poly = [project(lat0, lon0, x, y) for x, y in t["_corners"]]
        t_poly_inf = inflate(t_poly, pad_km)

        for i in iir_cand:
            if abs(i["_lat_c"] - t["_lat_c"]) > 25.0:
                continue

            i_poly = [project(lat0, lon0, x, y) for x, y in i["_corners"]]
            if polys_overlap(t_poly_inf, i_poly):
                mean_inc = (t["_inc"] + i["_inc"]) / 2.0
                max_inc = max(t["_inc"], i["_inc"])
                d_az = az_diff(t["_az"], i["_az"])
                matches.append({
                    "tmc": t,
                    "iirs": i,
                    "mean_inc": mean_inc,
                    "max_inc": max_inc,
                    "tmc_inc": t["_inc"],
                    "iir_inc": i["_inc"],
                    "d_az": d_az,
                })

    # Sort primarily by lowest combined solar incidence (brightest solar elevation)
    matches.sort(key=lambda x: x["mean_inc"])
    return matches[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Find best-illuminated overlapping TMC-2/IIRS pairs.")
    parser.add_argument(
        "--harvest-dir",
        type=Path,
        default=Path.home() / "Desktop/harvest",
        help="Directory containing tmc.csv and iirs.csv",
    )
    parser.add_argument("--pad", type=float, default=15.0, help="Footprint inflation in km")
    parser.add_argument("--max-gb", type=float, default=6.0, help="File size limit in GB per product")
    parser.add_argument("--top", type=int, default=10, help="Number of top candidates to report")
    args = parser.parse_args()

    tmc_p = args.harvest_dir / "tmc.csv"
    iir_p = args.harvest_dir / "iirs.csv"

    print("==========================================================================================")
    print("TOP CANDIDATE OVERLAPPING TMC-2 / IIRS PAIRS (RANKED BY LOWEST SOLAR INCIDENCE / BRIGHTEST)")
    print("==========================================================================================")

    best_pairs = find_bright_pairs(tmc_p, iir_p, pad_km=args.pad, max_gb=args.max_gb, top_n=args.top)

    if not best_pairs:
        print("No overlapping pairs found within specified constraints.")
        return

    print(f"Found candidate pairs. Top {len(best_pairs)} best-illuminated pairs:\n")

    for rank, p in enumerate(best_pairs, 1):
        t = p["tmc"]
        i = p["iirs"]
        d_az_str = f"{p['d_az']:.1f}°" if p["d_az"] is not None else "N/A"
        area = t.get("area") or i.get("area") or "Polar"
        rad_gain = (math.cos(math.radians(p["mean_inc"])) / math.cos(math.radians(76.93)))

        print(f"[{rank:2d}] Mean Solar Incidence: {p['mean_inc']:.2f}° (Radiance Gain: {rad_gain:.2f}× vs North Polar 76.9°)")
        print(f"     TMC-2: {t['product_id']}")
        print(f"            Incidence: {p['tmc_inc']:.2f}°, GSD: {t.get('pixel_resolution', '5.0')} m/px, Size: {p['tmc']['_mb']:.1f} MB, Area: {t.get('area')}")
        print(f"     IIRS:  {i['product_id']}")
        print(f"            Incidence: {p['iir_inc']:.2f}°, GSD: {i.get('pixel_resolution', '80.0')} m/px, Size: {p['iirs']['_mb']:.1f} MB, Area: {i.get('area')}")
        print(f"     Delta Azimuth: {d_az_str}, Geographic Zone: {area}")
        print("     " + "-" * 85)


if __name__ == "__main__":
    main()
