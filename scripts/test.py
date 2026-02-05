#!/usr/bin/env python3
"""
test.py - Footprint ↔ LAS Alignment Diagnostic

Checks whether missing facade sides are due to:
- CRS mismatch
- Spatial offset
- Clipping issues
- One side having no nearby LiDAR points

Prints per-edge distance diagnostics.
"""

import numpy as np
import geopandas as gpd
import laspy
from shapely.geometry import Point, LineString
from pathlib import Path

# =========================
# INPUTS (HARDCODED)
# =========================

LAS_PATH = Path(r"C:\Projects\Omni2LOD3\outputs\02_clipped\NIMBB_112025_01_clipped.las")
SHP_PATH = Path(r"C:\Projects\Omni2LOD3\data\02_footprint\NIMBB_footprint_1.shp")

TARGET_EPSG = 32651

EDGE_SAMPLE_DIST = 1.0     # meters
NEAR_RADIUS = 2.0          # meters for diagnostics
BACKUP_RADIUS = 5.0        # fallback

# =========================
# HELPERS
# =========================

def sample_edge(linestring, dist):
    pts = []
    length = linestring.length
    n = max(2, int(np.ceil(length / dist)))
    for i in range(n):
        frac = i / max(1, n - 1)
        pts.append(linestring.interpolate(frac, normalized=True))
    return pts

def main():
    print("=" * 80)
    print("FOOTPRINT ↔ LAS ALIGNMENT DIAGNOSTIC")
    print("=" * 80)

    # -------------------------
    # Load LAS
    # -------------------------
    print("\n-> Reading LAS...")
    las = laspy.read(LAS_PATH)
    X = np.asarray(las.x)
    Y = np.asarray(las.y)
    Z = np.asarray(las.z)

    print(f"   LAS points: {len(X):,}")
    print(f"   LAS X range: {X.min():.2f} .. {X.max():.2f}")
    print(f"   LAS Y range: {Y.min():.2f} .. {Y.max():.2f}")
    print(f"   LAS Z range: {Z.min():.2f} .. {Z.max():.2f}")

    # -------------------------
    # Load footprint
    # -------------------------
    print("\n-> Reading footprint...")
    gdf = gpd.read_file(SHP_PATH)
    print(f"   Footprint CRS: {gdf.crs}")

    if gdf.crs is None or gdf.crs.to_epsg() != TARGET_EPSG:
        print(f"   Reprojecting footprint to EPSG:{TARGET_EPSG}")
        gdf = gdf.to_crs(epsg=TARGET_EPSG)

    footprints = gdf.geometry.values

    # Overall footprint bounds
    fp_bounds = gdf.total_bounds
    print(f"\n   Footprint bounds:")
    print(f"      X: {fp_bounds[0]:.2f} .. {fp_bounds[2]:.2f}")
    print(f"      Y: {fp_bounds[1]:.2f} .. {fp_bounds[3]:.2f}")

    # -------------------------
    # Global overlap sanity check
    # -------------------------
    print("\n-> Global overlap sanity check")

    dx_min = fp_bounds[0] - X.max()
    dx_max = X.min() - fp_bounds[2]
    dy_min = fp_bounds[1] - Y.max()
    dy_max = Y.min() - fp_bounds[3]

    if dx_min > 0 or dx_max > 0 or dy_min > 0 or dy_max > 0:
        print("❌ WARNING: Footprint and LAS DO NOT OVERLAP in XY!")
        print("   This WILL cause missing facade sides.")
    else:
        print("✓ Footprint and LAS bounding boxes overlap.")

    # -------------------------
    # Per-edge diagnostics
    # -------------------------
    print("\n-> Per-edge distance diagnostics")

    for fp_idx, poly in enumerate(footprints):
        if poly is None or poly.is_empty:
            continue

        print(f"\nFOOTPRINT {fp_idx + 1}")

        boundary = poly.exterior
        coords = list(boundary.coords)

        for i in range(len(coords) - 1):
            p0 = coords[i]
            p1 = coords[i + 1]
            edge = LineString([p0, p1])

            edge_pts = sample_edge(edge, EDGE_SAMPLE_DIST)

            min_dists = []
            hit_counts = 0

            for pt in edge_pts:
                x0, y0 = pt.x, pt.y

                dx = X - x0
                dy = Y - y0
                dist = np.sqrt(dx * dx + dy * dy)

                dmin = dist.min()
                min_dists.append(dmin)

                if dmin <= NEAR_RADIUS:
                    hit_counts += 1

            min_dists = np.array(min_dists)

            print(f"\n  EDGE {i + 1}")
            print(f"    From: ({p0[0]:.2f}, {p0[1]:.2f})")
            print(f"    To:   ({p1[0]:.2f}, {p1[1]:.2f})")
            print(f"    Samples: {len(edge_pts)}")
            print(f"    Min dist to LAS (min/med/max): "
                  f"{min_dists.min():.2f} / "
                  f"{np.median(min_dists):.2f} / "
                  f"{min_dists.max():.2f}")
            print(f"    Points within {NEAR_RADIUS} m: {hit_counts}/{len(edge_pts)}")

            if hit_counts == 0:
                print("    ❌ NO nearby LAS points on this edge!")
                print("       -> This edge WILL have missing facade columns.")
                print("       -> Likely causes:")
                print("          - Footprint offset")
                print("          - LAS clipping too tight")
                print("          - Wrong CRS or shift")
            elif min_dists.min() > BACKUP_RADIUS:
                print("    ⚠ Very far from LAS even with backup radius!")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
