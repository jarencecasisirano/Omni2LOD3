"""
Debug: why does 10_openings_json.py fail to place windows for nimbb_021726_merged.json?
Focus: windows_flat-2.las  (doors work fine)
"""
import json, os, sys, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LOD2toLOD3"))

try:
    import laspy
    from sklearn.cluster import DBSCAN
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("laspy/sklearn not found in current env — activate las-env")
    sys.exit(1)

VERTICAL_TOL    = 0.3
EPS             = 3.0
MIN_SAMPLES     = 30
DIST_TOL        = 2.0
Z_EXPAND        = 0.5
NIMBB_JSON      = "outputs/00_json_wall_merged/nimbb_021726_merged.json"
WINDOWS_LAS     = "outputs/11B_flat/windows_flat-2.las"

# ── helpers ────────────────────────────────────────────────────────────────
def decode_vertices(cm):
    raw  = np.array(cm["vertices"], dtype=np.float64)
    t    = cm.get("transform", {})
    s    = np.array(t.get("scale",     [1,1,1]), dtype=np.float64)
    tr   = np.array(t.get("translate", [0,0,0]), dtype=np.float64)
    return raw * s + tr

def newell_normal(coords):
    n = np.zeros(3)
    for i in range(len(coords)):
        c  = coords[i]
        nx = coords[(i+1) % len(coords)]
        n[0] += (c[1]-nx[1])*(c[2]+nx[2])
        n[1] += (c[2]-nx[2])*(c[0]+nx[0])
        n[2] += (c[0]-nx[0])*(c[1]+nx[1])
    return n

# ── 1. Load model & collect vertical surfaces ──────────────────────────────
print("=" * 70)
print("1. NIMBB VERTICAL SURFACES")
print("=" * 70)
with open(NIMBB_JSON) as f:
    cm = json.load(f)

wv   = decode_vertices(cm)
obj  = cm["CityObjects"]["1-0"]
geom = obj["geometry"][2]          # lod 2.2 Solid
shell = geom["boundaries"][0]

vert_surfs = []
for p_idx, polygon in enumerate(shell):
    ext_ring = polygon[0]
    coords   = wv[np.array(ext_ring)]
    if len(coords) < 3:
        continue
    n     = newell_normal(coords)
    n_len = np.linalg.norm(n)
    if n_len < 1e-9:
        continue
    n_unit = n / n_len
    if abs(n_unit[2]) > VERTICAL_TOL:
        continue
    nh  = n[:2]
    mag = np.linalg.norm(nh)
    if mag < 1e-6:
        continue
    n2d = nh / mag
    centroid = coords.mean(axis=0)
    vert_surfs.append({
        "idx":       len(vert_surfs),
        "poly_idx":  p_idx,
        "normal_2d": n2d,
        "origin_2d": centroid[:2],
        "z_min":     float(coords[:,2].min()),
        "z_max":     float(coords[:,2].max()),
        "xy_min":    coords[:,:2].min(axis=0),
        "xy_max":    coords[:,:2].max(axis=0),
        "coords":    coords,
        "angle":     float(np.degrees(np.arctan2(n2d[1], n2d[0]))),
        "npts":      len(coords),
    })

bldg_z_min = wv[:,2].min()
bldg_z_max = wv[:,2].max()
bldg_xy_min = wv[:,:2].min(axis=0)
bldg_xy_max = wv[:,:2].max(axis=0)

print(f"  Vertical surfaces: {len(vert_surfs)}")
print(f"  Building XY: [{bldg_xy_min[0]:.1f},{bldg_xy_min[1]:.1f}] – [{bldg_xy_max[0]:.1f},{bldg_xy_max[1]:.1f}]")
print(f"  Building Z:  {bldg_z_min:.3f} – {bldg_z_max:.3f}")

# ── 2. Load windows LAS ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. WINDOWS POINT CLOUD")
print("=" * 70)
las  = laspy.read(WINDOWS_LAS)
pts  = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
print(f"  Total points: {len(pts):,}")
print(f"  X: {pts[:,0].min():.3f} – {pts[:,0].max():.3f}")
print(f"  Y: {pts[:,1].min():.3f} – {pts[:,1].max():.3f}")
print(f"  Z: {pts[:,2].min():.3f} – {pts[:,2].max():.3f}")

# How many points are spatially near the nimbb building?
pad = 100
mask_near = ((pts[:,0] >= bldg_xy_min[0]-pad) & (pts[:,0] <= bldg_xy_max[0]+pad) &
             (pts[:,1] >= bldg_xy_min[1]-pad) & (pts[:,1] <= bldg_xy_max[1]+pad))
pts_near  = pts[mask_near]
print(f"  Points within {pad}m of nimbb building: {pts_near.shape[0]:,}")
if pts_near.shape[0] > 0:
    print(f"    Z range of nearby pts: {pts_near[:,2].min():.3f} – {pts_near[:,2].max():.3f}")

# ── 3. DBSCAN ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. DBSCAN CLUSTERS")
print("=" * 70)
labels  = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(pts)
ulabels = sorted(set(labels) - {-1})
n_noise = int((labels == -1).sum())
print(f"  Clusters: {len(ulabels)}   noise: {n_noise:,}")

clusters = []
for cid in ulabels:
    mask = labels == cid
    cpts = pts[mask]
    centroid = cpts.mean(axis=0)
    clusters.append({
        "id":       cid,
        "n_pts":    int(mask.sum()),
        "pts":      cpts,
        "centroid": centroid,
        "z_min":    float(cpts[:,2].min()),
        "z_max":    float(cpts[:,2].max()),
        "xy":       centroid[:2],
    })
clusters.sort(key=lambda c: c["n_pts"], reverse=True)

print(f"\n  Cluster summary (top 20):")
print(f"  {'ID':>4}  {'pts':>8}  {'x_cent':>10}  {'y_cent':>12}  {'z_min':>8}  {'z_max':>8}  {'near_nimbb':>10}")
for cl in clusters[:20]:
    near = ("YES" if (bldg_xy_min[0]-pad <= cl["centroid"][0] <= bldg_xy_max[0]+pad and
                      bldg_xy_min[1]-pad <= cl["centroid"][1] <= bldg_xy_max[1]+pad) else "no")
    print(f"  {cl['id']:>4}  {cl['n_pts']:>8,}  {cl['centroid'][0]:>10.1f}  {cl['centroid'][1]:>12.1f}  "
          f"{cl['z_min']:>8.2f}  {cl['z_max']:>8.2f}  {near:>10}")

# ── 4. Stage-1 matching simulation for the first cluster near nimbb ────────
print("\n" + "=" * 70)
print("4. STAGE-1 MATCHING SIMULATION (clusters near nimbb)")
print("=" * 70)

def stage1_match(cpts, vert_surfs, dist_tol=DIST_TOL, z_expand=Z_EXPAND):
    cl_z_min  = float(cpts[:,2].min())
    cl_z_max  = float(cpts[:,2].max())
    centroid  = cpts.mean(axis=0)
    cxy       = centroid[:2]
    best_surf = None
    best_dist = float("inf")
    for vs in vert_surfs:
        if cl_z_max < vs["z_min"] - z_expand or cl_z_min > vs["z_max"] + z_expand:
            continue
        n2d = vs["normal_2d"]
        d   = float(np.dot(cxy - vs["origin_2d"], n2d))
        if abs(d) >= dist_tol:
            continue
        proj_xy = cxy - d * n2d
        pad2 = dist_tol
        if (proj_xy[0] < vs["xy_min"][0] - pad2 or proj_xy[0] > vs["xy_max"][0] + pad2 or
                proj_xy[1] < vs["xy_min"][1] - pad2 or proj_xy[1] > vs["xy_max"][1] + pad2):
            continue
        if abs(d) < best_dist:
            best_dist = abs(d)
            best_surf = vs
    return best_surf, best_dist

for cl in clusters:
    near = (bldg_xy_min[0]-pad <= cl["centroid"][0] <= bldg_xy_max[0]+pad and
            bldg_xy_min[1]-pad <= cl["centroid"][1] <= bldg_xy_max[1]+pad)
    if not near:
        continue
    matched, dist = stage1_match(cl["pts"], vert_surfs)
    print(f"\n  Cluster {cl['id']} ({cl['n_pts']:,} pts) @ z={cl['z_min']:.1f}–{cl['z_max']:.1f}")
    print(f"    centroid_xy = ({cl['centroid'][0]:.2f}, {cl['centroid'][1]:.2f})")
    if matched:
        print(f"    Stage1 match: poly[{matched['poly_idx']}] angle={matched['angle']:+.1f}° dist={dist:.3f}m "
              f"z=[{matched['z_min']:.1f},{matched['z_max']:.1f}]")
        # Check projection degeneracy
        txy_n3 = np.array([matched["normal_2d"][0], matched["normal_2d"][1], 0.0])
        z_up   = np.array([0,0,1.0])
        tang   = np.cross(z_up, txy_n3)
        t_norm = np.linalg.norm(tang)
        tang  /= t_norm
        txy    = tang[:2]
        t_pts  = cl["pts"][:,:2] @ txy
        t_min_cl, t_max_cl = float(t_pts.min()), float(t_pts.max())
        z_min_cl, z_max_cl = float(cl["pts"][:,2].min()), float(cl["pts"][:,2].max())
        wall_t  = matched["coords"][:,:2] @ txy
        t_min   = max(t_min_cl, float(wall_t.min()))
        t_max   = min(t_max_cl, float(wall_t.max()))
        z_min   = max(z_min_cl, matched["z_min"])
        z_max   = min(z_max_cl, matched["z_max"])
        print(f"    Projection: t=[{t_min:.3f},{t_max:.3f}]  z=[{z_min:.3f},{z_max:.3f}]")
        if t_max <= t_min or z_max <= z_min:
            print(f"    *** DEGENERATE PROJECTION — cluster extent doesn't overlap wall in t or z ***")
            print(f"      t_cl=[{t_min_cl:.2f},{t_max_cl:.2f}]  wall_t=[{wall_t.min():.2f},{wall_t.max():.2f}]")
            print(f"      z_cl=[{z_min_cl:.2f},{z_max_cl:.2f}]  wall_z=[{matched['z_min']:.2f},{matched['z_max']:.2f}]")
        else:
            print(f"    Projection OK → rect {t_max-t_min:.2f}m × {z_max-z_min:.2f}m")
    else:
        print(f"    *** NO STAGE1 MATCH ***")
        # Show why each surface failed
        cxy = cl["centroid"][:2]
        print(f"    Checking each wall surface:")
        for vs in vert_surfs[:5]:
            cl_z_min = cl["z_min"]; cl_z_max = cl["z_max"]
            z_fail = cl_z_max < vs["z_min"] - Z_EXPAND or cl_z_min > vs["z_max"] + Z_EXPAND
            n2d = vs["normal_2d"]
            d   = float(np.dot(cxy - vs["origin_2d"], n2d))
            d_fail = abs(d) >= DIST_TOL
            proj_xy = cxy - d * n2d
            xy_fail = (proj_xy[0] < vs["xy_min"][0] - DIST_TOL or proj_xy[0] > vs["xy_max"][0] + DIST_TOL or
                       proj_xy[1] < vs["xy_min"][1] - DIST_TOL or proj_xy[1] > vs["xy_max"][1] + DIST_TOL)
            print(f"      poly[{vs['poly_idx']}] angle={vs['angle']:+.1f}° z=[{vs['z_min']:.1f},{vs['z_max']:.1f}] "
                  f"z_fail={z_fail}  d={d:.2f}  d_fail={d_fail}  xy_fail={xy_fail}")

print("\nDone.")
