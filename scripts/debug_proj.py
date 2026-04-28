"""
Stage 2: Simulate find_matched_surface and compute_surface_projection
for the two window clusters vs nimbb wall surfaces.
"""
import laspy, numpy as np, json
from sklearn.cluster import DBSCAN

VERTICAL_TOL = 0.3
EPS, MIN_SAMPLES = 3.0, 30
DIST_TOL, Z_EXPAND = 2.0, 0.5

# ─── Load model ─────────────────────────────────────────────────────────────
with open('outputs/00_json_wall_merged/nimbb_021726_merged.json') as f:
    cm = json.load(f)
raw = np.array(cm['vertices'], dtype=np.float64)
t   = cm.get('transform', {})
s   = np.array(t.get('scale',[1,1,1]), dtype=np.float64)
tr  = np.array(t.get('translate',[0,0,0]), dtype=np.float64)
wv  = raw * s + tr

def newell_normal(coords):
    n = np.zeros(3)
    for i in range(len(coords)):
        c  = coords[i]; nx = coords[(i+1)%len(coords)]
        n[0] += (c[1]-nx[1])*(c[2]+nx[2])
        n[1] += (c[2]-nx[2])*(c[0]+nx[0])
        n[2] += (c[0]-nx[0])*(c[1]+nx[1])
    return n

# ─── Collect vertical surfaces ───────────────────────────────────────────────
shell = cm['CityObjects']['1-0']['geometry'][2]['boundaries'][0]
vert_surfs = []
for p_idx, polygon in enumerate(shell):
    coords = wv[np.array(polygon[0])]
    if len(coords) < 3: continue
    n = newell_normal(coords); n_len = np.linalg.norm(n)
    if n_len < 1e-9: continue
    n_unit = n / n_len
    if abs(n_unit[2]) > VERTICAL_TOL: continue
    nh = n[:2]; mag = np.linalg.norm(nh)
    if mag < 1e-6: continue
    n2d = nh / mag
    centroid = coords.mean(axis=0)
    vert_surfs.append({
        'idx': len(vert_surfs), 'poly_idx': p_idx, 'normal_2d': n2d,
        'origin_2d': centroid[:2], 'coords': coords,
        'z_min': float(coords[:,2].min()), 'z_max': float(coords[:,2].max()),
        'xy_min': coords[:,:2].min(axis=0), 'xy_max': coords[:,:2].max(axis=0),
        'angle': float(np.degrees(np.arctan2(n2d[1], n2d[0]))),
    })
print(f'Vertical surfaces: {len(vert_surfs)}')

# ─── Load LAS & cluster ──────────────────────────────────────────────────────
las  = laspy.read('outputs/11B_flat/windows_flat-2.las')
pts  = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(pts)
clusters = []
for cid in sorted(set(labels)-{-1}):
    m = labels == cid
    clusters.append({'id': cid, 'pts': pts[m]})
clusters.sort(key=lambda c: len(c['pts']), reverse=True)
print(f'Clusters: {len(clusters)}\n')

# ─── Simulate full 3-stage matching for each cluster ─────────────────────────
def _pca_normal_2d(points):
    if len(points) < 3: return None
    xy = points[:,:2]; cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    _, eig_vecs = np.linalg.eigh(cov)
    tang = eig_vecs[:,1]
    return np.array([-tang[1], tang[0]])

def find_matched(cpts):
    cl_z_min = float(cpts[:,2].min()); cl_z_max = float(cpts[:,2].max())
    cxy = cpts.mean(axis=0)[:2]
    best_surf = None; best_dist = float('inf')
    # Stage 1
    for vs in vert_surfs:
        if cl_z_max < vs['z_min']-Z_EXPAND or cl_z_min > vs['z_max']+Z_EXPAND: continue
        d = float(np.dot(cxy - vs['origin_2d'], vs['normal_2d']))
        if abs(d) >= DIST_TOL: continue
        proj = cxy - d*vs['normal_2d']
        pad = DIST_TOL
        if (proj[0] < vs['xy_min'][0]-pad or proj[0] > vs['xy_max'][0]+pad or
                proj[1] < vs['xy_min'][1]-pad or proj[1] > vs['xy_max'][1]+pad): continue
        if abs(d) < best_dist:
            best_dist = abs(d); best_surf = vs
    if best_surf:
        return best_surf, 'Stage1', best_dist
    # Stage 2
    pca_n = _pca_normal_2d(cpts)
    if pca_n is not None:
        best_dot = -1.0
        for vs in vert_surfs:
            dot = abs(float(np.dot(pca_n, vs['normal_2d'])))
            if dot > best_dot: best_dot = dot; best_surf = vs
        if best_surf:
            return best_surf, 'Stage2', best_dot
    # Stage 3
    for vs in vert_surfs:
        d = abs(float(np.dot(cxy - vs['origin_2d'], vs['normal_2d'])))
        if d < best_dist: best_dist = d; best_surf = vs
    return best_surf, 'Stage3', best_dist

def compute_proj(cpts, surf):
    n2d = surf['normal_2d']
    wall_depth = float(np.dot(surf['origin_2d'], n2d))
    z_up = np.array([0.,0.,1.]); wall_n3 = np.array([n2d[0],n2d[1],0.])
    tang = np.cross(z_up, wall_n3); t_norm = np.linalg.norm(tang)
    tang = tang / t_norm if t_norm > 1e-9 else np.array([1.,0.,0.])
    txy = tang[:2]
    t_pts = cpts[:,:2] @ txy; z_pts = cpts[:,2]
    t_min_cl, t_max_cl = float(t_pts.min()), float(t_pts.max())
    z_min_cl, z_max_cl = float(z_pts.min()), float(z_pts.max())
    wall_t = surf['coords'][:,:2] @ txy
    t_min = max(t_min_cl, float(wall_t.min())); t_max = min(t_max_cl, float(wall_t.max()))
    z_min = max(z_min_cl, surf['z_min']);        z_max = min(z_max_cl, surf['z_max'])
    return t_min, t_max, z_min, z_max, t_min_cl, t_max_cl, z_min_cl, z_max_cl

for cl in clusters:
    cpts = cl['pts']
    print(f"=== Cluster {cl['id']} ({len(cpts):,} pts) ===")
    centroid = cpts.mean(axis=0)
    print(f"  centroid : ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    print(f"  z_range  : {cpts[:,2].min():.3f} – {cpts[:,2].max():.3f}")
    print(f"  xy_range : x=[{cpts[:,0].min():.1f},{cpts[:,0].max():.1f}]  y=[{cpts[:,1].min():.1f},{cpts[:,1].max():.1f}]")

    surf, stage, score = find_matched(cpts)
    if surf is None:
        print(f"  *** NO MATCH ***")
        continue
    print(f"  Matched  : poly[{surf['poly_idx']}] via {stage} (score={score:.3f})")
    print(f"  Wall     : angle={surf['angle']:+.1f}°  z=[{surf['z_min']:.2f},{surf['z_max']:.2f}]")
    print(f"  Wall xy  : [{surf['xy_min'][0]:.1f},{surf['xy_min'][1]:.1f}]–[{surf['xy_max'][0]:.1f},{surf['xy_max'][1]:.1f}]")

    t_min, t_max, z_min, z_max, t_min_cl, t_max_cl, z_min_cl, z_max_cl = compute_proj(cpts, surf)
    print(f"\n  Projection details:")
    print(f"    cluster t span : [{t_min_cl:.3f}, {t_max_cl:.3f}]  width={t_max_cl-t_min_cl:.3f} m")
    print(f"    wall    t span : [{surf['coords'][:,:2].min(axis=0) @ surf['normal_2d']:.3f}, ...]")
    print(f"    clipped t span : [{t_min:.3f}, {t_max:.3f}]  width={t_max-t_min:.3f} m")
    print(f"    cluster z span : [{z_min_cl:.3f}, {z_max_cl:.3f}]  height={z_max_cl-z_min_cl:.3f} m")
    print(f"    wall    z span : [{surf['z_min']:.3f}, {surf['z_max']:.3f}]  height={surf['z_max']-surf['z_min']:.3f} m")
    print(f"    clipped z span : [{z_min:.3f}, {z_max:.3f}]  height={z_max-z_min:.3f} m")

    if t_max <= t_min:
        print(f"\n  *** DEGENERATE: t overlap = 0 (cluster t-range doesn't intersect wall t-range) ***")
    elif z_max <= z_min:
        print(f"\n  *** DEGENERATE: z overlap = 0 ***")
    else:
        print(f"\n  Projection OK: {t_max-t_min:.3f} m wide × {z_max-z_min:.3f} m tall")
        print(f"  (This should work — check why no opening was created in output)")
    print()
