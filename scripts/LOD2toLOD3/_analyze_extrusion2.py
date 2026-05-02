"""Analyze which walls each cluster matches to and why it extends."""
import json, numpy as np, laspy
from sklearn.cluster import DBSCAN

data = json.load(open('outputs/13_openings_json/ICHEM-final-6-best.json'))
t = data.get('transform', {})
sc = np.array(t.get('scale',[1,1,1]), dtype=np.float64)
tr = np.array(t.get('translate',[0,0,0]), dtype=np.float64)
raw = np.array(data['vertices'], dtype=np.float64)
wv = raw * sc + tr

las = laspy.read('outputs/11B_flat/ICHEM-extrusion.las')
pts = np.vstack((las.x, las.y, las.z)).T
median_xy = np.array([np.median(pts[:,0]), np.median(pts[:,1])])
print(f'Point cloud median XY: ({median_xy[0]:.2f}, {median_xy[1]:.2f})')

# Run DBSCAN
labels = DBSCAN(eps=0.3, min_samples=30).fit_predict(pts)
unique = sorted(set(labels) - {-1})
print(f'{len(unique)} clusters found')

# Parse walls
VERTICAL_TOL = 0.3
walls = []
for obj_id, obj in data['CityObjects'].items():
    if obj['type'] in {'Window','Door','BuildingInstallation','OtherConstruction'}:
        continue
    for geom in obj.get('geometry', []):
        boundaries = geom.get('boundaries', [])
        geom_type = geom.get('type', '')
        if geom_type == 'Solid':
            shells = list(enumerate(boundaries))
        else:
            shells = [(0, boundaries)]
        for s_idx, shell in shells:
            for p_idx, polygon in enumerate(shell):
                try:
                    coords = wv[np.array(polygon[0])]
                except:
                    continue
                if len(coords) < 3: continue
                n = np.zeros(3)
                for i in range(len(coords)):
                    c = coords[i]; nx = coords[(i+1)%len(coords)]
                    n[0] += (c[1]-nx[1])*(c[2]+nx[2])
                    n[1] += (c[2]-nx[2])*(c[0]+nx[0])
                    n[2] += (c[0]-nx[0])*(c[1]+nx[1])
                n_len = np.linalg.norm(n)
                if n_len < 1e-9: continue
                nu = n/n_len
                if abs(nu[2]) > VERTICAL_TOL: continue
                nh = n[:2]; mag = np.linalg.norm(nh)
                if mag < 1e-6: continue
                n2d = nh/mag
                cent = coords.mean(axis=0)
                wall_d = float(np.dot(cent[:2], n2d))
                walls.append({
                    'n2d': n2d, 'wall_d': wall_d, 'cent': cent,
                    'z_min': float(coords[:,2].min()), 'z_max': float(coords[:,2].max()),
                    'xy_min': coords[:,:2].min(axis=0), 'xy_max': coords[:,:2].max(axis=0),
                    'origin_2d': cent[:2].copy(), 'p_idx': p_idx
                })

for label in unique:
    cpts = pts[labels == label]
    cen = cpts.mean(axis=0)
    print(f'\n--- Cluster {label+1} ({len(cpts)} pts) cen=({cen[0]:.1f},{cen[1]:.1f},{cen[2]:.1f}) ---')
    
    # Find which walls are parallel and what their distances are
    # First find approximate normal of this cluster
    xy = cpts[:,:2]
    cxy = xy - xy.mean(axis=0)
    cov = cxy.T @ cxy
    _, vecs = np.linalg.eigh(cov)
    tang = vecs[:, 1]
    pca_n = np.array([-tang[1], tang[0]])
    
    # Find all walls with similar normal
    print(f'  PCA normal: ({pca_n[0]:.3f}, {pca_n[1]:.3f})')
    
    matching = []
    for w in walls:
        dot = abs(float(np.dot(pca_n, w['n2d'])))
        if dot > 0.95:  # nearly parallel
            # signed dist from cluster centroid to wall plane
            cluster_d = float(np.dot(cen[:2], w['n2d']))
            dist = cluster_d - w['wall_d']
            matching.append({'w': w, 'dist': dist, 'dot': dot})
    
    matching.sort(key=lambda x: abs(x['dist']))
    print(f'  {len(matching)} parallel walls found:')
    for m in matching[:10]:
        w = m['w']
        # Check if wall is on the median side
        median_d = float(np.dot(median_xy, w['n2d']))
        median_dist = median_d - w['wall_d']
        print(f'    p{w["p_idx"]} d={w["wall_d"]:.1f} dist_cluster={m["dist"]:.2f} dist_median={median_dist:.2f} dot={m["dot"]:.3f}')
