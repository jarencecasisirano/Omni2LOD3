"""Temp analysis script for extrusion debugging."""
import json, numpy as np, laspy, sys

data = json.load(open('outputs/13_openings_json/ICHEM-final-6-best.json'))
t = data.get('transform', {})
sc = np.array(t.get('scale',[1,1,1]), dtype=np.float64)
tr = np.array(t.get('translate',[0,0,0]), dtype=np.float64)
raw = np.array(data['vertices'], dtype=np.float64)
wv = raw * sc + tr

las = laspy.read('outputs/11B_flat/ICHEM-extrusion.las')
pts = np.vstack((las.x, las.y, las.z)).T
print('Point count:', len(pts))
print('X range:', pts[:,0].min(), '-', pts[:,0].max())
print('Y range:', pts[:,1].min(), '-', pts[:,1].max())
print('Z range:', pts[:,2].min(), '-', pts[:,2].max())
print('Median XY:', np.median(pts[:,0]), np.median(pts[:,1]))
print()

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
                if len(coords) < 3:
                    continue
                n = np.zeros(3)
                for i in range(len(coords)):
                    c = coords[i]
                    nx = coords[(i+1)%len(coords)]
                    n[0] += (c[1]-nx[1])*(c[2]+nx[2])
                    n[1] += (c[2]-nx[2])*(c[0]+nx[0])
                    n[2] += (c[0]-nx[0])*(c[1]+nx[1])
                n_len = np.linalg.norm(n)
                if n_len < 1e-9: continue
                nu = n/n_len
                if abs(nu[2]) > VERTICAL_TOL: continue
                nh = n[:2]
                mag = np.linalg.norm(nh)
                if mag < 1e-6: continue
                n2d = nh/mag
                cent = coords.mean(axis=0)
                wall_d = float(np.dot(cent[:2], n2d))
                angle = np.degrees(np.arctan2(n2d[1], n2d[0]))
                walls.append({'angle': angle, 'wall_d': wall_d, 'cent': cent, 'n2d': n2d, 's_idx': s_idx, 'p_idx': p_idx})
                print(f'  wall s{s_idx}p{p_idx} angle={angle:+.1f} d={wall_d:.1f} cent=({cent[0]:.1f},{cent[1]:.1f},{cent[2]:.1f}) n2d=({n2d[0]:.4f},{n2d[1]:.4f})')

# Group walls by similar normal direction (within 5 degrees)
print('\n--- Parallel wall groups (same normal direction) ---')
groups = {}
for w in walls:
    # normalize angle to 0-180 range to group parallel walls
    a = w['angle'] % 180
    key = round(a / 5) * 5
    groups.setdefault(key, []).append(w)

median_xy = np.array([np.median(pts[:,0]), np.median(pts[:,1])])
print(f'Point cloud median XY: ({median_xy[0]:.2f}, {median_xy[1]:.2f})')

for key in sorted(groups.keys()):
    g = groups[key]
    print(f'\n  Group ~{key}deg ({len(g)} walls):')
    for w in g:
        # distance from median to wall plane
        dist = np.dot(median_xy, w['n2d']) - w['wall_d']
        print(f'    s{w["s_idx"]}p{w["p_idx"]} d={w["wall_d"]:.1f} angle={w["angle"]:+.1f} dist_from_median={dist:.2f}')
