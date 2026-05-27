import laspy, numpy as np, json

# ─── 1. LAS info ────────────────────────────────────────────────────────────
las = laspy.read('outputs/11B_flat/windows_flat-2.las')
pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
print('LAS total pts:', pts.shape[0])
print('  x:', round(float(pts[:,0].min()),2), '–', round(float(pts[:,0].max()),2))
print('  y:', round(float(pts[:,1].min()),2), '–', round(float(pts[:,1].max()),2))
print('  z:', round(float(pts[:,2].min()),2), '–', round(float(pts[:,2].max()),2))

# ─── 2. Model bounds ────────────────────────────────────────────────────────
with open('outputs/00_json_wall_merged/nimbb_021726_merged.json') as f:
    cm = json.load(f)
raw = np.array(cm['vertices'], dtype=np.float64)
t   = cm.get('transform', {})
s   = np.array(t.get('scale',[1,1,1]), dtype=np.float64)
tr  = np.array(t.get('translate',[0,0,0]), dtype=np.float64)
wv  = raw * s + tr
print('\nNIMBB building bounds:')
print('  x:', round(float(wv[:,0].min()),2), '–', round(float(wv[:,0].max()),2))
print('  y:', round(float(wv[:,1].min()),2), '–', round(float(wv[:,1].max()),2))
print('  z:', round(float(wv[:,2].min()),2), '–', round(float(wv[:,2].max()),2))

# ─── 3. How many LAS pts are near the building? ─────────────────────────────
pad = 100
mask = ((pts[:,0] >= wv[:,0].min()-pad) & (pts[:,0] <= wv[:,0].max()+pad) &
        (pts[:,1] >= wv[:,1].min()-pad) & (pts[:,1] <= wv[:,1].max()+pad))
pnear = pts[mask]
print('\nPts within 100m of nimbb:', pnear.shape[0])
if pnear.shape[0] > 0:
    print('  z range nearby:', round(float(pnear[:,2].min()),2), '–', round(float(pnear[:,2].max()),2))

# ─── 4. DBSCAN ──────────────────────────────────────────────────────────────
from sklearn.cluster import DBSCAN
print('\nRunning DBSCAN (eps=3, min_samples=30)...')
labels = DBSCAN(eps=3.0, min_samples=30).fit_predict(pts)
ulabels = sorted(set(labels) - {-1})
print('  Clusters:', len(ulabels), '  noise:', int((labels==-1).sum()))

clusters = []
for cid in ulabels:
    m = labels == cid
    cp = pts[m]
    clusters.append({'id': cid, 'n': int(m.sum()), 'cx': float(cp[:,0].mean()),
                     'cy': float(cp[:,1].mean()), 'z_min': float(cp[:,2].min()), 'z_max': float(cp[:,2].max())})
clusters.sort(key=lambda x: x['n'], reverse=True)

bx0,by0 = float(wv[:,0].min()), float(wv[:,1].min())
bx1,by1 = float(wv[:,0].max()), float(wv[:,1].max())
print('\nTop-20 clusters:')
print('  {:>4}  {:>8}  {:>11}  {:>12}  {:>8}  {:>8}  {}'.format('ID','pts','cx','cy','z_min','z_max','near_nimbb'))
for cl in clusters[:20]:
    near = 'YES' if (bx0-pad <= cl['cx'] <= bx1+pad and by0-pad <= cl['cy'] <= by1+pad) else 'no'
    print('  {:>4}  {:>8,}  {:>11.1f}  {:>12.1f}  {:>8.2f}  {:>8.2f}  {}'.format(
        cl['id'], cl['n'], cl['cx'], cl['cy'], cl['z_min'], cl['z_max'], near))
