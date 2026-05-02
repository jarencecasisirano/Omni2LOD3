"""Simplified analysis for extrusion debugging."""
import json, numpy as np, laspy, sys, traceback
try:
    data = json.load(open('outputs/13_openings_json/ICHEM-final-6-best.json'))
    t = data.get('transform', {})
    sc = np.array(t.get('scale',[1,1,1]), dtype=np.float64)
    tr = np.array(t.get('translate',[0,0,0]), dtype=np.float64)
    raw = np.array(data['vertices'], dtype=np.float64)
    wv = raw * sc + tr
    print(f'Vertices: {len(wv)}')
    
    las = laspy.read('outputs/11B_flat/ICHEM-extrusion.las')
    pts = np.vstack((las.x, las.y, las.z)).T
    median_xy = np.array([np.median(pts[:,0]), np.median(pts[:,1])])
    print(f'Point cloud median XY: ({median_xy[0]:.2f}, {median_xy[1]:.2f})')
    print(f'Point count: {len(pts)}')
    
    # DBSCAN  
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=0.3, min_samples=30).fit_predict(pts)
    unique = sorted(set(labels) - {-1})
    print(f'{len(unique)} clusters')
    
    for label in unique[:5]:
        cpts = pts[labels == label]
        cen = cpts.mean(axis=0)
        print(f'\nCluster {label+1}: {len(cpts)} pts, cen=({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f})')
        print(f'  X range: {cpts[:,0].min():.2f} - {cpts[:,0].max():.2f}')
        print(f'  Y range: {cpts[:,1].min():.2f} - {cpts[:,1].max():.2f}')
        print(f'  Z range: {cpts[:,2].min():.2f} - {cpts[:,2].max():.2f}')
        
except Exception as e:
    traceback.print_exc()
