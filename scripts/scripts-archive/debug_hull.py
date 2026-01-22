import laspy, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint

# --- load building points only ---
las = laspy.read(r"D:\Projects\Thesis\outputs\building_classification\NIMBB 111725_02.las")
mask = las.classification == 6
xy = np.vstack((las.x[mask], las.y[mask])).T

# --- remove noise (same as your script) ---
db   = DBSCAN(eps=1.0, min_samples=10).fit(xy)
core = xy[db.labels_ != -1]

# --- outermost ring ---
hull = MultiPoint(core).convex_hull
x, y  = hull.exterior.xy

plt.figure(figsize=(6, 6))
plt.plot(core[:, 0], core[:, 1], 'k.', markersize=1, label='roof pts')
plt.plot(x, y, 'r-', linewidth=2, label='convex hull')
plt.gca().set_aspect('equal')
plt.title(f"roof pts {len(core)}  –  hull vertices {len(x)}")
plt.legend()
plt.show()