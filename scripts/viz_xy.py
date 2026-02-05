import numpy as np
import geopandas as gpd
import laspy
import matplotlib.pyplot as plt
from pathlib import Path

LAS_PATH = Path(r"C:\Projects\Omni2LOD3\outputs\02_clipped\NIMBB_112025_01_clipped.las")
SHP_PATH = Path(r"C:\Projects\Omni2LOD3\data\02_footprint\NIMBB_footprint_1.shp")

MAX_PLOT_POINTS = 200_000  # downsample for speed

print("Loading LAS...")
las = laspy.read(LAS_PATH)
X = np.asarray(las.x)
Y = np.asarray(las.y)
Z = np.asarray(las.z)

# Downsample for plotting
if len(X) > MAX_PLOT_POINTS:
    idx = np.random.choice(len(X), MAX_PLOT_POINTS, replace=False)
    Xp = X[idx]
    Yp = Y[idx]
    Zp = Z[idx]
else:
    Xp, Yp, Zp = X, Y, Z

print("Loading footprint...")
gdf = gpd.read_file(SHP_PATH)
poly = gdf.geometry.iloc[0]

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(Xp, Yp, c=Zp, s=1)
x_fp, y_fp = poly.exterior.xy
plt.plot(x_fp, y_fp, color='red', linewidth=2)

plt.title("Nadir View: LAS + Footprint Overlay")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Z (height)")
plt.axis("equal")
plt.show()
