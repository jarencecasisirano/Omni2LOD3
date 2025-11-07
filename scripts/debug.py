import laspy
import numpy as np
import os

las_path = os.path.join("..", "outputs", "downsampled", "MBB_Roof_downsampled_0_5.las")
las = laspy.read(las_path)
z = las.z
print("Z min:", np.min(z))
print("Z max:", np.max(z))
