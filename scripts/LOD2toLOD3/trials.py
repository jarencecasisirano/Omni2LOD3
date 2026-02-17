import laspy

las = laspy.read("outputs/03_pointclouds/NIMBB-3-BEST/D.las")
print(las.header)
