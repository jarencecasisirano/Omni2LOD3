import os
import sys
import numpy as np
import geopandas as gpd
import osmnx as ox

from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate, scale
from scipy.spatial import cKDTree

# ------------------ paths ------------------
ROOT = r"C:\Projects\Thesis"
FOOTPRINT_DIR = os.path.join(ROOT, "outputs", "footprint")
CLEAN_DIR = os.path.join(ROOT, "outputs", "clean_footprint")
os.makedirs(CLEAN_DIR, exist_ok=True)

CRS_EPSG = "EPSG:32651"

# ------------------ helpers ------------------
def get_latest_geojson(folder):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".geojson")
    ]
    if not files:
        raise RuntimeError("❌ No GeoJSON files found in outputs/footprint")

    latest = max(files, key=os.path.getmtime)
    return latest


def pca_angle(coords):
    """Get principal axis angle from coordinates"""
    coords = coords - coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return np.arctan2(axis[1], axis[0])


def fetch_osm_building(footprint_poly_wgs, buffer_m=50):
    """Fetch OSM building footprint"""
    
    if not footprint_poly_wgs.is_valid:
        print("⚠️  Invalid geometry detected, attempting to fix...")
        footprint_poly_wgs = footprint_poly_wgs.buffer(0)
    
    if footprint_poly_wgs.is_empty:
        raise RuntimeError("❌ Empty geometry after validation")

    buffer_deg = buffer_m / 111_320.0

    minx, miny, maxx, maxy = footprint_poly_wgs.bounds
    north = maxy + buffer_deg
    south = miny - buffer_deg
    east  = maxx + buffer_deg
    west  = minx - buffer_deg

    print(f"  Searching OSM bbox: N={north:.6f}, S={south:.6f}, E={east:.6f}, W={west:.6f}")

    try:
        gdf = ox.features_from_polygon(
            footprint_poly_wgs.buffer(buffer_deg),
            tags={"building": True}
        )
    except Exception as e:
        print(f"⚠️  Polygon query failed, trying bbox method: {e}")
        from shapely.geometry import box
        bbox_poly = box(west, south, east, north)
        gdf = ox.features_from_polygon(bbox_poly, tags={"building": True})

    if gdf.empty:
        raise RuntimeError("❌ No OSM buildings found nearby")

    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    
    if gdf.empty:
        raise RuntimeError("❌ No polygonal OSM buildings found nearby")

    gdf_proj = gdf.to_crs("EPSG:3857")
    gdf["area"] = gdf_proj.geometry.area

    rough_proj = (
        gpd.GeoSeries([footprint_poly_wgs], crs="EPSG:4326")
        .to_crs("EPSG:3857")
        .iloc[0]
    )

    def overlap_score(geom):
        try:
            return geom.intersection(rough_proj).area
        except:
            return 0

    gdf["overlap"] = gdf_proj.geometry.apply(overlap_score)
    gdf = gdf[gdf["overlap"] > 0]
    
    if gdf.empty:
        print("⚠️  No overlapping buildings, using nearest by distance")
        gdf_proj["distance"] = gdf_proj.geometry.distance(rough_proj)
        best = gdf.loc[gdf_proj["distance"].idxmin()].geometry
    else:
        best = gdf.sort_values(["overlap", "area"], ascending=False).iloc[0].geometry

    if best.geom_type == "MultiPolygon":
        best = max(best.geoms, key=lambda g: g.area)

    print(f"✅ Found OSM building with {len(best.exterior.coords)-1} vertices")
    return best


def align_and_scale_osm(osm_poly, lidar_poly):
    """
    Align OSM to LiDAR: rotation + translation + scaling
    This gets OSM roughly in the right place and size
    """
    
    osm_coords = np.array(osm_poly.exterior.coords[:-1])
    lidar_coords = np.array(lidar_poly.exterior.coords[:-1])
    
    # Get orientations
    osm_angle = pca_angle(osm_coords)
    lidar_angle = pca_angle(lidar_coords)
    rotation_angle = lidar_angle - osm_angle
    
    print(f"  📐 Rotation: {np.degrees(rotation_angle):.2f}°")
    
    # Step 1: Rotate
    aligned = rotate(osm_poly, rotation_angle, origin='centroid', use_radians=True)
    
    # Step 2: Scale to match LiDAR size
    osm_area = osm_poly.area
    lidar_area = lidar_poly.area
    scale_factor = np.sqrt(lidar_area / osm_area)
    
    print(f"  📏 Scale: {scale_factor:.3f}x")
    
    aligned = scale(aligned, xfact=scale_factor, yfact=scale_factor, origin='centroid')
    
    # Step 3: Translate to LiDAR centroid
    dx = lidar_poly.centroid.x - aligned.centroid.x
    dy = lidar_poly.centroid.y - aligned.centroid.y
    aligned = translate(aligned, xoff=dx, yoff=dy)
    
    return aligned


def deform_osm_to_lidar(osm_poly, lidar_poly, influence_distance=5.0, smoothing=0.5):
    """
    Deform OSM vertices towards LiDAR footprint boundary.
    
    Parameters:
    - influence_distance: max distance to pull vertices (meters)
    - smoothing: 0=no movement, 1=full movement to nearest point
    """
    
    osm_coords = np.array(osm_poly.exterior.coords[:-1])
    lidar_coords = np.array(lidar_poly.exterior.coords[:-1])
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(lidar_coords)
    
    deformed_coords = []
    
    print(f"\n   Deforming {len(osm_coords)} OSM vertices...")
    
    for i, osm_pt in enumerate(osm_coords):
        # Find nearest LiDAR point
        dist, idx = tree.query(osm_pt)
        nearest_lidar = lidar_coords[idx]
        
        # Only move if within influence distance
        if dist <= influence_distance:
            # Interpolate between OSM position and LiDAR position
            weight = smoothing * (1.0 - dist / influence_distance)
            new_pt = osm_pt + weight * (nearest_lidar - osm_pt)
            deformed_coords.append(new_pt)
        else:
            # Too far, keep OSM position
            deformed_coords.append(osm_pt)
    
    return Polygon(deformed_coords)


def smart_simplify(poly, tolerance=0.3):
    """
    Simplify while trying to preserve important corners
    """
    simplified = poly.simplify(tolerance, preserve_topology=True)
    
    # Make sure we didn't over-simplify
    if len(simplified.exterior.coords) < 4:
        return poly
    
    return simplified


# ------------------ main ------------------
def main(influence_distance=8.0, smoothing=0.7, simplify_tolerance=0.3):
    """
    Parameters to tune the deformation:
    
    influence_distance (meters):
    - How far to "pull" OSM vertices towards LiDAR
    - Larger = more aggressive fitting
    - Try: 5.0 (conservative) to 15.0 (aggressive)
    
    smoothing (0.0 to 1.0):
    - How much to move vertices
    - 0.0 = keep OSM exactly
    - 1.0 = move fully to nearest LiDAR point
    - Try: 0.5 (balanced) to 0.8 (more fitting)
    
    simplify_tolerance (meters):
    - Clean up micro-details
    - Try: 0.2 (keep details) to 0.5 (cleaner)
    """
    
    latest_geojson = get_latest_geojson(FOOTPRINT_DIR)
    print(f"\n📂 Using footprint:\n   {latest_geojson}")

    gdf = gpd.read_file(latest_geojson)

    if gdf.empty or not isinstance(gdf.geometry.iloc[0], Polygon):
        raise RuntimeError("❌ Invalid footprint geometry")

    gdf = gdf.to_crs(CRS_EPSG)
    rough_poly = gdf.geometry.iloc[0]
    
    if not rough_poly.is_valid:
        print("⚠️  Rough polygon invalid, attempting fix...")
        rough_poly = rough_poly.buffer(0)

    print(f"\n🔍 LiDAR footprint:")
    print(f"   Area: {rough_poly.area:.2f} m²")
    print(f"   Vertices: {len(rough_poly.exterior.coords)-1}")

    print("\n🌐 Fetching OSM building reference…")
    rough_gdf = gpd.GeoDataFrame(geometry=[rough_poly], crs=CRS_EPSG)
    rough_wgs = rough_gdf.to_crs("EPSG:4326").geometry.iloc[0]
    
    if not rough_wgs.is_valid:
        print("⚠️  WGS84 polygon invalid, fixing...")
        rough_wgs = rough_wgs.buffer(0)

    osm_poly_wgs = fetch_osm_building(rough_wgs, buffer_m=50)
    osm_gdf = gpd.GeoDataFrame(geometry=[osm_poly_wgs], crs="EPSG:4326")
    osm_poly = osm_gdf.to_crs(CRS_EPSG).geometry.iloc[0]

    print(f"\n📐 OSM reference:")
    print(f"   Area: {osm_poly.area:.2f} m²")
    print(f"   Vertices: {len(osm_poly.exterior.coords)-1}")

    # Step 1: Align and scale OSM to roughly match LiDAR
    print("\n🔄 Aligning & scaling OSM to LiDAR...")
    osm_aligned = align_and_scale_osm(osm_poly, rough_poly)
    
    print(f"   Aligned area: {osm_aligned.area:.2f} m²")

    # Debug: save aligned OSM
    debug_dir = os.path.join(ROOT, "outputs", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    gpd.GeoDataFrame(geometry=[osm_aligned], crs=CRS_EPSG).to_file(
        os.path.join(debug_dir, "osm_aligned_scaled.geojson"),
        driver="GeoJSON"
    )
    print(f"   💾 Saved aligned OSM to debug/osm_aligned_scaled.geojson")

    # Step 2: Deform OSM vertices towards LiDAR boundary
    print(f"\n🔨 Deforming OSM to fit LiDAR data...")
    print(f"   Influence distance: {influence_distance}m")
    print(f"   Smoothing factor: {smoothing}")
    
    clean_poly = deform_osm_to_lidar(
        osm_aligned, 
        rough_poly, 
        influence_distance=influence_distance,
        smoothing=smoothing
    )
    
    # Step 3: Light simplification
    if simplify_tolerance > 0:
        print(f"\n✨ Simplifying (tolerance={simplify_tolerance}m)...")
        clean_poly = smart_simplify(clean_poly, simplify_tolerance)
    
    # Validation
    if not clean_poly.is_valid:
        print("⚠️  Fixing invalid geometry...")
        clean_poly = clean_poly.buffer(0)

    print(f"\n✨ Clean footprint:")
    print(f"   Area: {clean_poly.area:.2f} m²")
    print(f"   Vertices: {len(clean_poly.exterior.coords)-1}")

    # Save result
    out_name = os.path.splitext(os.path.basename(latest_geojson))[0]
    out_path = os.path.join(CLEAN_DIR, f"{out_name}_clean.geojson")

    out_gdf = gpd.GeoDataFrame(geometry=[clean_poly], crs=CRS_EPSG)
    out_gdf.to_file(out_path, driver="GeoJSON")

    print(f"\n✅ Clean footprint saved to:")
    print(f"   {out_path}")
    
    print(f"\n📊 Comparison:")
    print(f"   OSM:    {osm_poly.area:.2f} m² | {len(osm_poly.exterior.coords)-1} vertices")
    print(f"   LiDAR:  {rough_poly.area:.2f} m² | {len(rough_poly.exterior.coords)-1} vertices")
    print(f"   Clean:  {clean_poly.area:.2f} m² | {len(clean_poly.exterior.coords)-1} vertices")
    
    area_diff = abs(rough_poly.area - clean_poly.area)
    area_pct = (area_diff / rough_poly.area) * 100
    print(f"   Difference from LiDAR: {area_diff:.2f} m² ({area_pct:.1f}%)")


# ------------------ run ------------------
if __name__ == "__main__":
    # 🎛️ TUNING GUIDE:
    # 
    # If result is too close to OSM (not fitting your data well):
    #   - INCREASE influence_distance (try 10-15)
    #   - INCREASE smoothing (try 0.8-0.9)
    #
    # If result is too wonky (losing OSM's clean shape):
    #   - DECREASE influence_distance (try 3-5)
    #   - DECREASE smoothing (try 0.4-0.6)
    #
    # If edges are too jagged:
    #   - INCREASE simplify_tolerance (try 0.4-0.6)
    
    main(
        influence_distance=8.0,   # meters - how far to search for LiDAR points
        smoothing=0.7,            # 0-1 - how much to move vertices
        simplify_tolerance=0.3    # meters - cleanup tolerance
    )