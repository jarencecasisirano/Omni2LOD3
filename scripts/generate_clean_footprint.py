import os
import time
import numpy as np
import geopandas as gpd
import osmnx as ox

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import snap

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
    coords = coords - coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return np.arctan2(axis[1], axis[0])


def fetch_osm_building(footprint_poly_wgs, buffer_m=30):
    """
    footprint_poly_wgs MUST be EPSG:4326
    """
    
    # Validate geometry first
    if not footprint_poly_wgs.is_valid:
        print("⚠️  Invalid geometry detected, attempting to fix...")
        footprint_poly_wgs = footprint_poly_wgs.buffer(0)
    
    if footprint_poly_wgs.is_empty:
        raise RuntimeError("❌ Empty geometry after validation")

    # --- convert meter buffer → degrees (safe for small areas) ---
    buffer_deg = buffer_m / 111_320.0  # meters per degree ≈

    minx, miny, maxx, maxy = footprint_poly_wgs.bounds

    north = maxy + buffer_deg
    south = miny - buffer_deg
    east  = maxx + buffer_deg
    west  = minx - buffer_deg

    print(f"  Searching OSM bbox: N={north:.6f}, S={south:.6f}, E={east:.6f}, W={west:.6f}")

    try:
        # Use the polygon directly instead of bbox for better compatibility
        gdf = ox.features_from_polygon(
            footprint_poly_wgs.buffer(buffer_deg),
            tags={"building": True}
        )
    except Exception as e:
        print(f"⚠️  Polygon query failed, trying bbox method: {e}")
        # Fallback to manual bbox creation
        from shapely.geometry import box
        bbox_poly = box(west, south, east, north)
        gdf = ox.features_from_polygon(
            bbox_poly,
            tags={"building": True}
        )

    if gdf.empty:
        raise RuntimeError("❌ No OSM buildings found nearby")

    # keep only polygonal buildings
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    
    if gdf.empty:
        raise RuntimeError("❌ No polygonal OSM buildings found nearby")

    # --- compute area SAFELY ---
    gdf_proj = gdf.to_crs("EPSG:3857")
    gdf["area"] = gdf_proj.geometry.area

    # choose best candidate by overlap, not just area
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

    # Filter out buildings with zero overlap
    gdf = gdf[gdf["overlap"] > 0]
    
    if gdf.empty:
        print("⚠️  No overlapping buildings, using nearest by distance instead")
        gdf_proj["distance"] = gdf_proj.geometry.distance(rough_proj)
        best = gdf.loc[gdf_proj["distance"].idxmin()].geometry
    else:
        best = gdf.sort_values(
            ["overlap", "area"], ascending=False
        ).iloc[0].geometry

    # unwrap multipolygon
    if best.geom_type == "MultiPolygon":
        best = max(best.geoms, key=lambda g: g.area)

    print(f"✅ Found OSM building with {len(best.exterior.coords)} vertices")
    return best


def align_osm_to_footprint(osm_poly, target_poly):
    # Target orientation
    coords = np.array(target_poly.exterior.coords)
    target_angle = pca_angle(coords[:, :2])

    # Rotate OSM
    aligned = rotate(
        osm_poly,
        target_angle,
        origin="centroid",
        use_radians=True
    )

    # Translate to match centroid
    dx = target_poly.centroid.x - aligned.centroid.x
    dy = target_poly.centroid.y - aligned.centroid.y

    aligned = translate(aligned, xoff=dx, yoff=dy)

    return aligned


# ------------------ main ------------------
def main():
    latest_geojson = get_latest_geojson(FOOTPRINT_DIR)
    print(f"\n📂 Using footprint:\n   {latest_geojson}")

    gdf = gpd.read_file(latest_geojson)

    if gdf.empty or not isinstance(gdf.geometry.iloc[0], Polygon):
        raise RuntimeError("❌ Invalid footprint geometry")

    gdf = gdf.to_crs(CRS_EPSG)
    rough_poly = gdf.geometry.iloc[0]
    
    # Validate the rough polygon
    if not rough_poly.is_valid:
        print("⚠️  Rough polygon invalid, attempting fix...")
        rough_poly = rough_poly.buffer(0)

    print(f"\n🔍 Rough footprint area: {rough_poly.area:.2f} m²")

    print("\n🌐 Fetching OSM building footprint…")
    # --- reproject rough footprint to WGS84 for OSMnx ---
    rough_gdf = gpd.GeoDataFrame(geometry=[rough_poly], crs=CRS_EPSG)
    rough_wgs = rough_gdf.to_crs("EPSG:4326").geometry.iloc[0]
    
    # Additional validation after reprojection
    if not rough_wgs.is_valid:
        print("⚠️  WGS84 polygon invalid, fixing...")
        rough_wgs = rough_wgs.buffer(0)

    osm_poly_wgs = fetch_osm_building(rough_wgs, buffer_m=50)

    # --- bring OSM back to UTM ---
    osm_gdf = gpd.GeoDataFrame(geometry=[osm_poly_wgs], crs="EPSG:4326")
    osm_poly = osm_gdf.to_crs(CRS_EPSG).geometry.iloc[0]

    # Debug output
    debug_dir = os.path.join(ROOT, "outputs", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    gpd.GeoDataFrame(
        geometry=[osm_poly],
        crs=CRS_EPSG
    ).to_file(
        os.path.join(debug_dir, "osm_raw_debug.geojson"),
        driver="GeoJSON"
    )
    
    print(f"✅ OSM raw footprint saved to debug folder")

    print("\n🔄 Aligning OSM footprint to LiDAR footprint…")
    aligned = align_osm_to_footprint(osm_poly, rough_poly)

    print("📍 Snapping to rough footprint edges…")
    aligned = snap(aligned, rough_poly, tolerance=1.0)

    # Slight simplification for clean walls
    aligned = aligned.simplify(0.2, preserve_topology=True)
    
    # Final validation
    if not aligned.is_valid:
        print("⚠️  Final polygon invalid, fixing...")
        aligned = aligned.buffer(0)

    out_name = os.path.splitext(os.path.basename(latest_geojson))[0]
    out_path = os.path.join(CLEAN_DIR, f"{out_name}_clean.geojson")

    out_gdf = gpd.GeoDataFrame(geometry=[aligned], crs=CRS_EPSG)
    out_gdf.to_file(out_path, driver="GeoJSON")

    print(f"\n✅ Clean footprint saved to:")
    print(f"   {out_path}")
    print(f"   Area: {aligned.area:.2f} m²")


# ------------------ run ------------------
if __name__ == "__main__":
    main()