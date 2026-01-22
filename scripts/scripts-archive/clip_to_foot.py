# clip_to_footprint.py (ROBUST VERSION)
import os, sys, json
import laspy
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "clipped")

def clip_to_building(input_las, footprint_geojson, output_las=None):
    """
    Clip point cloud to building footprint bounds (+buffer)
    Handles paths with spaces and provides clear error messages.
    """
    # Normalize paths
    input_las = os.path.abspath(input_las.strip('"').strip("'"))
    footprint_geojson = os.path.abspath(footprint_geojson.strip('"').strip("'"))
    
    # Validate input LAS exists
    if not os.path.exists(input_las):
        print(f"[ERROR] Input LAS file not found:\n   {input_las}")
        return False
    
    # Validate footprint exists
    if not os.path.exists(footprint_geojson):
        print(f"[ERROR] Footprint GeoJSON not found:\n   {footprint_geojson}")
        return False
    
    # Validate file extensions
    if not input_las.lower().endswith(('.las', '.laz')):
        print(f"[ERROR] Input file must be .las or .laz, got:\n   {input_las}")
        return False
    
    if not footprint_geojson.lower().endswith('.geojson'):
        print(f"[ERROR] Footprint must be .geojson, got:\n   {footprint_geojson}")
        return False
    
    # Load footprint and get bounds
    try:
        with open(footprint_geojson) as f:
            geojson = json.load(f)
        
        # Extract coordinates (handles both Feature and FeatureCollection)
        if geojson['type'] == 'FeatureCollection':
            if not geojson['features']:
                print("[ERROR] GeoJSON has no features")
                return False
            coords = geojson['features'][0]['geometry']['coordinates'][0]
        else:
            coords = geojson['geometry']['coordinates'][0]
            
        xs, ys = zip(*coords)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
    except Exception as e:
        print(f"[ERROR] Could not parse GeoJSON: {e}")
        print(f"   File: {footprint_geojson}")
        return False
    
    # Add 2m buffer
    buffer = 2.0
    x_min, x_max = x_min - buffer, x_max + buffer
    y_min, y_max = y_min - buffer, y_max + buffer
    
    # Load LAS and clip
    try:
        las = laspy.read(input_las)
        mask = (
            (las.x >= x_min) & (las.x <= x_max) &
            (las.y >= y_min) & (las.y <= y_max)
        )
        clipped = las[mask]
    except Exception as e:
        print(f"[ERROR] Could not process LAS file: {e}")
        return False
    
    # Generate output path if not provided
    if output_las is None:
        name = os.path.splitext(os.path.basename(input_las))[0]
        output_las = os.path.join(DEFAULT_OUTPUT_DIR, f"{name}_clipped.las")
    else:
        output_las = os.path.abspath(output_las.strip('"').strip("'"))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_las), exist_ok=True)
    
    # Save
    try:
        clipped.write(output_las)
        print(f"\n✓ SUCCESS: Clipped from {len(las)} to {len(clipped)} points")
        print(f"📁 Saved to: {output_las}")
        print(f"📐 Bounds: X[{x_min:.2f}, {x_max:.2f}] Y[{y_min:.2f}, {y_max:.2f}]")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save clipped file: {e}")
        return False

if __name__ == "__main__":
    print("CLIP POINT CLOUD TO FOOTPRINT")
    print("="*50)
    
    # Print usage with example including quotes
    if len(sys.argv) < 3:
        print("Usage: python clip_to_footprint.py \"input.las\" \"footprint.geojson\" [\"output.las\"]")
        print("\n⚠️  IMPORTANT: Use quotes around paths with spaces!")
        print("\nExample:")
        print(r'python clip_to_footprint.py "D:\path with spaces\input.las" "D:\path\footprint.geojson"')
        sys.exit(1)
    
    # Get arguments
    input_las = sys.argv[1]
    footprint_geojson = sys.argv[2]
    output_las = sys.argv[3] if len(sys.argv) > 3 else None
    
    print(f"Input LAS: {input_las}")
    print(f"Footprint: {footprint_geojson}")
    if output_las:
        print(f"Output: {output_las}")
    print("="*50)
    
    success = clip_to_building(input_las, footprint_geojson, output_las)
    sys.exit(0 if success else 1)