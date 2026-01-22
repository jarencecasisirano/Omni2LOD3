# shift_buildings_up.py
# Standalone script to fix vertical clearance for CityForge
# Usage: python shift_buildings_up.py "D:\path\to\your_file.las"

import os
import sys
import numpy as np
import laspy

def shift_building_points_up(input_path, output_dir=None):
    """
    Shifts building points (Class 6) vertically to ensure >3m clearance above ground.
    
    Args:
        input_path: Path to input LAS file
        output_dir: Directory to save shifted file (default: thesis/outputs/shifted)
    """
    # Setup paths
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return False

    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "..", "outputs", "shifted")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, f"{name}_shifted{ext}")
    
    print(f"\n=== Processing: {basename} ===")
    
    # Load LAS file
    try:
        las = laspy.read(input_path)
        print(f"✓ Loaded {len(las.points)} points")
    except Exception as e:
        print(f"[ERROR] Failed to load LAS file: {e}")
        return False
    
    # Get ground and building points
    ground_mask = las.classification == 2
    building_mask = las.classification == 6
    
    if not np.any(ground_mask):
        print("[ERROR] No ground points (Class 2) found. Cannot calculate clearance.")
        return False
    
    if not np.any(building_mask):
        print("[ERROR] No building points (Class 6) found. Nothing to shift.")
        return False
    
    # Calculate current clearance
    ground_max_z = np.max(las.points.Z[ground_mask])
    building_min_z = np.min(las.points.Z[building_mask])
    clearance = (building_min_z - ground_max_z) * las.header.scales[2]
    
    print(f"\n--- Before Shift ---")
    print(f"Ground Z-max: {ground_max_z * las.header.scales[2]:.2f}m")
    print(f"Building Z-min: {building_min_z * las.header.scales[2]:.2f}m")
    print(f"Vertical clearance: {clearance:.2f}m (need >3.0m)")
    
    # Calculate shift if needed
    if clearance >= 3.0:
        print("✓ Clearance is already good. No shift needed.")
        shift_amount = 0
    else:
        shift_needed = (3.0 - clearance) + 0.5  # 0.5m buffer
        shift_amount = int(shift_needed / las.header.scales[2])  # Convert to integer shifts
        print(f"\n⚠️  Shifting building points up by {shift_needed:.2f}m ({shift_amount} units)")
        
        # Apply shift to building points only
        las.points.Z[building_mask] += shift_amount
    
    # Verify new clearance
    new_building_min_z = np.min(las.points.Z[building_mask])
    new_clearance = (new_building_min_z - ground_max_z) * las.header.scales[2]
    
    print(f"\n--- After Shift ---")
    print(f"Building Z-min: {new_building_min_z * las.header.scales[2]:.2f}m")
    print(f"New vertical clearance: {new_clearance:.2f}m")
    
    # Save shifted file
    try:
        las.write(output_path)
        print(f"\n✓ SUCCESS: Saved shifted file to:\n   {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        return False

if __name__ == "__main__":
    # Get input path from command line or prompt
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = input("Paste the full path to your LAS file: ").strip()
    
    # Handle quoted paths
    input_path = input_path.strip('"').strip("'")
    
    # Run the shift
    success = shift_building_points_up(input_path)
    
    if success:
        print("\n🎉 File is now CityForge-ready! Run check_las_info.py to verify.")
    else:
        print("\n❌ Script failed. Check errors above.")
        sys.exit(1)