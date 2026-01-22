# naturalize_and_subsample_auto.py
import os
import sys
import numpy as np
import laspy

# Auto-generate output directory relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "naturalized")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

def naturalize_and_subsample(input_path, working_path):
    """
    1. Subsample NIMBB to ~30k building points
    2. Add Perlin-like noise to break synthetic grid
    3. Auto-saves to thesis\outputs\naturalized folder
    """
    
    # Load files
    nimbb = laspy.read(input_path)
    working = laspy.read(working_path)
    
    # Get building masks
    nimbb_mask = nimbb.classification == 6
    work_mask = working.classification == 6
    
    nimbb_pts = np.vstack((nimbb.x[nimbb_mask], nimbb.y[nimbb_mask], nimbb.z[nimbb_mask])).T
    work_pts = np.vstack((working.x[work_mask], working.y[work_mask], working.z[work_mask])).T
    
    # **TARGET: 30,000 building points**
    target_count = 30000
    current_count = len(nimbb_pts)
    
    if current_count > target_count:
        # Intelligent subsampling (preserve high Z variance)
        z_gradient = np.abs(np.gradient(nimbb_pts[:, 2]))
        importance = z_gradient + np.random.random(len(nimbb_pts)) * 0.1
        
        keep_idx = np.argsort(importance)[-target_count:]
        keep_mask = np.zeros(current_count, dtype=bool)
        keep_mask[keep_idx] = True
        
        # Apply to full point cloud
        final_mask = np.where(nimbb_mask)[0][keep_mask]
        keep_global_mask = np.zeros(len(nimbb.points), dtype=bool)
        keep_global_mask[final_mask] = True
        
        # Keep ground points
        keep_global_mask |= nimbb.classification == 2
        
        # Create filtered LAS
        filtered = nimbb[keep_global_mask]
    else:
        filtered = nimbb
    
    # **Break the grid**: Add correlated Perlin-like noise
    building_indices = np.where(filtered.classification == 6)[0]
    n_building = len(building_indices)
    
    scale = 0.05  # 5cm noise
    freq = 0.1    # Low frequency for smoothness
    
    noise_x = np.sin(filtered.x[building_indices] * freq) * np.cos(filtered.y[building_indices] * freq) * scale
    noise_y = np.cos(filtered.x[building_indices] * freq) * np.sin(filtered.y[building_indices] * freq) * scale
    noise_z = np.sin(filtered.z[building_indices] * freq) * scale * 0.3
    
    filtered.x[building_indices] += noise_x
    filtered.y[building_indices] += noise_y
    filtered.z[building_indices] += noise_z
    
    # **Match XY entropy of working file**
    xy_rounded = np.round(np.vstack((filtered.x[building_indices], filtered.y[building_indices])).T * 10) / 10
    unique_xy = len(np.unique(xy_rounded, axis=0))
    entropy_ratio = len(building_indices) / unique_xy
    
    target_entropy_ratio = 1.4
    if entropy_ratio < target_entropy_ratio:
        additional_scale = (target_entropy_ratio / entropy_ratio) * 0.02
        filtered.x[building_indices] += np.random.uniform(-additional_scale, additional_scale, n_building)
        filtered.y[building_indices] += np.random.uniform(-additional_scale, additional_scale, n_building)
    
    # **AUTO-SAVE TO thesis\outputs\naturalized**
    input_basename = os.path.basename(input_path)
    name, ext = os.path.splitext(input_basename)
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{name}_natural{ext}")
    
    # Save
    filtered.write(output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ NATURALIZED & SAVED")
    print(f"{'='*60}")
    print(f"📁 Input:  {os.path.basename(input_path)}")
    print(f"📁 Output: {os.path.basename(output_path)}")
    print(f"💾 Saved to: {output_path}")
    print(f"{'-'*60}")
    print(f"Building points: {current_count:,} → {target_count:,}")
    
    # Verify entropy improvement
    xy_rounded_new = np.round(np.vstack((filtered.x[building_indices], filtered.y[building_indices])).T * 10) / 10
    unique_xy_new = len(np.unique(xy_rounded_new, axis=0))
    entropy_new = len(building_indices) / unique_xy_new
    print(f"Entropy ratio: {entropy_ratio:.3f} → {entropy_new:.3f} (target: ~1.4)")
    
    if entropy_new > 1.2:
        print("✓ Grid pattern sufficiently broken")
    else:
        print("⚠️  May need more noise adjustment")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python naturalize_and_subsample_auto.py \"failing.las\" \"working.las\"")
        print(f"\nOutput will be saved to: {DEFAULT_OUTPUT_DIR}")
        sys.exit(1)
    
    naturalize_and_subsample(sys.argv[1], sys.argv[2])