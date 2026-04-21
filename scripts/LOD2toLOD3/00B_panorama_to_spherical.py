import os
import re
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from scipy.ndimage import map_coordinates

def convert_to_spherical(input_dir, output_dir, out_w=2560, out_h=1280):
    """
    Converts 6 separate tiles (N, E, S, W, Zenith, Nadir) from Google Street View
    into a mathematically correct equirectangular (spherical) panorama.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    locations = set()
    for f in os.listdir(input_dir):
        match = re.match(r"(tile_loc_\d+_[0-9\.\-]+_[0-9\.\-]+)_[a-z]+\.jpg", f, re.IGNORECASE)
        if match:
            locations.add(match.group(1))
            
    if not locations:
        print(f"No valid tile sets found in {input_dir}.")
        return

    print(f"Found {len(locations)} locations to process. Using OpenCV: {HAS_CV2}")
    
    # Pre-calculate spherical coordinate maps for re-projection
    print("Pre-calculating rectilinear-to-equirectangular projection maps...")
    
    x_px, y_px = np.meshgrid(np.arange(out_w), np.arange(out_h))
    yaw = (x_px / out_w) * 2 * np.pi - np.pi
    pitch = np.pi/2 - (y_px / (out_h - 1)) * np.pi
    
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)
    
    max_abs = np.maximum(np.abs(x), np.maximum(np.abs(y), np.abs(z)))
    
    is_front = (x == max_abs)
    is_back  = (x == -max_abs) & ~is_front
    is_right = (y == max_abs) & ~(is_front | is_back)
    is_left  = (y == -max_abs) & ~(is_front | is_back | is_right)
    is_zenith= (z == max_abs) & ~(is_front | is_back | is_right | is_left)
    is_nadir = (z == -max_abs) & ~(is_front | is_back | is_right | is_left | is_zenith)
    
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    face_idx = np.zeros(x.shape, dtype=int)
    
    u[is_front]  = y[is_front] / x[is_front]
    v[is_front]  = -z[is_front] / x[is_front]
    face_idx[is_front] = 0
    
    u[is_right]  = -x[is_right] / y[is_right]
    v[is_right]  = -z[is_right] / y[is_right]
    face_idx[is_right] = 1
    
    u[is_back]   = y[is_back] / x[is_back]
    v[is_back]   = z[is_back] / x[is_back]
    face_idx[is_back] = 2
    
    u[is_left]   = -x[is_left] / y[is_left]
    v[is_left]   = z[is_left] / y[is_left]
    face_idx[is_left] = 3
    
    u[is_zenith] = y[is_zenith] / z[is_zenith]
    v[is_zenith] = x[is_zenith] / z[is_zenith]
    face_idx[is_zenith] = 4
    
    u[is_nadir]  = -y[is_nadir] / z[is_nadir]
    v[is_nadir]  = x[is_nadir] / z[is_nadir]
    face_idx[is_nadir] = 5

    face_names = ['front', 'right', 'back', 'left', 'zenith', 'nadir']

    for loc_prefix in sorted(locations):
        faces = []
        skip = False
        for name in face_names:
            input_path = os.path.join(input_dir, f"{loc_prefix}_{name}.jpg")
            try:
                with Image.open(input_path) as img:
                    faces.append(np.array(img.convert('RGB')))
            except Exception as e:
                print(f"Skipping {loc_prefix}, error loading {name}: {e}")
                skip = True
                break
                
        if skip:
            continue
            
        try:
            H_face, W_face = faces[0].shape[:2]
            img_pano = np.vstack(faces) # shape (6*H_face, W_face, 3)
            
            px_u = (u + 1) * 0.5 * W_face
            px_v = (v + 1) * 0.5 * H_face
            
            map_x = np.clip(px_u, 0, W_face - 1).astype(np.float32)
            map_y = np.clip(px_v + face_idx * H_face, 0, 6 * H_face - 1).astype(np.float32)
            
            output_filename = f"{loc_prefix}_pano.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            if HAS_CV2:
                spherical = cv2.remap(img_pano, map_x, map_y, interpolation=cv2.INTER_CUBIC, 
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            else:
                coords = np.array([map_y.ravel(), map_x.ravel()])
                spherical = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                for ch in range(3):
                    sampled = map_coordinates(
                        img_pano[..., ch].astype(np.float32),
                        coords,
                        order=1, # Bilinear
                        mode='constant',
                        cval=0.0
                    )
                    spherical[..., ch] = sampled.reshape(out_h, out_w).astype(np.uint8)
            
            spherical_img = Image.fromarray(spherical, 'RGB')
            spherical_img.save(output_path, quality=95)
            print(f"  -> Re-projected and saved: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {loc_prefix}: {e}")
            
    print(f"\nAll equirectangular spherical models saved successfully to: {output_dir}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    input_dir = os.path.join(project_root, "data", "google-street-view4")
    output_dir = os.path.join(project_root, "data", "google-street-view-spherical-4")
    
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print("-" * 50)
    
    convert_to_spherical(input_dir, output_dir, out_w=2560, out_h=1280)

if __name__ == "__main__":
    main()
