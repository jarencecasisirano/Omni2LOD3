import os
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
    Converts 4 concatenated rectilinear images (from Google Street View API) 
    into a mathematically correct equirectangular (spherical) panorama.
    Similar to how PTGui stitches perspective images onto a sphere.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(input_dir) if f.casefold().endswith(valid_extensions)]
    
    if not images:
        print(f"No images found in {input_dir}.")
        return

    print(f"Found {len(images)} images to process. Using OpenCV: {HAS_CV2}")
    
    # Pre-calculate spherical coordinate maps for re-projection
    print("Pre-calculating rectilinear-to-equirectangular projection maps...")
    
    # Create the coordinate grid for the output equirectangular image
    x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))
    
    # Map X to longitude (yaw). Center of the image (x=out_w/2) will be North (yaw=0).
    yaw = (x / out_w) * 2 * np.pi - np.pi
    
    # Map Y to latitude (pitch). Center of image is horizon (pitch=0).
    pitch = np.pi/2 - (y / (out_h - 1)) * np.pi
    
    # Determine the corresponding rectilinear camera face (0, 1, 2, 3) 
    # based on the yaw angle. Each camera covers 90 degrees (pi/2).
    c = np.floor((yaw + np.pi/4) / (np.pi/2)).astype(int) % 4
    cam_angles = c * (np.pi / 2)
    
    # Relative yaw angle from the center of the selected camera
    delta_yaw = yaw - cam_angles
    # Wrap to [-pi, pi] to handle angles correctly
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    
    # Project spherical coordinates onto the rectilinear projection plane (Z=1)
    u = np.tan(delta_yaw)
    cos_dy = np.cos(delta_yaw)
    cos_dy[cos_dy == 0] = 1e-5 # Avoid division by zero
    v = np.tan(pitch) / cos_dy
    
    # Valid pixels mask (where vertical FOV is within the perspective camera bounds)
    mask = np.abs(v) <= 1.0
    
    H_face = 640 # Assuming height of typical GSV query is 640
    W_face = 640 # Assuming width per face is 640
    
    # Pixel coordinates on the face
    px_u = (u + 1) * 0.5 * W_face
    px_v = (1 - v) * 0.5 * H_face
    
    # Map to the corresponding face segment in the concatenated image strip
    map_x = px_u + c * W_face
    map_y = px_v
    
    # Clip bounds to prevent out-of-index errors
    map_x = np.clip(map_x, 0, (4 * W_face) - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, H_face - 1).astype(np.float32)

    for filename in images:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            with Image.open(input_path) as img:
                img_pano = np.array(img.convert('RGB'))
                
            # Allow dynamic face sizes depending on the image downloaded
            actual_H_face = img_pano.shape[0]
            actual_W_face = img_pano.shape[1] // 4
            
            # Recalculate mappings if the image dimensions differ from 640x640 default
            if actual_H_face != H_face or actual_W_face != W_face:
                cur_px_u = (u + 1) * 0.5 * actual_W_face
                cur_px_v = (1 - v) * 0.5 * actual_H_face
                cur_map_x = np.clip(cur_px_u + c * actual_W_face, 0, img_pano.shape[1] - 1).astype(np.float32)
                cur_map_y = np.clip(cur_px_v, 0, img_pano.shape[0] - 1).astype(np.float32)
            else:
                cur_map_x, cur_map_y = map_x, map_y
            
            if HAS_CV2:
                spherical = cv2.remap(img_pano, cur_map_x, cur_map_y, interpolation=cv2.INTER_CUBIC, 
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                spherical[~mask] = 0
            else:
                coords = np.array([cur_map_y.ravel(), cur_map_x.ravel()])
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
                spherical[~mask] = 0
            
            spherical_img = Image.fromarray(spherical, 'RGB')
            spherical_img.save(output_path, quality=95)
            print(f"  -> Re-projected and saved: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"\nAll equirectangular spherical models saved successfully to: {output_dir}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    input_dir = os.path.join(project_root, "data", "google-street-view")
    output_dir = os.path.join(project_root, "data", "google-street-view-spherical")
    
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print("-" * 50)
    
    convert_to_spherical(input_dir, output_dir, out_w=2560, out_h=1280)

if __name__ == "__main__":
    main()
