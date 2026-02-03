import os
import numpy as np
from PIL import Image
from rectify import render_face

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    faces = ['pz', 'nz', 'px', 'nx', 'py', 'ny']
    
    for img_name in images:
        print(f"Processing {img_name}...")
        img_path = os.path.join(input_dir, img_name)
        
        # Load image and convert to RGBA
        with Image.open(img_path) as img:
            img = img.convert('RGBA')
            width, height = img.size
            img_array = np.array(img)
        
        read_data = {
            'width': width,
            'height': height,
            'data': img_array
        }
        
        base_name = os.path.splitext(img_name)[0]
        
        for face in faces:
            print(f"  Rendering face: {face}")
            params = {
                'data': read_data,
                'face': face,
                'rotation': 0,
                'interpolation': 'lanczos',
                'maxWidth': 2048 # Adjust as needed
            }
            
            result = render_face(params)
            
            # Save result
            face_img = Image.fromarray(result['data'], 'RGBA')
            # Convert to RGB for saving as JPG
            face_img = face_img.convert('RGB')
            
            output_name = f"{base_name}_{face}.jpg"
            output_path = os.path.join(output_dir, output_name)
            face_img.save(output_path, quality=95)
            print(f"  Saved {output_name}")

if __name__ == "__main__":
    INPUT_DIR = r"c:\Users\Sky Torneros\Omni2LOD3\data\raw_images"
    OUTPUT_DIR = r"c:\Users\Sky Torneros\Omni2LOD3\outputs\01_cubemaps"
    
    process_images(INPUT_DIR, OUTPUT_DIR)
