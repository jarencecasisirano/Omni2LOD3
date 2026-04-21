import os
import numpy as np
from PIL import Image
from rectify import render_face

def process_images(input_dir, output_dir, pitch=0.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    
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
                'maxWidth': 2048, # Adjust as needed
                'pitch': pitch
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
    # Determine project root (assuming script is in scripts/LOD2toLOD3)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    base_input_dir = os.path.join(project_root, "data", "raw_images", "google-street-view3")
    base_output_dir = os.path.join(project_root, "outputs", "01_cubemaps")
    
    if not os.path.exists(base_input_dir):
        print(f"Error: Input directory not found: {base_input_dir}")
        exit(1)
        
    # List subdirectories
    subdirs = [d for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))]
    subdirs.sort()
    
    if not subdirs:
        print(f"No subdirectories found in {base_input_dir}")
        exit(1)
        
    print("Available folders to process:")
    for i, d in enumerate(subdirs):
        print(f"{i + 1}: {d}")
        
    while True:
        try:
            selection = input("\nEnter the number of the folder you want to process: ")
            index = int(selection) - 1
            if 0 <= index < len(subdirs):
                selected_subdir = subdirs[index]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    INPUT_DIR = os.path.join(base_input_dir, selected_subdir)
    OUTPUT_DIR = os.path.join(base_output_dir, selected_subdir)
    
    # Prompt for X-axis (pitch) rotation
    while True:
        try:
            pitch_input = input("\nEnter the X-axis camera pitch angle in degrees (positive = tilt up, negative = tilt down) [default 0]: ").strip()
            if pitch_input == '':
                pitch_angle = 0.0
            else:
                pitch_angle = float(pitch_input)
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value (e.g. 15, -10, 0).")
    
    print(f"\nProcessing folder: {selected_subdir}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Pitch:  {pitch_angle} degrees")
    
    process_images(INPUT_DIR, OUTPUT_DIR, pitch=pitch_angle)
