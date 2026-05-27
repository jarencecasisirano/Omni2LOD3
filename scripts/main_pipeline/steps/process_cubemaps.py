import os
import numpy as np
from PIL import Image
from utils.rectify import render_face

FACES = ['pz', 'nz', 'px', 'nx', 'py', 'ny']


def process_images(input_dir, output_dir, yaw=0.0, pitch=0.0, roll=0.0):
    os.makedirs(output_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for img_name in images:
        print(f"Processing {img_name}...")

        img_path = os.path.join(input_dir, img_name)

        with Image.open(img_path) as img:
            img = img.convert('RGBA')
            img_array = np.array(img)
            width, height = img.size

        read_data = {
            'width': width,
            'height': height,
            'data': img_array
        }

        base_name = os.path.splitext(img_name)[0]

        for face in FACES:
            params = {
                'data': read_data,
                'face': face,
                'rotation': 0,
                'interpolation': 'lanczos',
                'maxWidth': 2048,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll
            }

            result = render_face(params)

            face_img = Image.fromarray(result['data'], 'RGBA').convert('RGB')

            output_name = f"{base_name}_{face}.jpg"
            output_path = os.path.join(output_dir, output_name)

            face_img.save(output_path, quality=95)


def run(input_dir, output_dir, config=None):
    """
    Pipeline-safe wrapper for cubemap generation.
    """

    if input_dir is None:
        raise ValueError("input_dir is required")

    if output_dir is None:
        raise ValueError("output_dir is required")

    config = config or {}

    yaw = float(config.get("yaw", 0.0))
    pitch = float(config.get("pitch", 0.0))
    roll = float(config.get("roll", 0.0))

    print("STEP 04: Cubemap Processing")

    process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        yaw=yaw,
        pitch=pitch,
        roll=roll
    )

    return {
        "output": output_dir,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll
    }