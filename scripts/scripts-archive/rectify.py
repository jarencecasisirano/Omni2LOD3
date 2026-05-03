import math
import numpy as np
from scipy.ndimage import map_coordinates

def mod(x, n):
    return ((x % n) + n) % n

def render_face(params):
    read_data    = params['data']
    face         = params['face']
    rotation     = params['rotation']
    interpolation= params['interpolation']
    max_width    = params.get('maxWidth', float('inf'))
    yaw_deg      = params.get('yaw', 0.0)
    yaw_rad      = math.radians(yaw_deg)
    # pitch: Y-axis rotation in degrees (positive = tilt up, negative = tilt down)
    pitch_deg    = params.get('pitch', 0.0)
    pitch_rad    = math.radians(pitch_deg)
    roll_deg     = params.get('roll', 0.0)
    roll_rad     = math.radians(roll_deg)

    read_width  = read_data['width']
    read_height = read_data['height']

    if len(read_data['data'].shape) == 1:
        img_data = read_data['data'].reshape((read_height, read_width, 4))
    else:
        img_data = read_data['data']

    face_width  = int(min(max_width, read_width / 4))
    face_height = face_width

    # Coordinate grids normalised to [-1, 1]
    xv, yv = np.meshgrid(np.arange(face_width), np.arange(face_height))
    nx = 2.0 * (xv + 0.5) / face_width  - 1.0
    ny = 2.0 * (yv + 0.5) / face_height - 1.0

    # Face -> 3-D direction vector
    if face == 'pz':
        cx, cy, cz = -np.ones_like(nx), -nx,            -ny
    elif face == 'nz':
        cx, cy, cz =  np.ones_like(nx),  nx,            -ny
    elif face == 'px':
        cx, cy, cz =  nx,               -np.ones_like(nx), -ny
    elif face == 'nx':
        cx, cy, cz = -nx,                np.ones_like(nx), -ny
    elif face == 'py':
        cx, cy, cz = -ny,               -nx,             np.ones_like(nx)
    elif face == 'ny':
        cx, cy, cz =  ny,               -nx,            -np.ones_like(nx)
    else:
        raise ValueError(f"Unknown face: {face}")

    # X-axis (roll) rotation
    if roll_rad != 0.0:
        cos_r = math.cos(roll_rad)
        sin_r = math.sin(roll_rad)
        cx, cy, cz = (cx,
                      cy * cos_r - cz * sin_r,
                      cy * sin_r + cz * cos_r)

    # Y-axis (pitch) rotation – rotates the camera up/down
    # cx' =  cx*cos - cz*sin  (wait, standard Ry: cx'=cx*cos+cz*sin, cz'=-cx*sin+cz*cos)
    # Rotation matrix around Y-axis:
    #   cx' =  cx * cos(pitch) + cz * sin(pitch)
    #   cy' =  cy
    #   cz' = -cx * sin(pitch) + cz * cos(pitch)
    if pitch_rad != 0.0:
        cos_p = math.cos(pitch_rad)
        sin_p = math.sin(pitch_rad)
        cx, cy, cz = (cx * cos_p + cz * sin_p,
                      cy,
                     -cx * sin_p + cz * cos_p)

    # Z-axis (yaw) rotation
    if yaw_rad != 0.0:
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        cx, cy, cz = (cx * cos_y - cy * sin_y,
                      cx * sin_y + cy * cos_y,
                      cz)

    # Project to spherical coordinates
    r   = np.sqrt(cx*cx + cy*cy + cz*cz)
    lon = mod(np.arctan2(cy, cx) + rotation, 2 * np.pi)
    lat = np.arccos(np.clip(cz / r, -1.0, 1.0))

    # Map to source pixel coordinates
    x_from = (read_width  * lon / (2 * np.pi) - 0.5).ravel()
    y_from = (read_height * lat / np.pi        - 0.5).ravel()

    # scipy.ndimage.map_coordinates interpolation order:
    #   0 = nearest, 1 = bilinear, 3 = bicubic spline (used for cubic & lanczos)
    order = {'linear': 1, 'cubic': 3, 'lanczos': 3}.get(interpolation, 0)

    coords = np.array([y_from, x_from])  # (row, col) order

    res = np.zeros((face_height, face_width, 4), dtype=np.uint8)
    for ch in range(3):
        sampled = map_coordinates(
            img_data[..., ch].astype(np.float32),
            coords,
            order=order,
            mode='nearest'   # clamp at borders; lon already wrapped via mod()
        )
        res[..., ch] = np.clip(sampled, 0, 255).reshape(face_height, face_width).astype(np.uint8)
    res[..., 3] = 255

    return {'width': face_width, 'height': face_height, 'data': res}


if __name__ == "__main__":
    print("Rectify script loaded (scipy fast path).")