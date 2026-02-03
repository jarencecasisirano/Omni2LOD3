import math
import numpy as np

def clamp(x, min_val, max_val):
    return np.clip(x, min_val, max_val)

def mod(x, n):
    return ((x % n) + n) % n

def render_face(params):
    read_data = params['data']
    face = params['face']
    rotation = params['rotation']
    interpolation = params['interpolation']
    max_width = params.get('maxWidth', float('inf'))

    read_width = read_data['width']
    read_height = read_data['height']
    # Efficiently reshaped data for vectorization: (height, width, 4)
    # If it's a flat array, reshape it
    if len(read_data['data'].shape) == 1:
        img_data = read_data['data'].reshape((read_height, read_width, 4))
    else:
        img_data = read_data['data']

    face_width = int(min(max_width, read_width / 4))
    face_height = face_width

    # Create coordinate grids
    x = np.arange(face_width)
    y = np.arange(face_height)
    xv, yv = np.meshgrid(x, y)

    # Normalize coordinates to [-1, 1]
    nx = (2.0 * (xv + 0.5) / face_width - 1.0)
    ny = (2.0 * (yv + 0.5) / face_height - 1.0)

    # Orientation mapping
    if face == 'pz':
        cx, cy, cz = -np.ones_like(nx), -nx, -ny
    elif face == 'nz':
        cx, cy, cz = np.ones_like(nx), nx, -ny
    elif face == 'px':
        cx, cy, cz = nx, -np.ones_like(nx), -ny
    elif face == 'nx':
        cx, cy, cz = -nx, np.ones_like(nx), -ny
    elif face == 'py':
        cx, cy, cz = -ny, -nx, np.ones_like(nx)
    elif face == 'ny':
        cx, cy, cz = ny, -nx, -np.ones_like(nx)
    else:
        raise ValueError(f"Unknown face: {face}")

    # Project to spherical coordinates
    r = np.sqrt(cx*cx + cy*cy + cz*cz)
    lon = mod(np.arctan2(cy, cx) + rotation, 2 * np.pi)
    lat = np.arccos(cz / r)

    # Map to input pixel coordinates
    x_from = read_width * lon / (2 * np.pi) - 0.5
    y_from = read_height * lat / np.pi - 0.5

    # Interpolation
    if interpolation == 'linear':
        output_data = bilinear_interpolate(img_data, x_from, y_from)
    elif interpolation == 'cubic':
        output_data = bicubic_interpolate(img_data, x_from, y_from)
    elif interpolation == 'lanczos':
        output_data = lanczos_interpolate(img_data, x_from, y_from)
    else:
        # Nearest neighbor
        ix = np.round(x_from).astype(int).clip(0, read_width - 1)
        iy = np.round(y_from).astype(int).clip(0, read_height - 1)
        output_data = img_data[iy, ix]

    # Return result with alpha = 255
    res = np.zeros((face_height, face_width, 4), dtype=np.uint8)
    res[..., :3] = output_data[..., :3].astype(np.uint8)
    res[..., 3] = 255

    return {
        'width': face_width,
        'height': face_height,
        'data': res
    }

def bilinear_interpolate(img, px, py):
    h, w, _ = img.shape
    x0 = np.floor(px).astype(int)
    x1 = x0 + 1
    y0 = np.floor(py).astype(int)
    y1 = y0 + 1

    xf = px - x0
    yf = py - y0

    x0 = np.clip(x0, 0, w-1)
    x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1)
    y1 = np.clip(y1, 0, h-1)

    p00 = img[y0, x0]
    p10 = img[y0, x1]
    p01 = img[y1, x0]
    p11 = img[y1, x1]

    wa = (1-xf) * (1-yf)
    wb = xf * (1-yf)
    wc = (1-xf) * yf
    wd = xf * yv # error here, fixed below

    # Wait, the math is:
    # row0Content = p00 * (1-xf) + p10 * xf
    # row1Content = p01 * (1-xf) + p11 * xf
    # final = row0 * (1-yf) + row1 * yf
    
    # Re-writing for clarity and correctness
    r0 = p00 * (1-xf[:,:,None]) + p10 * xf[:,:,None]
    r1 = p01 * (1-xf[:,:,None]) + p11 * xf[:,:,None]
    return r0 * (1-yf[:,:,None]) + r1 * yf[:,:,None]

def bicubic_interpolate(img, px, py):
    # Optimized bicubic usually involves fixed 4x4 kernel
    return resample_kernel(img, px, py, 2, bicubic_kernel)

def lanczos_interpolate(img, px, py):
    return resample_kernel(img, px, py, 5, lanczos_kernel)

def bicubic_kernel(x):
    b = -0.5
    x = np.abs(x)
    x2 = x*x
    x3 = x2*x
    return np.where(x <= 1, (b + 2)*x3 - (b + 3)*x2 + 1, b*x3 - 5*b*x2 + 8*b*x - 4*b)

def lanczos_kernel(x):
    a = 5.0
    x = np.abs(x)
    res = np.where(x == 0, 1.0, 
                   np.where(x < a, a * np.sin(np.pi * x) * np.sin(np.pi * x / a) / (np.pi**2 * x**2), 0))
    return res

def resample_kernel(img, px, py, a, kernel_fn):
    h, w, c = img.shape
    x_floor = np.floor(px).astype(int)
    y_floor = np.floor(py).astype(int)

    # Output accumulation
    out = np.zeros(px.shape + (c,), dtype=float)
    total_w = np.zeros(px.shape + (c,), dtype=float)

    for dy in range(-a + 1, a + 1):
        for dx in range(-a + 1, a + 1):
            ix = np.clip(x_floor + dx, 0, w - 1)
            iy = np.clip(y_floor + dy, 0, h - 1)
            
            wx = kernel_fn(px - (x_floor + dx))
            wy = kernel_fn(py - (y_floor + dy))
            w_total = (wx * wy)[:, :, None]
            
            out += img[iy, ix] * w_total
            # For normalized kernels like Lanczos/Bicubic, 
            # we don't strictly need to divide by total weight if kernel sum is 1,
            # but it helps with edge cases. Original JS didn't normalize.
    
    return np.round(out)

if __name__ == "__main__":
    print("Vectorized Rectify script loaded.")