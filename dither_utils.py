import numpy as np
import cv2

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not found. Dithering processing will be significantly slower.")

def fast_jit(func):
    if HAS_NUMBA:
        return jit(nopython=True)(func)
    return func

@fast_jit
def apply_atkinson_core(img_float, threshold):
    h, w = img_float.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            old_val = img_float[y, x]
            new_val = 255.0 if old_val >= threshold else 0.0
            out[y, x] = int(new_val)
            err = (old_val - new_val) / 8.0
            if x + 1 < w: img_float[y, x + 1] += err
            if x + 2 < w: img_float[y, x + 2] += err
            if y + 1 < h and x - 1 >= 0: img_float[y + 1, x - 1] += err
            if y + 1 < h: img_float[y + 1, x] += err
            if y + 1 < h and x + 1 < w: img_float[y + 1, x + 1] += err
            if y + 2 < h: img_float[y + 2, x] += err
    return out

def apply_halftone(img, dot_size, threshold):
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    cy = (y // dot_size) * dot_size + dot_size // 2
    cx = (x // dot_size) * dot_size + dot_size // 2
    dist_sq = (y - cy)**2 + (x - cx)**2
    max_dist_sq = (dot_size / 2.0)**2
    threshold_map = dist_sq / max_dist_sq
    darkness = 1.0 - (img / 255.0)
    sensitivity = 1.0 + (threshold / 128.0)
    darkness = np.clip(darkness * sensitivity, 0, 1)
    out = np.ones((h, w), dtype=np.uint8) * 255
    out[darkness > threshold_map] = 0
    return out

def apply_adjustments(img, brightness, contrast, gamma, grain):
    img = img * (brightness / 100.0)
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    img = factor * (img - 128) + 128
    if gamma != 1.0 and gamma > 0:
        img = np.power(np.maximum(img, 0) / 255.0, 1.0 / gamma) * 255.0
    if grain > 0:
        noise = (np.random.rand(*img.shape) - 0.5) * grain * 2.55
        img += noise
    return np.clip(img, 0, 255)

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3: hex_str = ''.join([c*2 for c in hex_str])
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
