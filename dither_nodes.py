import torch
import numpy as np
import cv2
from .dither_utils import apply_adjustments, apply_atkinson_core, apply_halftone, hex_to_rgb

def process_channel(gray_img, params, as_rgba=False):
    nh, nw = int(gray_img.shape[0] * params['scale']), int(gray_img.shape[1] * params['scale'])
    if params['scale'] != 1.0:
        gray_img = cv2.resize(gray_img, (nw, nh), interpolation=cv2.INTER_AREA)
        
    adjusted = apply_adjustments(gray_img.astype(np.float32), params['brightness'], params['contrast'], params['gamma'], params['grain'])
    
    if params['method'] == "atkinson":
        processed = apply_atkinson_core(adjusted, params['threshold'])
    else:
        processed = apply_halftone(adjusted, params['dot_size'], params['threshold'])
        
    c1 = np.array(hex_to_rgb(params['color1']), dtype=np.uint8)
    c2 = np.array(hex_to_rgb(params['color2']), dtype=np.uint8)
    
    if as_rgba:
        # "channel color to alpha"
        # All pixels have RGB = color2 (channel color)
        # Alpha is opaque where processed == 0 (dots), transparent where processed == 255 (background)
        res = np.empty((nh, nw, 4), dtype=np.uint8)
        res[..., :3] = c2
        res[..., 3] = 255 - processed
    else:
        res = np.empty((nh, nw, 3), dtype=np.uint8)
        mask = processed == 255
        res[mask] = c2
        res[~mask] = c1
    return res

def blend_images(base, overlay, mode):
    # Perform alpha blending with Porter-Duff compositing and standard blend modes
    base = base.astype(np.float32) / 255.0
    overlay = overlay.astype(np.float32) / 255.0
    
    c_dst, a_dst = base[..., :3], base[..., 3:4]
    c_src, a_src = overlay[..., :3], overlay[..., 3:4]
    
    if mode == "multiply":
        f = c_dst * c_src
    elif mode == "screen":
        f = 1.0 - (1.0 - c_dst) * (1.0 - c_src)
    elif mode == "overlay":
        mask = c_dst < 0.5
        f = np.zeros_like(c_dst)
        f[mask] = 2.0 * c_dst[mask] * c_src[mask]
        f[~mask] = 1.0 - 2.0 * (1.0 - c_dst[~mask]) * (1.0 - c_src[~mask])
    elif mode == "add":
        f = np.clip(c_dst + c_src, 0.0, 1.0)
    elif mode == "darken":
        f = np.minimum(c_dst, c_src)
    elif mode == "lighten":
        f = np.maximum(c_dst, c_src)
    else:
        f = c_src
        
    a_res = a_src + a_dst * (1.0 - a_src)
    safe_a_res = np.where(a_res > 0.0, a_res, 1.0)
    
    c_res = (c_src * a_src * (1.0 - a_dst) + 
             c_dst * a_dst * (1.0 - a_src) + 
             a_src * a_dst * f) / safe_a_res
             
    res = np.empty_like(base)
    res[..., :3] = c_res
    res[..., 3:4] = a_res
    return (res * 255.0).astype(np.uint8)

def extract_rgb(img):
    r = (img[..., 0] * 255).astype(np.uint8)
    g = (img[..., 1] * 255).astype(np.uint8)
    b = (img[..., 2] * 255).astype(np.uint8)
    return r, g, b

def rgb_to_cmyk(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    k = 1.0 - np.max(img, axis=-1)
    k_mask = k < 1.0
    c = np.zeros_like(k)
    m = np.zeros_like(k)
    y = np.zeros_like(k)
    
    c[k_mask] = (1.0 - r[k_mask] - k[k_mask]) / (1.0 - k[k_mask])
    m[k_mask] = (1.0 - g[k_mask] - k[k_mask]) / (1.0 - k[k_mask])
    y[k_mask] = (1.0 - b[k_mask] - k[k_mask]) / (1.0 - k[k_mask])
    
    return (c*255).astype(np.uint8), (m*255).astype(np.uint8), (y*255).astype(np.uint8), (k*255).astype(np.uint8)

class DitherImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "method": (["atkinson", "halftone"],),
            "threshold": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 255.0, "step": 1.0}),
            "contrast": ("FLOAT", {"default": 150.0, "min": 0.0, "max": 200.0, "step": 1.0}),
            "brightness": ("FLOAT", {"default": 65.0, "min": 0.0, "max": 200.0, "step": 1.0}),
            "gamma": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            "grain": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            "dot_size": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1}),
            "color1": ("STRING", {"default": "#000000"}),
            "color2": ("STRING", {"default": "#FFFFFF"}),
            "scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/dither"

    def process(self, image, method, threshold, contrast, brightness, gamma, grain, dot_size, color1, color2, scale):
        b, h, w, c = image.shape
        
        output = []
        for i in range(b):
            img_np = image[i].cpu().numpy()
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            elif img_np.shape[-1] == 4:
                img_np = img_np[..., :3]
                
            img_uint8 = (img_np * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            params = {
                'method': method,
                'threshold': threshold,
                'contrast': contrast,
                'brightness': brightness,
                'gamma': gamma,
                'grain': grain,
                'dot_size': dot_size,
                'color1': color1,
                'color2': color2,
                'scale': scale
            }
            res = process_channel(gray, params)
            output.append(res.astype(np.float32) / 255.0)
            
        return (torch.from_numpy(np.stack(output)),)

def create_channel_inputs(prefix, default_color1="#000000", default_color2="#FFFFFF"):
    return {
        f"{prefix}_method": (["atkinson", "halftone"],),
        f"{prefix}_threshold": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 255.0, "step": 1.0}),
        f"{prefix}_contrast": ("FLOAT", {"default": 150.0, "min": 0.0, "max": 200.0, "step": 1.0}),
        f"{prefix}_brightness": ("FLOAT", {"default": 65.0, "min": 0.0, "max": 200.0, "step": 1.0}),
        f"{prefix}_gamma": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
        f"{prefix}_grain": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 1.0}),
        f"{prefix}_dot_size": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1}),
        f"{prefix}_color1": ("STRING", {"default": default_color1}),
        f"{prefix}_color2": ("STRING", {"default": default_color2}),
        f"{prefix}_scale": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
    }

class DitherByChannel:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "image": ("IMAGE",),
            "color_space": (["RGB", "CMYK"],),
            "blend_mode": (["multiply", "screen", "overlay", "add", "darken", "lighten"],)
        }
        # Reasonable defaults for RGB or CMYK blending
        inputs.update(create_channel_inputs("ch1", default_color1="#000000", default_color2="#FF0000"))
        inputs.update(create_channel_inputs("ch2", default_color1="#000000", default_color2="#00FF00"))
        inputs.update(create_channel_inputs("ch3", default_color1="#000000", default_color2="#0000FF"))
        inputs.update(create_channel_inputs("ch4", default_color1="#000000", default_color2="#FFFFFF"))
        return {"required": inputs}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("CH1", "CH2", "CH3", "CH4", "OVERLAY")
    FUNCTION = "process"
    CATEGORY = "image/dither"

    def process(self, image, color_space, blend_mode, **kwargs):
        b, h, w, c = image.shape
        
        out_ch1, out_ch2, out_ch3, out_ch4, out_overlay = [], [], [], [], []
        
        for i in range(b):
            img_np = image[i].cpu().numpy()
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            elif img_np.shape[-1] == 4:
                img_np = img_np[..., :3]
                
            if color_space == "RGB":
                ch1, ch2, ch3 = extract_rgb(img_np)
                ch4 = None
            else:
                ch1, ch2, ch3, ch4 = rgb_to_cmyk(img_np)
                
            params = [{}, {}, {}, {}]
            for ch_idx in range(4):
                prefix = f"ch{ch_idx+1}"
                params[ch_idx] = {
                    'method': kwargs[f'{prefix}_method'],
                    'threshold': kwargs[f'{prefix}_threshold'],
                    'contrast': kwargs[f'{prefix}_contrast'],
                    'brightness': kwargs[f'{prefix}_brightness'],
                    'gamma': kwargs[f'{prefix}_gamma'],
                    'grain': kwargs[f'{prefix}_grain'],
                    'dot_size': kwargs[f'{prefix}_dot_size'],
                    'color1': kwargs[f'{prefix}_color1'],
                    'color2': kwargs[f'{prefix}_color2'],
                    'scale': kwargs[f'{prefix}_scale']
                }
                
            res1 = process_channel(ch1, params[0], as_rgba=True)
            res2 = process_channel(ch2, params[1], as_rgba=True)
            res3 = process_channel(ch3, params[2], as_rgba=True)
            
            if ch4 is not None:
                res4 = process_channel(ch4, params[3], as_rgba=True)
            else:
                res4 = np.zeros((res1.shape[0], res1.shape[1], 4), dtype=np.uint8)
                
            channels = [res1, res2, res3]
            if ch4 is not None:
                channels.append(res4)
                
            max_h = max([c.shape[0] for c in channels])
            max_w = max([c.shape[1] for c in channels])
            
            resized_channels = []
            for ch_img in channels:
                if ch_img.shape[0] != max_h or ch_img.shape[1] != max_w:
                    resized = cv2.resize(ch_img, (max_w, max_h), interpolation=cv2.INTER_NEAREST)
                else:
                    resized = ch_img
                resized_channels.append(resized)
                
            merged = resized_channels[0]
            for idx in range(1, len(resized_channels)):
                merged = blend_images(merged, resized_channels[idx], blend_mode)
                
            out_ch1.append(res1.astype(np.float32) / 255.0)
            out_ch2.append(res2.astype(np.float32) / 255.0)
            out_ch3.append(res3.astype(np.float32) / 255.0)
            out_ch4.append(res4.astype(np.float32) / 255.0)
            out_overlay.append(merged.astype(np.float32) / 255.0)
            
        return (
            torch.from_numpy(np.stack(out_ch1)),
            torch.from_numpy(np.stack(out_ch2)),
            torch.from_numpy(np.stack(out_ch3)),
            torch.from_numpy(np.stack(out_ch4)),
            torch.from_numpy(np.stack(out_overlay))
        )
