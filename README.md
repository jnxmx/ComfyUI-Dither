# ComfyUI-Dither

A custom node pack for ComfyUI that applies high-quality image dithering (Atkinson and Halftone), including channel-split dithering for advanced RGB/CMYK printing effects.

## Nodes Included
1. **Dither**: A simple node that applies dithering to the input image or batch.
   - Adjust `threshold`, `contrast`, `brightness`, `gamma`, and `grain` before dithering.
   - Choose between `Atkinson` error diffusion or `Halftone` grid patterns.
   - Set custom `color1` and `color2` (HEX strings) to map the output colors.
   - `scale`: Resize the image during processing (e.g., `2.0` for 200% resolution) to create high-res dithered outputs.

2. **Dither by channel**: An advanced node that splits the image into `RGB` or `CMYK` channels, applies dithering to each channel separately, and blends them back together.
   - Allows independent settings (method, threshold, contrast, scale, colors) for each channel.
   - Outputs each processed channel individually (`CH1`, `CH2`, `CH3`, `CH4`) and an `OVERLAY` composed of all channels blended together.
   - Choose `blend_mode` for the `OVERLAY` output (multiply, screen, overlay, add, darken, lighten).

## Installation
Clone this repository into your `ComfyUI/custom_nodes/` directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/YOUR_GITHUB_USERNAME/ComfyUI-Dither.git
```

Install the requirements:
```bash
pip install -r ComfyUI-Dither/requirements.txt
```

*(Note: Numba is highly recommended as it speeds up the Atkinson dithering process by 100x)*
