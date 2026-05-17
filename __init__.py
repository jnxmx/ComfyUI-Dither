from .dither_nodes import DitherImage, DitherByChannel

NODE_CLASS_MAPPINGS = {
    "DitherImage": DitherImage,
    "DitherByChannel": DitherByChannel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DitherImage": "Dither",
    "DitherByChannel": "Dither by channel"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
