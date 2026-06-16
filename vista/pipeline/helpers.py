"""
Helper functions for the VISTA pipeline.
"""
from typing import Tuple
from PIL import Image

def crop_track_with_padding(
    image: Image.Image, 
    bbox: Tuple[float, float, float, float], 
    padding_factor: float = 0.2
) -> Image.Image:
    """
    Crops a bounding box from an image, adding a padding margin.
    Essential for providing spatial context to Action Recognition/Captioning models.
    
    Args:
        image: The original image in PIL Image format.
        bbox: Tuple with coordinates (x1, y1, x2, y2).
        padding_factor: Percentage of padding to add (default 20%).
        
    Returns:
        The cropped image.
    """
    img_width, img_height = image.size
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    # Calculate padding in pixels
    pad_x = w * padding_factor
    pad_y = h * padding_factor

    # Apply the padding ensuring we don't go outside the frame boundaries
    new_x1 = max(0, int(x1 - pad_x))
    new_y1 = max(0, int(y1 - pad_y))
    new_x2 = min(img_width, int(x2 + pad_x))
    new_y2 = min(img_height, int(y2 + pad_y))
