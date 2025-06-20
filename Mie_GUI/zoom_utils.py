"""Helpers for simple integer-based zoom operations."""

from PIL import Image
import numpy as np


def enlarge_image(img: Image.Image, factor: int) -> Image.Image:
    """Return ``img`` scaled by ``factor`` using nearest-neighbor tiling.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image to scale.
    factor : int
        Integer zoom factor. Values less than or equal to 1 return the original
        image.
    """
    if factor <= 1:
        return img
    arr = np.array(img)
    # Repeat pixels along both axes to create a tiled effect
    arr = arr.repeat(factor, axis=0).repeat(factor, axis=1)
    return Image.fromarray(arr, mode=img.mode)