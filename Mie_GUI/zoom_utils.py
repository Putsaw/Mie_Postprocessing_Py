"""Helpers for simple integer-based zoom operations."""

from PIL import Image
import numpy as np
from cv2 import resize, INTER_NEAREST


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
    # arr = arr.repeat(factor, axis=0).repeat(factor, axis=1)
    arr = resize(arr, None, fx=factor, fy=factor, interpolation=INTER_NEAREST)

    return Image.fromarray(arr, mode=img.mode)