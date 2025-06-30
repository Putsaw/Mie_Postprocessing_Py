import numpy as np
import cv2
from functools import lru_cache
from typing import Optional, Tuple

CropRect = Tuple[int, int, int, int]  # x, y, w, h

@lru_cache(maxsize=8)
def make_rotation_maps(
    frame_size: Tuple[int, int],
    angle: float,
    crop_rect: Optional[CropRect] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute map_x, map_y for cv2.remap.

    Args:
      frame_size: (h, w) of the source frames.
      angle:      Rotation angle in degrees (positive = CCW).
      crop_rect:  (x, y, w, h) of the desired crop in the rotated image;
                  if None, uses full (w, h).

    Returns:
      map_x, map_y arrays of shape (out_h, out_w), dtype=float32.
    """
    h, w = frame_size
    # Output crop defaults to full frame
    if crop_rect is None:
        x0, y0, out_w, out_h = 0, 0, w, h
    else:
        x0, y0, out_w, out_h = crop_rect

    # Build grid of (i', j') in output
    j_coords, i_coords = np.meshgrid(np.arange(out_w), np.arange(out_h))
    # Shift to center of rotated image
    cx, cy = w / 2.0, h / 2.0
    # Absolute output coordinates in rotated image
    abs_x = j_coords + x0
    abs_y = i_coords + y0

    # Inverse rotation matrix
    θ = np.deg2rad(angle)
    cos, sin = np.cos(θ), np.sin(θ)
    # [ [ cos,  sin], 
    #   [-sin, cos] ]  is R⁻¹ for a CCW rotation by θ
    # Translate coords to origin, rotate back, translate back
    x_rel = abs_x - cx
    y_rel = abs_y - cy
    src_x =  cos * x_rel + sin * y_rel + cx
    src_y = -sin * x_rel + cos * y_rel + cy

    # Build float32 maps for remap
    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)

    return map_x, map_y

def rotate_and_crop(
    frame: np.ndarray,
    angle: float,
    crop_rect: Optional[CropRect] = None
) -> np.ndarray:
    """
    Rotate `frame` by `angle` (degrees CCW) and optionally crop the result.
    Uses precomputed remap tables for efficiency.
    """
    h, w = frame.shape[:2]
    map_x, map_y = make_rotation_maps((h, w), angle, crop_rect)

    # Choose interpolation per dtype
    if frame.dtype == np.bool_:
        # Boolean mask → uint8 for remap
        tmp = (frame.astype(np.uint8) * 255)
        remapped = cv2.remap(tmp, map_x, map_y,
                             interpolation=cv2.INTER_NEAREST)
        return remapped > 127

    else:
        # Normal image: bicubic for quality
        return cv2.remap(frame, map_x, map_y,
                         interpolation=cv2.INTER_CUBIC)
    

def main():
    
    frame = cv2.imread('test.png')  # Load your image/frame here

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    
    # One-time (first call builds maps)
    angle = 60
    crop = (150, 150, 400, 400)  # x, y, width, height

    # For each frame in your video loop:
    rotated = rotate_and_crop(frame, angle, crop)
    # ... process or display `rotated` ...

    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


