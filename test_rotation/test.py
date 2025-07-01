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

'''


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
'''
                         
def rotate_and_crop(
    frame: np.ndarray,
    angle: float,
    crop_rect: Optional[CropRect] = None
) -> np.ndarray:
    """
    Rotate `frame` by `angle` (degrees CCW) and optionally crop the result.
    Uses precomputed remap tables for efficiency.
    Uses CUDA acceleration if available.
    """
    h, w = frame.shape[:2]
    map_x, map_y = make_rotation_maps((h, w), angle, crop_rect)
    
    # Check if CUDA is available
    use_cuda = False
    try:
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            use_cuda = True
    except (AttributeError, ImportError):
        # cv2.cuda is not available
        use_cuda = False

    # Choose interpolation per dtype
    if frame.dtype == np.bool_:
        # Boolean mask → uint8 for remap
        tmp = (frame.astype(np.uint8) * 255)
        
        if use_cuda:
            try:
                # Convert to GPU
                gpu_tmp = cv2.cuda.GpuMat()
                gpu_tmp.upload(tmp)
                
                # Create GPU maps
                gpu_map_x = cv2.cuda.GpuMat()
                gpu_map_x.upload(map_x)
                gpu_map_y = cv2.cuda.GpuMat()
                gpu_map_y.upload(map_y)
                
                # Perform remap on GPU
                gpu_result = cv2.cuda.remap(gpu_tmp, gpu_map_x, gpu_map_y, 
                                            cv2.INTER_NEAREST)
                
                # Download result
                remapped = gpu_result.download()
            except Exception:
                # Fall back to CPU if CUDA operation fails
                remapped = cv2.remap(tmp, map_x, map_y,
                                   interpolation=cv2.INTER_NEAREST)
        else:
            remapped = cv2.remap(tmp, map_x, map_y,
                               interpolation=cv2.INTER_NEAREST)
            
        return remapped > 127

    else:
        # Normal image: bicubic for quality
        if use_cuda:
            try:
                # Convert to GPU
                gpu_frame = cv2.cuda.GpuMat()
                gpu_frame.upload(frame)
                
                # Create GPU maps
                gpu_map_x = cv2.cuda.GpuMat()
                gpu_map_x.upload(map_x)
                gpu_map_y = cv2.cuda.GpuMat()
                gpu_map_y.upload(map_y)
                
                # Perform remap on GPU
                gpu_result = cv2.cuda.remap(gpu_frame, gpu_map_x, gpu_map_y, 
                                           cv2.INTER_CUBIC)
                
                # Download result
                return gpu_result.download()
            except Exception:
                # Fall back to CPU if CUDA operation fails
                return cv2.remap(frame, map_x, map_y,
                              interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.remap(frame, map_x, map_y,
                          interpolation=cv2.INTER_CUBIC)


def main():
    
    frame = cv2.imread('test.png')  # Load your image/frame here

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    
    # One-time (first call builds maps)
    angle = 60
    crop = (50, 150, 400, 400)  # x, y, width, height

    # For each frame in your video loop:
    rotated = rotate_and_crop(frame, angle, crop)
    # ... process or display `rotated` ...

    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


