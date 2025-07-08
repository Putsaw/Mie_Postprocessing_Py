import numpy as np
import cv2
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

CropRect = Tuple[int, int, int, int]  # x, y, w, h

@lru_cache(maxsize=8)
def make_rotation_maps(
    frame_size: Tuple[int, int],
    angle: float,
    crop_rect: Optional[CropRect] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute map_x and map_y for cv2.remap."""
    h, w = frame_size
    if crop_rect is None:
        x0, y0, out_w, out_h = 0, 0, w, h
    else:
        x0, y0, out_w, out_h = crop_rect

    j_coords, i_coords = np.meshgrid(np.arange(out_w), np.arange(out_h))
    cx, cy = w / 2.0, h / 2.0
    abs_x = j_coords + x0
    abs_y = i_coords + y0

    theta = np.deg2rad(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    x_rel = abs_x - cx
    y_rel = abs_y - cy
    src_x = cos_a * x_rel + sin_a * y_rel + cx
    src_y = -sin_a * x_rel + cos_a * y_rel + cy

    return src_x.astype(np.float32), src_y.astype(np.float32)



def _remap_frame(frame: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, use_cuda: bool) -> np.ndarray:
    if frame.dtype == np.bool_:
        tmp = frame.astype(np.uint8) * 255
        if use_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(tmp)
                gpu_map_x = cv2.cuda_GpuMat(); gpu_map_x.upload(map_x)
                gpu_map_y = cv2.cuda_GpuMat(); gpu_map_y.upload(map_y)
                result = cv2.cuda.remap(gpu_frame, gpu_map_x, gpu_map_y, cv2.INTER_NEAREST)
                remapped = result.download()
            except Exception:
                remapped = cv2.remap(tmp, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        else:
            remapped = cv2.remap(tmp, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        return remapped > 127
    else:
        if use_cuda:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_map_x = cv2.cuda_GpuMat(); gpu_map_x.upload(map_x)
                gpu_map_y = cv2.cuda_GpuMat(); gpu_map_y.upload(map_y)
                result = cv2.cuda.remap(gpu_frame, gpu_map_x, gpu_map_y, cv2.INTER_CUBIC)
                return result.download()
            except Exception:
                return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC)


def rotate_and_crop(
    array: np.ndarray,
    angle: float,
    crop_rect: Optional[CropRect] = None,
    is_video: bool = False,
    max_workers: Optional[int] = None
) -> np.ndarray:
    """Rotate an image or video by ``angle`` and optionally crop the result."""
    if is_video:
        h, w = array.shape[1:3]
    else:
        h, w = array.shape[:2]
    map_x, map_y = make_rotation_maps((h, w), angle, crop_rect)

    use_cuda = False
    try:
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            use_cuda = True
    except Exception:
        use_cuda = False

    if is_video:
        num_frames = array.shape[0]
        rotated = [None] * num_frames

        def task(idx: int):
            return idx, _remap_frame(array[idx], map_x, map_y, use_cuda)

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(task, i) for i in range(num_frames)]
            for fut in as_completed(futures):
                idx, out = fut.result()
                rotated[idx] = out
        return np.stack(rotated, axis=0)
    else:
        return _remap_frame(array, map_x, map_y, use_cuda)