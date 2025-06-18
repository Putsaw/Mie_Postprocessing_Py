from packages import *


# -----------------------------
# Cine video reading and playback
# -----------------------------
def read_frame(cine_file_path, frame_offset, width, height):
    with open(cine_file_path, "rb") as f:
        f.seek(frame_offset)
        frame_data = np.fromfile(f, dtype=np.uint16, count=width * height).reshape(height, width)
    return frame_data

def load_cine_video(cine_file_path):
    # Read the header
    header = cine.read_header(cine_file_path)
    # Extract width, height, and total frame count
    width = header['bitmapinfoheader'].biWidth
    height = header['bitmapinfoheader'].biHeight
    frame_offsets = header['pImage']  # List of frame offsets
    frame_count = len(frame_offsets)
    print(f"Video Info - Width: {width}, Height: {height}, Frames: {frame_count}")

    # Initialize an empty 3D NumPy array to store all frames
    video_data = np.zeros((frame_count, height, width), dtype=np.uint16)
    # Use ThreadPoolExecutor to read frames in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_index = {
            executor.submit(read_frame, cine_file_path, frame_offsets[i], width, height): i
            for i in range(frame_count)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                video_data[index] = future.result()
            except Exception as e:
                print(f"Error reading frame {index}: {e}")
    return video_data

def get_subfolder_names(parent_folder):
    parent_folder = Path(parent_folder)
    subfolder_names = [item.name for item in parent_folder.iterdir() if item.is_dir()]
    return subfolder_names



def play_video_cv2(video, gain=1, binarize=False, thresh=0.5):
    """
    Play a list/array of video frames with OpenCV, with optional binarization.

    Parameters
    ----------
    video : sequence of np.ndarray. Int, float or bool
        视频帧列表，每帧可以是整数、浮点数，也可以是布尔数组。
    gain : float, optional. 
        灰度增益，对原始数值做线性放缩（默认 1）。
    binarize : bool, optional
        是否先将帧转换为布尔再显示（默认 False）。
    thresh : float, optional
        当 binarize=True 且输入不是布尔类型时，使用该阈值做二值化（浮点[0,1]或任意范围均可）。
    """
    total_frames = len(video)
    if total_frames == 0:
        return

    # 先检测第 1 帧的数据类型
    first_dtype = video[0].dtype

    for i in range(total_frames):
        frame = video[i]

        # —— 二值化分支 ——
        if binarize:
            # 如果是非布尔类型，先做阈值处理
            if frame.dtype != bool:
                # 假定浮点帧在 [0,1]，或任意数值，都可以用 thresh 来分割
                frame_bool = frame > thresh
            else:
                frame_bool = frame
            # True→255, False→0
            frame_uint8 = (frame_bool.astype(np.uint8)) * 255

        # —— 原有灰度／色阶分支 ——
        else:
            dtype = frame.dtype
            # 整数：假设是 16-bit 量程，缩到 8-bit
            if np.issubdtype(dtype, np.integer):
                frame_uint8 = gain * (frame / 16).astype(np.uint8)
            # 浮点：假设在 [0,1]，放大到 0–255
            elif np.issubdtype(dtype, np.floating):
                frame_uint8 = np.clip(gain * (frame * 255), 0, 255).astype(np.uint8)
            # 其他类型回退到整数缩放
            else:
                frame_uint8 = gain * (frame / 16).astype(np.uint8)

        # 显示
        cv2.imshow('Frame', frame_uint8)
        # ~60fps 播放，按 'q' 退出
        if cv2.waitKey(17) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()




# -----------------------------
# Rotation and Filtering functions
# -----------------------------
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if frame.dtype == np.bool_:
        # Convert boolean mask to uint8: True becomes 255, False becomes 0.
        frame_uint8 = (frame.astype(np.uint8)) * 255
        # Use INTER_NEAREST to preserve mask values.
        rotated_uint8 = cv2.warpAffine(frame_uint8, M, (w, h), flags=cv2.INTER_NEAREST)
        # Convert back to boolean mask.
        rotated = rotated_uint8 > 127
    else:
        rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return rotated


def rotate_video(video_array, angle=0, max_workers=None):
    num_frames = video_array.shape[0]
    rotated_frames = [None] * num_frames
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(rotate_frame, video_array[i], angle): i 
                           for i in range(num_frames)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                rotated_frames[idx] = future.result()
            except Exception as exc:
                print(f"Frame {idx} generated an exception during rotation: {exc}")
    return np.array(rotated_frames)


from concurrent.futures import as_completed, ThreadPoolExecutor

# -----------------------------
# CUDA-accelerated Rotation
# -----------------------------
def rotate_frame_cuda(frame, angle, stream=None):
    """
    在 GPU 上旋转单帧图像／掩码。
    
    Parameters
    ----------
    frame : np.ndarray (H×W or H×W×C) or bool mask
    angle : float
    stream: cv2.cuda.Stream (optional) — 用于异步操作
    
    Returns
    -------
    rotated : 同 frame 类型
    """
    h, w = frame.shape[:2]
    # 计算仿射矩阵（在 CPU 上）
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
    
    # 上传到 GPU
    if frame.dtype == np.bool_:
        # 布尔先转 uint8（0/255）
        cpu_uint8 = (frame.astype(np.uint8)) * 255
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(cpu_uint8, stream)
        # warpAffine（最近邻保留掩码边界）
        gpu_rot = cv2.cuda.warpAffine(
            gpu_mat, M, (w, h),
            flags=cv2.INTER_NEAREST, stream=stream
        )
        # 下载并阈值回布尔
        out_uint8 = gpu_rot.download(stream)
        rotated = out_uint8 > 127
    else:
        # 对普通灰度或多通道图像
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(frame, stream)
        gpu_rot = cv2.cuda.warpAffine(
            gpu_mat, M, (w, h),
            flags=cv2.INTER_CUBIC, stream=stream
        )
        rotated = gpu_rot.download(stream)
    
    # 等待 GPU 流完成
    if stream is not None:
        stream.waitForCompletion()
    return rotated


def rotate_video_cuda(video_array, angle=0, max_workers=4):
    """
    并行地在 GPU 上旋转整个视频（每帧独立流）。
    
    Parameters
    ----------
    video_array : np.ndarray, shape=(N, H, W) 或 (N, H, W, C) 或 bool
    angle : float — 旋转角度（度）
    max_workers : int — 并行线程数（每线程管理一个 cv2.cuda.Stream）
    
    Returns
    -------
    np.ndarray — 与输入同形状、同 dtype
    """
    num_frames = video_array.shape[0]
    rotated = [None] * num_frames

    # 预创建若干 CUDA 流
    streams = [cv2.cuda.Stream() for _ in range(max_workers)]

    def task(idx, frame):
        # 分配一个流（简单轮询）
        stream = streams[idx % max_workers]
        return idx, rotate_frame_cuda(frame, angle, stream)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(task, i, video_array[i]) for i in range(num_frames)]
        for fut in as_completed(futures):
            idx, out = fut.result()
            rotated[idx] = out

    return np.stack(rotated, axis=0)



# -----------------------------
# Masking and Binarization Pipeline
# -----------------------------
'''
def mask_frame(i, video, chamber_mask_bool):
    return video[i] * chamber_mask_bool


def mask_video(video: np.ndarray, chamber_mask: np.ndarray):
    num_frames, height, width = video.shape
    masked_video = np.zeros_like(video)
    # Ensure chamber_mask is boolean
    chamber_mask_bool = chamber_mask if chamber_mask.dtype == bool else (chamber_mask > 0)
    
    # Use executor.map with the top-level mask_frame function.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Pass video and chamber_mask_bool as iterables by repeating them for each frame.
        results = list(executor.map(mask_frame, range(num_frames), [video]*num_frames, [chamber_mask_bool]*num_frames))
    
    for i, frame in enumerate(results):
        masked_video[i] = frame
        
    return masked_video
'''
def mask_video(video: np.ndarray, chamber_mask: np.ndarray) -> np.ndarray:
    # Ensure chamber_mask is boolean.
    chamber_mask_bool = chamber_mask if chamber_mask.dtype == bool else (chamber_mask > 0)
    # Use broadcasting: multiplies each frame elementwise with the mask.
    return video * chamber_mask_bool


# -----------------------------
# Global Threshold Binarization
# -----------------------------
def binarize_video_global_threshold(video, method='otsu', thresh_val=None):
    if method == 'otsu':
        # Compute threshold over the whole video (flattened)
        threshold = threshold_otsu(video)
    elif method == 'fixed':
        if thresh_val is None:
            raise ValueError("Provide a threshold value for 'fixed' method.")
        threshold = thresh_val
    else:
        raise ValueError("Invalid method. Use 'otsu' or 'fixed'.")
    
    # Broadcasting applies the comparison element-wise across the entire video array.
    binary_video = (video >= threshold).astype(np.uint8) * 255
    return binary_video

def is_opencv_cuda_available():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except AttributeError:
        return False
    
def rotate_video_auto(video_array, angle=0, max_workers=4):
    if is_opencv_cuda_available():
        print("Using CUDA for rotation.")
        return rotate_video_cuda(video_array, angle=angle, max_workers=max_workers)
    else:
        print("CUDA not available, falling back to CPU.")
        return rotate_video(video_array, angle=angle, max_workers=max_workers)
