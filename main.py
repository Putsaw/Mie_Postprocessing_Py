from functions_videos import *
import matplotlib.pyplot as plt
import subprocess
from scipy.signal import convolve2d
import asyncio
from rotate_crop import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc
from ssim import compute_ssim_segments
import json

global parent_folder
global plumes 
global offset
global centre

parent_folder = r"C:\Users\LJI008\OneDrive - Wärtsilä Corporation\Documents\DS300_ex"

# Define a semaphore with a limit on concurrent tasks
SEMAPHORE_LIMIT = 2  # Adjust this based on your CPU capacity
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


async def play_video_cv2_async(video, gain=1, binarize=False, thresh=0.5, intv=17):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, play_video_cv2, video, gain, binarize, thresh, intv)

def MIE_pipeline(video, number_of_plumes, offset, centre):
    foreground = subtract_median_background(video, frame_range=slice(0, 30))
    # play_video_cv2(foreground, intv=17)
    '''
    gamma = foreground ** 2 # gamma correction
    gamma[gamma < 2e-2 ] = 0  # thresholding
    gain = gamma * 5  # gain correction
    gain[gain > 1] = 1  # limit gain to 1
    # play_video_cv2(gain, intv=17)

    print("Gain correction has range from %f to %f" % (gain.min(), gain.max()))
    '''
    gain = foreground
    gamma = foreground 
    
    # centre = (384.9337805142379, 382.593916979227)
    # crop = (round(centre[0]), round(centre[1]- 768/16), round(768/2), round(768/8))

    ir_ = 14
    or_ = 380

    # centre_x = 384.9337805142379
    # centre_y = 382.593916979227
    centre_x = float(centre[0])
    centre_y = float(centre[1])

    # Generate the crop rectangle based on the plume parameters
    crop = generate_CropRect(ir_, or_, number_of_plumes, centre_x, centre_y)

    # offset = 2
    angles = np.linspace(0, 360, number_of_plumes, endpoint=False) + offset
    mask = generate_plume_mask(ir_, or_, crop[2], crop[3])

    segments = []

    # Multithreaded rotation and cropping
    with ThreadPoolExecutor(max_workers=min(len(angles), os.cpu_count() or 1)) as exe:
        future_map = {
            exe.submit(
                rotate_and_crop, gain, angle, crop, centre,
                is_video=True, mask=mask
            ): idx for idx, angle in enumerate(angles)
        }
        segments_with_idx = []
        for fut in as_completed(future_map):
            idx = future_map[fut]
            result = fut.result()
            segments_with_idx.append((idx, result))
        # Sort by index to preserve order
        segments_with_idx.sort(key=lambda x: x[0])
        segments = [seg for idx, seg in segments_with_idx]
    
    # Free intermediate arrays to reduce peak memory usage
    del foreground, gain, gamma
    gc.collect()

    '''
    # Stacking the segments into a 4D array
    segments = [seg for seg in segments if seg is not None]
    if not segments:
        raise ValueError("No valid segments to stack.")
    '''
    segments = np.stack(segments, axis=0)

    average_segment = np.mean(segments, axis=0) # Average across the segments

    # for seg in segments:
        # pass  # segments are already masked during rotation

        # play_video_cv2(seg, intv=17)
        # Launch SSIM computation asynchronously; results are not needed immediately
    # loop = asyncio.get_running_loop()
    # ssim_task = loop.create_task(asyncio.to_thread(compute_ssim_segments, segments,
                                       # average_segment))


    play_video_cv2(video, intv=17)
    total_frames = len(video)
    binary_video = []
    for i in range(total_frames):
        frame = video[i]

        # Apply Gaussian Blur
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Otsu's binarization requires 8 or 16 bit data
        blur_8bit = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        # Normalize to 16-bit range
        image_16bit = cv2.normalize(blur, None, 0, 65535, cv2.NORM_MINMAX).astype('uint16')

        # Apply Otsu's Binarization
        _, binary8bit = cv2.threshold(blur_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary16bit = cv2.threshold(image_16bit, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Show before and after
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Binarized Frame', binary16bit)
        cv2.waitKey(200)

        binary_video.append(binary16bit)

    
    cv2.destroyAllWindows()

    #play_video_cv2(binary_video, intv=100)

    # ssim_matrix = await ssim_task    # this yields the ndarray of SSIM scores
    ssim_matrix = compute_ssim_segments(segments,average_segment)
    # plt.imshow(ssim_matrix, aspect='auto')
    # plt.colorbar()
    plt.plot(ssim_matrix.transpose())
    
    plt.show()

    labels = kmeans_label_video(video, k=3)
    print('labels shape', labels.shape)
    print('unique labels', np.unique(labels))
    playable = labels_to_playable_video(labels, k=2)
    print('playable min', playable.min(), 'max', playable.max())
    # play_video_cv2(playable)
    play_videos_side_by_side([video, playable], intv = 170)

    playable = labels_to_playable_video(labels, k=2)
    print('playable min', playable.min(), 'max', playable.max())
    # play_video_cv2(playable)
    play_videos_side_by_side([video, playable], intv = 170)

    1

async def main():
    # parent_folder = r"G:\Master_Thesis\BC20220627 - Heinzman DS300 - Mie Top view\Cine\Interest"
    
    #parent_folder = r"E:\TP_example"
    subfolders = get_subfolder_names(parent_folder)  # Ensure get_subfolder_names is defined or imported

    
    if os.path.exists("chamber_mask.npy"):
        chamber_mask = np.load("chamber_mask.npy")
    else: 
        subprocess.run(["python", "masking.py"], check=True)
        chamber_mask = np.load("chamber_mask.npy")
    
    if os.path.exists("test_mask.npy"):
        test_mask = np.load("test_mask.npy")==0

    if os.path.exists("region_unblocked.npy"):
        region_unblocked = np.load("region_unblocked.npy")
    


    for subfolder in subfolders:
        print(subfolder)
    
        # Specify the directory path
        directory_path = Path(parent_folder + "\\" + subfolder)
    
        # Get a list of all files in the directory
        files = [file for file in directory_path.iterdir() if file.is_file()]

        for file in files:
            if file.name == 'config.json':
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # process the data
                    # for item in data:
                        # print(item)
                    plumes = int(data['plumes'])
                    offset = float(data['offset'])
                    centre = [float(data['centre_x']), float(data['centre_y'])]

        # print(files)
        for file in files:
            if file.suffix == '.cine':
                print("Procssing:", file)

                video = load_cine_video(file).astype(np.float32)/4096  # Ensure load_cine_video is defined or imported
                frames, height, width = video.shape
                # video = video.astype(float)
                # play_video_cv2(video)
                # video = video**2

                if "Shadow" in file.name:
                    continue

                    # Angle of Rotation
                    rotation = 45

                    # Strip cutting
                    x_start = 1
                    x_end = -1
                    y_start = 150
                    y_end = -250

                    
                    '''
                    start_time = time.time()
                    RT = rotate_video(video, rotation)
                    elapsed_time = time.time() - start_time
                    print(f"Rotating video with CPU finished in {elapsed_time:.2f} seconds.")
                    '''
                    
                    start_time = time.time()
                    RT = rotate_video_cuda(video, rotation)
                    elapsed_time = time.time() - start_time
                    print(f"Rotating video with GPU finished in {elapsed_time:.2f} seconds.")
                    
                    



                    # frame, y, x
                    strip = RT[15:400, y_start:y_end, x_start:x_end]
                    strip = strip.astype(float)
                    
                    
                    masked_strip = strip

                    # play_video_cv2(masked_strip)

                    lap = np.array([[1,1,1],[1,-8,1],[1,1,1]], dtype=float)
                    
                    HP = filter_video_fft(strip, lap, mode='same')

                    avg = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=float)

                    HP_avg = filter_video_fft(HP, avg, mode='same')
                    # play_video_cv2(HP_avg)
                    await play_video_cv2_async(HP_avg)

                        
                    # STD filtering
                    # parameters:
                    # Standard deviation filter window size
                    # Downsampling factor
                    # std_video = stdfilt_video(strip, std_size, pool_size)

                    '''                
                    std_video = stdfilt_video_parallel_optimized(masked_strip, std_size=3, pool_size=2)

                    bw_std = binarize_video_global_threshold(std_video, method='fixed', thresh_val=2E2)

                    bw_std_filled = fill_video_holes_parallel(bw_std)

                    '''

                    '''                
                    start_time = time.time()
                    velocity_field = compute_optical_flow(strip)
                    elapsed_time = time.time() - start_time
                    print(f"OFE with CPU finished in {elapsed_time:.2f} seconds.")
                    '''

                    HP_delay = np.zeros(HP_avg.shape, dtype=float)

                    HP_delay[1:-1, :, :] = HP_avg[0:-2, :,:]

                    HP_res = np.abs(HP_delay-HP_avg)
                    # import numpy as np
                    import cupy as cp
                    import matplotlib.pyplot as plt

                    # 2) Upload to GPU
                    vol_gpu = cp.array(HP_res)
                    # 3) Plan and execute 3D complex‐to‐complex FFT in place
                    #    (cuFFT automatically picks the fastest algorithm)
                    vol_fft = cp.fft.fftn(vol_gpu, axes=(0,1,2))

                    # 4) Shift zero‐frequency to center (optional)
                    vol_fft = cp.fft.fftshift(vol_fft, axes=(0,1,2))

                    # 5) Compute magnitude (abs) and bring back to CPU
                    mag_gpu = cp.abs(vol_fft)
                    mag = cp.asnumpy(mag_gpu)
                    nx, ny, nz = HP_res.shape
                    # 6) Visualize three orthogonal slices
                    fig, axes = plt.subplots(1,3, figsize=(12,4))
                    slices = [
                        mag[nx//2, :, :],  # Y–Z at center X
                        mag[:, ny//2, :],  # X–Z at center Y
                        mag[:, :, nz//2],  # X–Y at center Z
                    ]
                    titles = ['Slice X=mid','Slice Y=mid','Slice Z=mid']
                    for ax, slc, title in zip(axes, slices, titles):
                        im = ax.imshow(np.log1p(slc), origin='lower')
                        ax.set_title(title)
                        fig.colorbar(im, ax=ax, fraction=0.046)
                    plt.tight_layout()
                    plt.show()






                    # await play_video_cv2_async(HP_res/1024)

                                    
                    start_time = time.time()
                    velocity_field = compute_optical_flow_cuda(HP_res)
                    elapsed_time = time.time() - start_time
                    print(f"OFE with GPU finished in {elapsed_time:.2f} seconds.")

                    scalar_velocity_field = compute_flow_scalar_video(velocity_field, multiplier=1, y_scale=1)
                    

                    start_time = time.time()
                    scalar_velocity_field_med = median_filter_video_cuda(HP_res, 5, 5)
                    elapsed_time = time.time() - start_time
                    print(f"Medfilt with GPU finished in {elapsed_time:.2f} seconds.")

                    await play_video_cv2_async(scalar_velocity_field_med/1024)


                    bw_flow = mask_video(binarize_video_global_threshold(scalar_velocity_field_med, method='otsu'), chamber_mask)

                    await play_video_cv2_async(bw_flow, gain=255)
        
                    
                                    
                    start_time = time.time()
                    bw_flow_filled = fill_video_holes_parallel(bw_flow)
                    elapsed_time = time.time() - start_time
                    print(f"Hole-filling with CPU finished in {elapsed_time:.2f} seconds.")
                    

                    start_time = time.time()
                    bw_flow_filled = fill_video_holes_gpu(bw_flow)
                    elapsed_time = time.time() - start_time
                    print(f"Hole-filling with GPU finished in {elapsed_time:.2f} seconds.")

                    await play_video_cv2_async(bw_flow_filled, gain=255, binarize=True)


                    # results = compute_boundaries_parallel_all(bw_flow_filled)
                    

                    # results = compute_boundaries_parallel_all(bw_std_filled)


                    # play_video_cv2(strip)
                    # play_video_cv2(masked_strip)
                    # play_video_cv2(bw_std, 4)
                    # play_video_cv2(std_video/1E3)
                    # masked_std_video = mask_video(std_video, chamber_mask)
                    # play_video_cv2(masked_std_video/1E3)
                    # play_video_cv2(scalar_velocity_field, 1)
                    # play_video_cv2(bw_flow_filled, 10)



                    # ... after computing `results = compute_boundaries_parallel_all(bw_flow)` ...
                    
                    plt.ion()
                    fig, ax = plt.subplots()
                    im = ax.imshow(masked_strip[0], cmap='gray')
                    ax.set_title("Frame 0 Boundaries")
                    ax.axis('off')

                    for idx, res in enumerate(results):
                        frame = masked_strip[idx]
                        im.set_data(frame)
                        ax.set_title(f"Frame {idx} Boundaries")
                        
                        # 1) Remove old contour lines
                        #    This clears any Line2D objects currently on the axes.
                        for ln in ax.lines:
                            ln.remove()

                        '''
                        # Plotting all countors                    
                        # 2) Plot every contour for every component
                        for comp in res.components:
                            for contour in comp.boundaries:
                                y, x = contour[:, 0], contour[:, 1]
                                ax.plot(x, y, '-r', linewidth=2)
                        


                        # Countor of the biggest area
                        # 2) If any components found, plot only the largest one
                        if res.components:
                            # Find component with max area
                            largest_comp = max(res.components, key=lambda c: c.area)
                            
                            # Plot its first contour (there may be multiple loops)
                            if largest_comp.boundaries:
                                contour = largest_comp.boundaries[0]
                                y, x = contour[:, 0], contour[:, 1]
                                ax.plot(x, y, '-r', linewidth=2)
                    '''
                        # Longest contour
                        # 2) Find the single longest contour across all components
                        longest_contour = None
                        max_len = 0
                        for comp in res.components:
                            for contour in comp.boundaries:
                                n_pts = contour.shape[0]
                                if n_pts > max_len:
                                    max_len = n_pts
                                    longest_contour = contour
                        if longest_contour is not None:
                            y, x = longest_contour[:, 0], longest_contour[:, 1]
                            ax.plot(x, y, '-r', linewidth=2)

                        # 3) Redraw
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.1)
                    plt.close('all')
                elif "OH" in file.name:

                    continue

                    RT = rotate_video(video, -45)
                    strip = RT[0:150, 250:550, :]
                    LP_filtered = Gaussian_LP_video(strip, 40)
                    med = median_filter_video(LP_filtered, 5, 5)
                    

                    BW = binarize_video_global_threshold(med,"fixed", 800)

                    play_video_cv2(strip*10)
                    play_video_cv2(BW*255.0)

                    TD_map = calculate_TD_map(strip)
                    area = calculate_bw_area(BW)
                    
                    '''
                    plt.figure()
                    plt.imshow(TD_map, cmap='jet', aspect='auto')
                    plt.title("Average Time–Distance Map")
                    plt.xlabel("Time (frames)")
                    plt.ylabel("Distance (pixels)")
                    plt.colorbar(label="Sum Intensity")
                    plt.show()

                    plt.figure(figsize=(10, 4))
                    plt.plot(area, color='blue')
                    plt.xlabel("Frame")
                    plt.ylabel("Area (white pixels)")
                    plt.title("Area Over Time")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()'''
                else:
                    
                    # gamma correcetion of video
                    # mie_video = mask_video(video[15:150,:,:], chamber_mask)
                    mie_video = mask_video(video, ~chamber_mask)

                    MIE_pipeline(mie_video, plumes, offset, centre)
                    

if __name__ == '__main__':
    
    from multiprocessing import freeze_support
    freeze_support()
    import asyncio, time

    start = time.time()
    asyncio.run(main())
    print(f"Total elapsed: {time.time() - start:.2f}s")
    
