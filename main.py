# from fill import *
# from functions import *
from functions_videos import *
# from functions_optical_flow import *
# from boundary2 import *
import matplotlib.pyplot as plt



# from std_functions3 import *

import subprocess
# from filter_video import *
from scipy.signal import convolve2d

import asyncio
# Define a semaphore with a limit on concurrent tasks
SEMAPHORE_LIMIT = 2  # Adjust this based on your CPU capacity
semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)


async def play_video_cv2_async(video, gain=1, binarize=False, thresh=0.5, intv=17):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, play_video_cv2, video, gain, binarize, thresh, intv)



async def main():
    parent_folder = r"G:\Master_Thesis\BC20220627 - Heinzman DS300 - Mie Top view\Cine"
    
    #parent_folder = r"E:\TP_example"
    subfolders = get_subfolder_names(parent_folder)  # Ensure get_subfolder_names is defined or imported

    
    if os.path.exists("chamber_mask.npy"):
        chamber_mask = np.load("chamber_mask.npy")
    else: 
        subprocess.run(["python", "masking.py"], check=True)
        chamber_mask = np.load("chamber_mask.npy")
    
    if os.path.exists("test_mask.npy"):
        test_mask = np.load("test_mask.npy")==0
    


    for subfolder in subfolders:
        print(subfolder)
    
        # Specify the directory path
        directory_path = Path(parent_folder + "\\" + subfolder)
    
        # Get a list of all files in the directory
        files = [file for file in directory_path.iterdir() if file.is_file()]
    
        print(files)
        for file in files:
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
                mie_video = mask_video(video, test_mask)
                # play_video_cv2(mie_video, intv=17)

                # mapping the video to a 2D image of its pixel intensity ranges
                range_map = map_video_to_range(mie_video)

                # cv2.imshow("Range Map", range_map)

                filtered_range_map = cv2.GaussianBlur(range_map, (5, 5), 0)
                                
                cv2.imshow("Filtered Range Map", filtered_range_map)
                cv2.waitKey(0)      
                cv2.destroyAllWindows()
                
                              
                '''
                import matplotlib.pyplot as plt  
                bins = 100
                # hist = cv2.calcHist([filtered_range_map], [0], None, [bins], [0, 1], accumulate=False)
                # hist_norm = hist / hist.sum()          # probability mass
                hist, edges = np.histogram(filtered_range_map.ravel(), bins=bins, range=(0,1))
                centers = (edges[:-1] + edges[1:]) / 2
                # plt.plot(hist_norm); plt.xlim([0, bins]); plt.show()
                plt.plot(centers, hist); plt.ylim([0, 1e2]); plt.show()
                '''
                
                '''
                # Showing the histogram and CDF of the filtered range map
                import matplotlib.pyplot as plt

                bins = 1000
                hist, edges = np.histogram(filtered_range_map.ravel(),
                                        bins=bins, range=(0, 1))

                centers = (edges[:-1] + edges[1:]) / 2

                # ---------- 1. Histogram on a logarithmic y-axis ----------
                fig, ax = plt.subplots()
                ax.plot(centers, hist, lw=1.2)
                ax.set_yscale('log')                 # logarithmic scale
                ax.set_ylim(bottom=1)                # avoid log(0) issues
                ax.set_xlabel("Range value")
                ax.set_ylabel("Count (log scale)")
                ax.set_title("Histogram of filtered_range_map")
                ax.grid(True, which='both', ls='--', alpha=0.3)

                # ---------- 2. Cumulative distribution function -----------
                cdf = np.cumsum(hist).astype(float)
                cdf /= cdf[-1]                       # normalise so max == 1

                fig2, ax2 = plt.subplots()
                ax2.plot(centers, cdf, lw=1.2)
                ax2.set_ylim(0, 1.1)                   # cap at 100 %
                ax2.set_xlabel("Range value")
                ax2.set_ylabel("CDF")
                ax2.set_title("CDF of filtered_range_map")
                ax2.grid(True, ls='--', alpha=0.3)

                plt.show()
                '''
                
                '''
                bins = 1000
                hist, edges = np.histogram(filtered_range_map.ravel(), bins=bins, range=(0,1))
                centers = (edges[:-1] + edges[1:]) / 2
                plt.plot(centers, hist)
                plt.show()
                '''

                '''
                # Segmentation with kmeans
                pixel_vals = filtered_range_map.reshape((-1, 1))
                # Set k-means criteria and perform clustering (e.g., k=2 for two segments)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 2
                attempts = 10  

                compactness, labels, centers = cv2.kmeans(
                    pixel_vals, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
                ) # type: ignore
                labels = labels.flatten()
                segmented = centers[labels].reshape(filtered_range_map.shape).astype(np.uint16)
                
                # Optionally, convert labels to a binary mask:
                bw = ~(labels.reshape(filtered_range_map.shape) == np.argmax(np.bincount(labels))).astype(np.uint8) * 255
                cv2.imshow("Binary Map Kmeans", bw)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''


                # bw_map = cv2.threshold(filtered_range_map 0, 65535, cv2.NORM, 0, 255, cv2.THRESH_BINARY)[1]
                # bw_map = cv2.threshold(filtered_range_map,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                '''
                img16 = cv2.normalize(filtered_range_map, None, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
                T16, bw = cv2.threshold(img16, 0, 65535,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                print(f"Threshold value for glare mask: {T16/65536.0}")
                
                cv2.imshow("Binary Map", bw)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
                # Normalize the filtered range map to 16-bit unsigned integer
                img16 = cv2.normalize(filtered_range_map, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U) # type: ignore
                
                # Thresholding to create a binary mask
                # pct = np.percentile(img16, 5)
                pct = 0.05
                

                # Thresholding
                _, bw = cv2.threshold(img16, round(pct*65535), 65535,cv2.THRESH_BINARY)
                cv2.imshow("Binary Map", bw)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Attempt to solve 
                # First derivatives
                sobelx = cv2.Sobel(filtered_range_map.astype(np.float64), cv2.CV_64F, dx=1, dy=0, ksize=3)
                sobely = cv2.Sobel(filtered_range_map.astype(np.float64), cv2.CV_64F, dx=0, dy=1, ksize=3)

                # Compute gradient magnitude
                grad_mag = np.sqrt(sobelx**2 + sobely**2)

                # Normalize to display as 8-bit
                grad_mag = cv2.convertScaleAbs(grad_mag)
                cv2.imshow("Gradient Magnitude", grad_mag*255.0)





                masked_video = 10*mask_video(mie_video, bw)
                play_video_cv2(masked_video, intv=17)
                play_video_cv2(mask_video(mie_video, ~bw), intv=17)


if __name__ == '__main__':
    
    
    from multiprocessing import freeze_support
    freeze_support()
    import asyncio, time

    start = time.time()
    asyncio.run(main())
    print(f"Total elapsed: {time.time() - start:.2f}s")
    
