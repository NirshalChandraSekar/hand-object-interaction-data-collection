import h5py
import cv2
import numpy as np

dataset_path = 'dataset/trial_dataset/check_video_data.h5'  # <- change this to your actual path if needed
serial_numbers = ['f1371463', 'f1371786']  # <- update these to your recorded camera serial numbers

with h5py.File(dataset_path, 'r') as dataset:
    # Get number of frames recorded (based on the number of color frames for the first camera)
    num_frames = len(dataset[serial_numbers[0]]['color'])

    for frame_idx in range(num_frames):
        for serial in serial_numbers:
            color_image = dataset[serial]['color'][str(frame_idx)][()]
            depth_image = dataset[serial]['depth'][str(frame_idx)][()]

            cv2.imshow(f'Color Image {serial}', color_image)
            # Normalize depth for visualization
            depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow(f'Depth Image {serial}', depth_colored)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
