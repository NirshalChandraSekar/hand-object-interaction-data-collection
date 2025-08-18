import h5py
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs
import math
import os
import calibration_utils as calib


def save_video_from_dataset(dataset_path, output_video_path):
    """
    Combines color and depth frames from an HDF5 dataset into a single video file.

    Each row in the output video corresponds to one camera, and each row contains
    side-by-side color and depth visualizations.

    Args:
        dataset_path (str): Path to the HDF5 dataset.
        output_video_path (str): Path to save the output video.
    """
     
    with h5py.File(dataset_path, 'r') as dataset:
        serial_numbers = list(dataset.keys())
        num_frames = min(len(dataset[f'{serial}/frames/color']) for serial in serial_numbers)
        frame_size = (1280, 720)  # Width x Height

        output_width = frame_size[0] * 2  # RGB + Depth
        output_height = frame_size[1] * len(serial_numbers)  # One row per camera

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (output_width, output_height))

        for frame_idx in range(num_frames):
            rows = []
            for serial in serial_numbers:
                # Load color and depth images
                color_image = dataset[f'{serial}/frames/color'][str(frame_idx)][()]
                depth_image = dataset[f'{serial}/frames/depth'][str(frame_idx)][()]

                # Normalize and colorize depth
                depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # Concatenate
                camera_row = np.hstack([color_image, depth_colored])
                rows.append(camera_row)

            if rows:
                combined_frame = np.vstack(rows)
                out.write(combined_frame)

        out.release()
        print(f"Video saved to {output_video_path}")


def run_mediapipe_on_videos(dataset_path, output_video_path):
    """
    Runs MediaPipe Hands on RGB frames from an HDF5 dataset and saves the output
    as a video showing hand landmarks.

    Args:
        dataset_path (str): Path to the HDF5 dataset.
        output_video_path (str): Path to save the annotated video.
    """
    serial_numbers = list(dataset.keys())
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    with h5py.File(dataset_path, 'r') as dataset:
        num_frames = len(dataset[serial_numbers[0]]['color'])
        frame_size = (1280, 720)  # Assuming all images are 640x480

        # Only RGB frames side by side
        output_width = frame_size[0] * len(serial_numbers)  # Width is now cameras side by side
        output_height = frame_size[1]  # Single row height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (output_width, output_height))

        for frame_idx in range(num_frames):
            rgb_frames = []
            
            for serial in serial_numbers:
                # Get color image only
                color_image = dataset[serial]['color'][str(frame_idx)][()]

                # Process the color image with MediaPipe Hands
                results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                color_image_resized = cv2.resize(color_image, frame_size)
                rgb_frames.append(color_image_resized)
            
            # Stack RGB frames horizontally
            combined_frame = np.hstack(rgb_frames)
            out.write(combined_frame)
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{num_frames} frames")

        out.release()
        print(f"Video with MediaPipe results saved to {output_video_path}")



def view_stream_from_dataset(dataset_path, serial_numbers):
    """
    Displays synchronized color and depth streams from the dataset for all cameras.

    Args:
        dataset_path (str): Path to the HDF5 dataset.
        serial_numbers (list): List of camera serial numbers to visualize.
    """
    with h5py.File(dataset_path, 'r') as dataset:
        num_frames = min(len(dataset[f'{serial_number}/frames/color']) for serial_number in serial_numbers)


        for frame_idx in range(num_frames):
            for serial in serial_numbers:
                color_image = dataset[f'{serial}/frames/color'][str(frame_idx)][()]
                depth_image = dataset[f'{serial}/frames/depth'][str(frame_idx)][()]

                cv2.imshow(f'Color Image {serial}', color_image)
                # Normalize depth for visualization
                depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow(f'Depth Image {serial}', depth_colored)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def view_single_pcd(serial_number, dataset_path):
    """
    Visualizes a 3D point cloud for a single camera using a specific frame from the dataset.

    Args:
        serial_number (str): Serial number of the camera.
        dataset_path (str): Path to the HDF5 dataset.
    """
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = (dataset[f'{serial_number}/frames/color'][str(20)][()]) #.astype(np.float32) * 0.001
        cv2.imshow('Color Image', color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Open3D
        depth_image = (dataset[f'{serial_number}/frames/depth'][str(20)][()]).astype(np.float32) * 0.001  # Convert depth to meters
        print("Depth image:", depth_image)
        intrinsics_ = np.load(f'data/{serial_number}_intrinsics.npy', allow_pickle=True).item()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsics_['width'],
            height=intrinsics_['height'],
            fx=intrinsics_['fx'],
            fy=intrinsics_['fy'],
            cx=intrinsics_['ppx'],
            cy=intrinsics_['ppy'])

        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1,  # Adjust based on your depth image scale
            depth_trunc=3.0,  # Truncate depth values beyond this distance
            convert_rgb_to_intensity=False)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        o3d.visualization.draw_geometries([pcd])

def view_combined_pcd(serial_nums, t_matrices, dataset_path='data'):
    """
    Loads and displays combined 3D point clouds from multiple cameras using saved images and calibration.

    Args:
        serial_nums (list): Ordered list of camera serial numbers.
        t_matrices (dict): Dictionary of transformation matrices between camera pairs.
        dataset_path (str): Path to directory with saved RGB, depth, and intrinsics.
    """
    pcds = []
    aggregated_transformation = np.eye(4)

    for i, serial_num in serial_nums.items():
        
        # Load color and depth images
        color_image = cv2.imread(os.path.join(dataset_path, f'{serial_num}_color.png'))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        depth_image = np.load(os.path.join(dataset_path, f'{serial_num}_depth.npy'), allow_pickle=True)
        depth_image = (depth_image * 0.001).astype(np.float32)  # Convert mm to meters

        # Load intrinsics
        intrinsic_data = np.load(os.path.join(dataset_path, f'{serial_num}_intrinsics.npy'), allow_pickle=True).item()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic_data['width'],
            height=intrinsic_data['height'],
            fx=intrinsic_data['fx'],
            fy=intrinsic_data['fy'],
            cx=intrinsic_data['ppx'],
            cy=intrinsic_data['ppy']
        )

        # Create RGBD image and point cloud
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1.0,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        # Apply transformation
        pcd.transform(aggregated_transformation)
        pcds.append(pcd)

        # Update transformation chain
        if i < len(serial_nums) - 1:
            master = serial_nums[i]
            slave = serial_nums[i + 1]
            t_matrix = t_matrices[f"{master}-{slave}"]

            aggregated_transformation = (aggregated_transformation @ t_matrix)

    o3d.visualization.draw_geometries(pcds)

serial_nums = {0: '213622251272', 1: '213522250729', 2:'037522250789'}

# list(serial_nums.values())
# view_stream_from_dataset("dataset/task2/videos/2025-08-15T19:04:19.412665+00:00.h5", list(serial_nums.values()))