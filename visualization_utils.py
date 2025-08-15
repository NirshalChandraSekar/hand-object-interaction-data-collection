import h5py
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs
import math
import os
# from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips, AudioClip, VideoFileClip
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
        serial_numbers = list(dataset['cameras'].keys())
        num_frames = min(len(dataset[f'cameras/{serial}/frames/color']) for serial in serial_numbers)
        frame_size = (640, 480)  # Width x Height

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

                # Get timestamp
                timestamp = dataset[f'{serial}/frames/timestamps'][frame_idx]
                timestamp_text = f"{serial} | Time: {timestamp:.2f} ms"

                # Normalize and colorize depth
                depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # Add timestamp to color image
                cv2.putText(color_image, timestamp_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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

def view_stream_from_dataset(dataset_path):
    """
    Displays synchronized color and depth streams from the dataset for all cameras.

    Args:
        dataset_path (str): Path to the HDF5 dataset.
        serial_numbers (list): List of camera serial numbers to visualize.
    """
    with h5py.File(dataset_path, 'r') as dataset:
        serial_numbers = list(dataset['cameras'].keys())
        if("t_matrices" in dataset):
            serial_numbers = serial_numbers[:-1]  # Exclude t_matrices if present
        num_frames = min(len(dataset[f'cameras/{serial_number}/frames/color']) for serial_number in serial_numbers)


        for frame_idx in range(num_frames):
            for serial in serial_numbers:
                color_image = dataset[f'cameras/{serial}/frames/color'][str(frame_idx)][()]
                # depth_image = dataset[f'{serial}/frames/depth'][str(frame_idx)][()]

                cv2.imshow(f'Color Image {serial}', color_image)
                # Normalize depth for visualization
                # depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                # depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                # cv2.imshow(f'Depth Image {serial}', depth_colored)

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
            depth_trunc=3.0,
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


# The following funtions are used to view combined point cloud videos
def load_intrinsics(dataset, serial_numbers):
    """
    Load camera intrinsics from the dataset for each camera serial number.
    
    Parameters:
    -----------
    dataset : h5py.File
        Opened HDF5 dataset file.
    serial_numbers : dict
        Dictionary mapping camera indices to serial number strings.
        
    Returns:
    --------
    cam_intrinsics : dict
        Dictionary mapping camera indices to Open3D PinholeCameraIntrinsic objects.
    intrinsics_dict : dict
        Dictionary mapping camera indices to raw intrinsic data groups from HDF5.
    """
    intrinsics_dict = {}
    cam_intrinsics = {}
    for index, serial in serial_numbers.items():
        intrinsic_data = dataset[serial]['params']['intrinsics']
        intrin = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic_data['width'][()],
            height=intrinsic_data['height'][()],
            fx=intrinsic_data['fx'][()],
            fy=intrinsic_data['fy'][()],
            cx=intrinsic_data['ppx'][()],
            cy=intrinsic_data['ppy'][()])
        cam_intrinsics[index] = intrin
        intrinsics_dict[index] = intrinsic_data
    return cam_intrinsics, intrinsics_dict

def create_pcd(index, serial_number, frame_index, dataset, intrinsic, transform_matrix):
    """
    Create a transformed colored point cloud (PCD) from a single camera frame.
    
    Parameters:
    -----------
    index : int
        Camera index.
    serial_number : str
        Camera serial number in the dataset.
    frame_index : int
        Index of the frame to read.
    dataset : h5py.File
        Opened HDF5 dataset file.
    intrinsic : o3d.camera.PinholeCameraIntrinsic
        Camera intrinsic parameters.
    transform_matrix : np.ndarray
        4x4 transformation matrix to apply to the point cloud.
        
    Returns:
    --------
    depth_image : np.ndarray
        Depth image in meters, useful for further processing.
    pcd : o3d.geometry.PointCloud
        Transformed point cloud for the given frame.
    """
    color_image = dataset[f'{serial_number}/frames/color'][str(frame_index)][()]
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    depth_image = (dataset[f'{serial_number}/frames/depth'][str(frame_index)][()] * 0.001).astype(np.float32)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1.0,
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # Apply transformation
    pcd.transform(transform_matrix)
    return depth_image, pcd

def media_on_single_frame(dataset, serial_number, frame_index=0):
    """
    Run MediaPipe hand tracking on a single color frame from the dataset.
    
    Parameters:
    -----------
    dataset : h5py.File
        Opened HDF5 dataset file.
    serial_number : str
        Camera serial number identifying the dataset group.
    frame_index : int, optional
        Index of the frame to process (default is 0).
        
    Returns:
    --------
    mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None
        MediaPipe hand tracking results for the frame.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    color_image = dataset[f'{serial_number}/frames/color'][str(frame_index)][()]
    results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    return results

def process_meidapipe_results(results, intrinsic, depth_image, t_matrix):
    """
    Process MediaPipe hand tracking results to extract 3D wrist coordinates.
    
    This involves:
    - Converting wrist landmark normalized coordinates to pixel coordinates.
    - Retrieving the depth value at the wrist pixel (with neighborhood fallback).
    - Backprojecting pixel + depth to 3D camera coordinates.
    - Transforming the 3D point to world coordinates.
    
    Parameters:
    -----------
    results : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        MediaPipe hand tracking landmarks from a frame.
    intrinsic : h5py.Group
        Raw intrinsic camera parameters from dataset.
    depth_image : np.ndarray
        Depth image aligned with color frame.
    t_matrix : np.ndarray
        4x4 transformation matrix to convert camera coords to world coords.
        
    Returns:
    --------
    np.ndarray or None
        3D wrist coordinates in world space (x, y, z), or None if no valid depth found.
    """
    wrist = results.multi_hand_landmarks[0].landmark[0]
    # Convert to pixel coordinates
    wrist_x_px = int(wrist.x * intrinsic['width'][()])
    wrist_y_px = int(wrist.y * intrinsic['height'][()])

    # Get depth at pixel
    wrist_depth = depth_image[wrist_y_px, wrist_x_px] # Weird indexing
    if wrist_depth == 0.0:
        patch = depth_image[max(0, wrist_y_px-1):wrist_y_px+2, max(0, wrist_x_px-1):wrist_x_px+2]
        valid = patch[patch > 0]
        if valid.size == 0:
            print("No valid depth around wrist.")
            return None
        wrist_depth = np.mean(valid) 

    # Backproject pixel + depth to 3D camera coords
    X = (wrist_x_px - intrinsic['ppx'][()]) * wrist_depth / intrinsic['fx'][()]
    Y = (wrist_y_px - intrinsic['ppy'][()]) * wrist_depth / intrinsic['fy'][()]
    Z = wrist_depth
    wrist_3d_homogeneous = np.array([X, Y, Z, 1.0])  # Make it [X, Y, Z, 1]
    wrist_3d_transformed = t_matrix @ wrist_3d_homogeneous  # Matrix-vector multiplication
    return wrist_3d_transformed[:3]  # Drop the homogeneous component

def render_dataset_with_hand_tracking(dataset_path, use_offline_calib=False):
    """
    Visualize a multi-camera RGB-D dataset with MediaPipe hand tracking.
    
    For each frame, this function:
    - Loads camera intrinsics.
    - Creates transformed point clouds from each camera.
    - Runs MediaPipe hand tracking to locate the wrist in 3D.
    - Visualizes the point clouds and a red sphere indicating wrist position.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the HDF5 dataset file.
    serial_numbers : dict
        Dictionary mapping camera indices to serial number strings.
    """
    with h5py.File(dataset_path, 'r') as dataset:
        #Setup
        serial_list = list(dataset.keys())[:3]  # example to get first 3 serial strings
        serial_numbers = dict(enumerate(serial_list))
        print("Cameras in dataset:", serial_numbers)

        pcd_formatted_intrinsics, intrinsic_data_dict = load_intrinsics(dataset, serial_numbers)
        
        transform_matrices = [np.eye(4)]

        if use_offline_calib or 't_matrices' not in dataset:
            calib.write_camera_intrinsics_to_file()
            calib.run_calibrations(serial_numbers, dataset_path)
            t_matrix = calib.get_transformation_matrices(serial_numbers)
        else:
            t_matrix = dataset['t_matrices']

        for i in range(1, len(serial_numbers)):
            key = f'{serial_numbers[i-1]}-{serial_numbers[i]}'
            transform_matrices.append(transform_matrices[-1] @ t_matrix[key][()])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color([1, 0, 0])  # red sphere
        sphere_visible = False
            
        min_frame_count = min(len(list(dataset[f'cameras/{serial}/frames/color'])) for serial in serial_numbers.values())
        # Visualization loop
        for frame_index in range(min_frame_count):
            wrist_3d = None
            pcds = []

            # Process each camera
            for serial_index, serial_num in serial_numbers.items():
                # PCD
                intrinsic = pcd_formatted_intrinsics[serial_index]
                t_matrix = transform_matrices[serial_index]
                depth_image, pcd = create_pcd(serial_index, serial_num, frame_index, dataset, intrinsic, t_matrix)
                pcds.append(pcd)
    
                # MediaPipe wrist tracking
                if wrist_3d is None:
                    media_results = media_on_single_frame(dataset, serial_num, frame_index)
                    if media_results.multi_hand_landmarks:
                        wrist_3d = process_meidapipe_results(media_results, intrinsic_data_dict[serial_index], depth_image, t_matrix)
                    else:
                        continue

            # Visualization
            if frame_index == 0:
                geometries = []
                for j in range(len(pcds)):
                    vis.add_geometry(pcds[j])
                    geometries.append(pcds[j])

                view_control = vis.get_view_control()
                view_control.rotate(0, 180, 0, 0)
                
                if wrist_3d is not None:
                    sphere.translate(wrist_3d, relative=False)
                    vis.add_geometry(sphere)
                    sphere_visible = True
            else:
                for j in range(len(pcds)):
                    geometries[j].points = pcds[j].points
                    geometries[j].colors = pcds[j].colors
                    vis.update_geometry(geometries[j])
                
                if wrist_3d is not None:
                    if not sphere_visible:
                        vis.add_geometry(sphere)
                        sphere_visible = True
                    sphere_center = sphere.get_center()
                    translation = wrist_3d - sphere_center
                    sphere.translate(translation, relative=True)
                    vis.update_geometry(sphere)
                else:
                    if sphere_visible:
                        vis.remove_geometry(sphere)
                        sphere_visible = False
            
            # Orient the view
            view_control.set_front([0, -0.0, -1.0])  
            view_control.set_lookat([0, 0, 0])       
            view_control.set_up([0, -1.0, 0])        
            view_control.set_zoom(0.5)

            # Update with each frame's point clouds
            vis.poll_events()
            vis.update_renderer()
    
    vis.destroy_window()


# The following functions are used to save synchronized multi-camera video with audio alignment

def save_video_fps(video_path, output_path='output_video.mp4'):
    """
    Save a .mp4 video from RealSense frames in an HDF5 dataset, using timestamps to determine FPS.
    """
    with h5py.File(video_path, 'r') as data:
        serial_num = list(data['cameras'].keys())[0]
        frames_group = data['cameras'][serial_num]['frames']
        color_frames = frames_group['color']
        timestamps = frames_group['timestamps'][:]  # Load all timestamps

        # Estimate FPS from timestamps
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)  # Should be in milliseconds or microseconds
            avg_interval = np.mean(intervals)
            
            # Normalize timestamp units if needed (heuristic)
            if avg_interval > 1000:  # Microseconds?
                avg_interval /= 1000.0  # Convert to milliseconds
            fps = 1.0 / avg_interval  # For seconds
  # ms to fps
        else:
            fps = 30  # Fallback

        print(f"Estimated FPS: {fps:.2f}")

        # Get frame dimensions from the first frame
        first_frame = color_frames['0'][:]
        frame_height, frame_width = first_frame.shape[:2]

        # Initialize VideoWriter
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )

        # Write frames in order
        frame_keys = sorted(color_frames.keys(), key=lambda x: int(x))
        for key in frame_keys:
            frame = color_frames[key][:]
            out.write(frame.astype(np.uint8))

        out.release()
        print(f"Saved video to {output_path}")

save_video_fps('dataset/task1/video/1755276910.3663642.h5')