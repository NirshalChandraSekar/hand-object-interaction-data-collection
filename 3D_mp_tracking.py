import h5py
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs
import os

dataset_path = '/home/pearl/Downloads/2025-08-08T21_55_47.951669+00_00.h5'  # <- change this to your actual path if needed

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
        depth_trunc=2.0,
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

def render_dataset_with_hand_tracking(dataset_path, t_matrices, serial_numbers):
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
    t_matrices : dict
        Dictionary mapping camera pair strings to 4x4 transformation numpy arrays.
    serial_numbers : dict
        Dictionary mapping camera indices to serial number strings.
    """
    with h5py.File(dataset_path, 'r') as dataset:
        #Setup
        pcd_formatted_intrinsics, intrinsic_data_dict = load_intrinsics(dataset, serial_numbers)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color([1, 0, 0])  # red sphere
        sphere_visible = False

        transform_matrices = [np.eye(4)]
        for i in range(1, len(serial_numbers)):
            master = serial_nums[i-1]
            slave = serial_nums[i]
            t_matrix = t_matrices[f"{master}-{slave}"]
            transform_matrices.append(transform_matrices[-1] @ t_matrix)
            
        min_frame_count = min(len(list(dataset[f'{serial}/frames/color'])) for serial in serial_numbers.values())

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

if __name__ == "__main__":

    transform_matrices = {
    "213522250729-213622251272": np.array([
        [ 0.3330974,  0.5598185, -0.7587156,  0.5890219],
        [-0.6346248,  0.7282318,  0.258708,  -0.192262 ],
        [ 0.6973503,  0.3953248,  0.5978469,  0.4151737],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ]), 
    "213622251272-037522250789": np.array([
        [-0.2002689, -0.6192168,  0.7592516, -0.4717122],
        [ 0.7569423,  0.3942262,  0.5211756, -0.4153918],
        [-0.6220376,  0.6790849,  0.3897601,  0.4593527],
        [ 0.        ,  0.        ,  0.        ,  1.        ]
    ])
}
    serial_nums = {0: '213522250729', 1:'213622251272', 2: '037522250789'}

    render_dataset_with_hand_tracking(dataset_path, transform_matrices, serial_nums)
