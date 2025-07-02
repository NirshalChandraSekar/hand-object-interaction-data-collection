import h5py
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs
import os

dataset_path = 'dataset/trial_dataset/2025-07-01T21:26:31.528537+00:00.h5'  # <- change this to your actual path if needed
serial_numbers = ['213522250729', '213622251272']  # <- update these to your recorded camera serial numbers

def media_on_single_frame(dataset_path, serial_number, frame_index=0):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = dataset[f'{serial_number}/frames/color'][str(frame_index)][()]
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        return results


'''
Transformation matrix:

[ -0.9872021, 0.06064954, -0.1474909, -0.05247459; -0.159466, -0.3662032, 0.9167692, -0.9855934; 0.001589981, 0.9285563, 0.3711881, 0.6698481 ]
'''

def mediapipe_combined_pcd(dataset_path, serial_numbers = serial_numbers):
    with h5py.File(dataset_path, 'r') as dataset:
        serial_master = serial_numbers[0]
        serial_slave = serial_numbers[1]

        intrinsics_master_ = np.load(os.path.join('data', serial_master + '_intrinsics.npy'), allow_pickle=True).item()
        intrinsic_master = o3d.camera.PinholeCameraIntrinsic(
                width=intrinsics_master_['width'],
                height=intrinsics_master_['height'],
                fx=intrinsics_master_['fx'],
                fy=intrinsics_master_['fy'],
                cx=intrinsics_master_['ppx'],
                cy=intrinsics_master_['ppy'])
        
        intrinsics_slave_ = np.load(os.path.join('data', serial_slave + '_intrinsics.npy'), allow_pickle=True).item()
        intrinsic_slave = o3d.camera.PinholeCameraIntrinsic(
                width=intrinsics_slave_['width'],
                height=intrinsics_slave_['height'],
                fx=intrinsics_slave_['fx'],
                fy=intrinsics_slave_['fy'],
                cx=intrinsics_slave_['ppx'],
                cy=intrinsics_slave_['ppy'])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color([1, 0, 0])  # red sphere
        sphere_visible = False
        
        pcd_master = None
        pcd_slave = None

        transform_matrix = np.array([[-0.9872021, 0.06064954, -0.1474909, -0.05247459],
                                       [-0.159466, -0.3662032, 0.9167692, -0.9855934],
                                       [ 0.001589981, 0.9285563, 0.3711881, 0.6698481],
                                       [0, 0, 0, 1]], dtype=np.float32)
        transform_inv = np.linalg.inv(transform_matrix)  # Invert the transformation matrix for the slave camera
        min_frame_count = min(len(list(dataset[f'{serial}/frames/color'])) for serial in serial_numbers)
        for frame_index in range(min_frame_count):
            color_image_master = dataset[f'{serial_master}/frames/color'][str(frame_index)][()]
            color_image_master = cv2.cvtColor(color_image_master, cv2.COLOR_BGR2RGB)
            
            depth_image_master = (dataset[f'{serial_master}/frames/depth'][str(frame_index)][()] * 0.001).astype(np.float32)
            
            rgbd_image_master = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image_master),
                o3d.geometry.Image(depth_image_master),
                depth_scale=1.0,
                depth_trunc=1.5,
                convert_rgb_to_intensity=False
            )
            
            new_pcd_master = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_master, 
                                                                    intrinsic_master)
            
            color_image_slave = dataset[f'{serial_slave}/frames/color'][str(frame_index)][()]
            color_image_slave = cv2.cvtColor(color_image_slave, cv2.COLOR_BGR2RGB)
            
            depth_image_slave = (dataset[f'{serial_slave}/frames/depth'][str(frame_index)][()] * 0.001).astype(np.float32)
            
            rgbd_image_slave = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image_slave),
                o3d.geometry.Image(depth_image_slave),
                depth_scale=1.0,
                depth_trunc=1.5,
                convert_rgb_to_intensity=False
            )
            
            new_pcd_slave = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_slave, 
                                                                    intrinsic_slave,
                                                                    extrinsic=transform_inv)
            
            media_results_master = media_on_single_frame(dataset_path, serial_master, frame_index)
            media_results_slave = media_on_single_frame(dataset_path, serial_slave, frame_index)

            # Uses the slave wrist position if the master wrist is not detected
            if media_results_master.multi_hand_landmarks:
                wrist = media_results_master.multi_hand_landmarks[0].landmark[0]
                
                # Convert to pixel coordinates
                wrist_x_px = int(wrist.x * intrinsics_master_['width'])
                wrist_y_px = int(wrist.y * intrinsics_master_['height'])

                # Get depth at pixel
                wrist_depth = depth_image_master[wrist_y_px, wrist_x_px] # Weird indexing
                if wrist_depth == 0.0:
                    patch = depth_image_master[max(0, wrist_y_px-1):wrist_y_px+2, max(0, wrist_x_px-1):wrist_x_px+2]
                    valid = patch[patch > 0]
                    if valid.size == 0:
                        print("No valid depth around wrist.")
                        continue
                    wrist_depth = np.mean(valid) 
                    
                # Backproject pixel + depth to 3D camera coords
                X = (wrist_x_px - intrinsics_master_['ppx']) * wrist_depth / intrinsics_master_['fx']
                Y = (wrist_y_px - intrinsics_master_['ppy']) * wrist_depth / intrinsics_master_['fy']
                Z = wrist_depth
                wrist_3d = np.array([X, Y, Z])
            elif media_results_slave.multi_hand_landmarks:
                wrist = media_results_slave.multi_hand_landmarks[0].landmark[0]
                
                # Convert to pixel coordinates
                wrist_x_px = int(wrist.x * intrinsics_slave_['width'])
                wrist_y_px = int(wrist.y * intrinsics_slave_['height'])

                # Get depth at pixel
                wrist_depth = depth_image_slave[wrist_y_px, wrist_x_px] # Weird indexing
                if wrist_depth == 0.0:
                    patch = depth_image_slave[max(0, wrist_y_px-1):wrist_y_px+2, max(0, wrist_x_px-1):wrist_x_px+2]
                    valid = patch[patch > 0]
                    if valid.size == 0:
                        print("No valid depth around wrist.")
                        continue
                    wrist_depth = np.mean(valid) 
                
                    # Backproject pixel + depth to 3D camera coords
                X = (wrist_x_px - intrinsics_slave_['ppx']) * wrist_depth / intrinsics_slave_['fx']
                Y = (wrist_y_px - intrinsics_slave_['ppy']) * wrist_depth / intrinsics_slave_['fy']
                Z = wrist_depth
                wrist_3d_homogeneous = np.array([X, Y, Z, 1.0])  # Make it [X, Y, Z, 1]
                wrist_3d_transformed = transform_inv @ wrist_3d_homogeneous  # Matrix-vector multiplication
                wrist_3d = wrist_3d_transformed[:3]  # Drop the homogeneous component
            else:
                wrist_3d = None
            
            if frame_index == 0:
                pcd_master = new_pcd_master
                pcd_slave = new_pcd_slave

                vis.add_geometry(pcd_master)
                vis.add_geometry(pcd_slave)

                view_control = vis.get_view_control()
                view_control.rotate(0, 180, 0, 0)
                
                if wrist_3d is not None:
                    sphere.translate(wrist_3d, relative=False)
                    vis.add_geometry(sphere)
                    sphere_visible = True
            else:
                pcd_master.points = new_pcd_master.points
                pcd_slave.points = new_pcd_slave.points
                pcd_master.colors = new_pcd_master.colors
                pcd_slave.colors = new_pcd_slave.colors
                vis.update_geometry(pcd_master)
                vis.update_geometry(pcd_slave)
                
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
    mediapipe_combined_pcd('dataset/demo_data/2025-07-02T19:28:16.042706+00:00.h5')
