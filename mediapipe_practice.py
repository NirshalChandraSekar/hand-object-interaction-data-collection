import h5py
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs
import os

dataset_path = 'dataset/trial_dataset/check_video_data.h5'  # <- change this to your actual path if needed
serial_numbers = ['213522250729', '213622251272']  # <- update these to your recorded camera serial numbers

def media_on_single_frame(dataset_path, serial_number = serial_numbers[1], frame_index=0):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = dataset[serial_number]['color'][str(frame_index)][()]
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        return results

def pointcloud(dataset_path, serial_number = serial_numbers[1], frame_index=0):
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = dataset[serial_number]['color'][str(frame_index)][()]
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = (dataset[serial_number]['depth'][str(frame_index)][()] * 0.001).astype(np.float32)  # Convert depth to meters

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
        pcd =  o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                                    intrinsic,
                                                                    extrinsic=np.eye(4))
        
        media_results = media_on_single_frame(dataset_path, serial_number, frame_index)
        if media_results.multi_hand_landmarks:
            wrist = media_results.multi_hand_landmarks[0].landmark[0]
 
            # Convert to pixel coordinates
            wrist_x_px = int(wrist.x * intrinsics_['width'])
            wrist_y_px = int(wrist.y * intrinsics_['height'])

            # Get depth at pixel
            wrist_depth = depth_image[wrist_y_px, wrist_x_px] # Weird indexing
            if wrist_depth == 0.0:
                print("Invalid depth at wrist, trying average in 3x3 patch")
                patch = depth_image[max(0, wrist_y_px-1):wrist_y_px+2, max(0, wrist_x_px-1):wrist_x_px+2]
                valid = patch[patch > 0]
                if valid.size == 0:
                    print("No valid depth around wrist.")
                    return
                wrist_depth = np.mean(valid)

            # Convert to 3D using intrinsics
            x = (wrist_x_px - intrinsics_['ppx']) * wrist_depth / intrinsics_['fx']
            y = (wrist_y_px - intrinsics_['ppy']) * wrist_depth / intrinsics_['fy']
            z = wrist_depth
            wrist_3d = np.array([x, y, z])

            # Create and overlay sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.paint_uniform_color([1, .5, .5])  # Red
            sphere.translate(wrist_3d)

            o3d.visualization.draw_geometries([pcd, sphere])
        else:
             o3d.visualization.draw_geometries([pcd])

def all_frames_pcd(dataset_path, serial_number, start_frame=0, end_frame=10):
    intrinsics_ = np.load(f'data/{serial_number}_intrinsics.npy', allow_pickle=True).item()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics_['width'],
        height=intrinsics_['height'],
        fx=intrinsics_['fx'],
        fy=intrinsics_['fy'],
        cx=intrinsics_['ppx'],
        cy=intrinsics_['ppy'])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    sphere.paint_uniform_color([1, 0, 0])  # red sphere
    sphere_visible = False
    
    pcd = None
    
    for frame_index in range(start_frame, end_frame):
        with h5py.File(dataset_path, 'r') as dataset:
            color_image = dataset[serial_number]['color'][str(frame_index)][()]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            depth_image = (dataset[serial_number]['depth'][str(frame_index)][()] * 0.001).astype(np.float32)
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image),
                o3d.geometry.Image(depth_image),
                depth_scale=1.0,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False
            )
            
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        
            media_results = media_on_single_frame(dataset_path, serial_number, frame_index)
            if media_results.multi_hand_landmarks:
                wrist = media_results.multi_hand_landmarks[0].landmark[0]
                
                # Convert to pixel coordinates
                wrist_x_px = int(wrist.x * intrinsics_['width'])
                wrist_y_px = int(wrist.y * intrinsics_['height'])

                # Get depth at pixel
                wrist_depth = depth_image[wrist_y_px, wrist_x_px] # Weird indexing
                if wrist_depth == 0.0:
                    patch = depth_image[max(0, wrist_y_px-1):wrist_y_px+2, max(0, wrist_x_px-1):wrist_x_px+2]
                    valid = patch[patch > 0]
                    if valid.size == 0:
                        print("No valid depth around wrist.")
                        continue
                    wrist_depth = np.mean(valid) 
                
                    # Backproject pixel + depth to 3D camera coords
                X = (wrist_x_px - intrinsics_['ppx']) * wrist_depth / intrinsics_['fx']
                Y = (wrist_y_px - intrinsics_['ppy']) * wrist_depth / intrinsics_['fy']
                Z = wrist_depth
                wrist_3d = np.array([X, Y, Z])
            else:
                wrist_3d = None
            
            if frame_index == start_frame:
                pcd = new_pcd
                vis.add_geometry(pcd)
                view_control = vis.get_view_control()
                view_control.rotate(0, 180, 0, 0)
                
                if wrist_3d is not None:
                    sphere.translate(wrist_3d, relative=False)
                    vis.add_geometry(sphere)
                    sphere_visible = True
            else:
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
                vis.update_geometry(pcd)
                
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
            
            vis.poll_events()
            vis.update_renderer()
    
    vis.destroy_window()


if __name__ == "__main__":
    # Example usage
    with h5py.File(dataset_path, 'r') as dataset:
        for serial in serial_numbers:
            all_frames_pcd(dataset_path, serial, 0, len(dataset[serial]['color'].keys()))
