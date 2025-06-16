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

def pointcloud(dataset_path, serial_number = serial_numbers[1], frame_index=50):
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

            sphere_x = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_x.translate([1, 0, 0])
            sphere_y = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_y.translate([0, 1, 0])
            sphere_z = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_z.translate([0, 0, 1])
            sphere_x.paint_uniform_color([1, 0, 0])  # Red
            sphere_y.paint_uniform_color([0, 1, 0])  # Green
            sphere_z.paint_uniform_color([0, 0, 1])  # Blue

            sphere_origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_origin.paint_uniform_color([0, 0, 0])  # Black

            o3d.visualization.draw_geometries([pcd, sphere, sphere_x, sphere_y, sphere_z, sphere_origin])  
        else:
             o3d.visualization.draw_geometries([pcd])

def mediapipe_singleview_pcd(dataset_path, serial_number):
    with h5py.File(dataset_path, 'r') as dataset:
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
        view_control = vis.get_view_control() 

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color([1, 0, 0])  # red sphere
        sphere_visible = False
        
        pcd = None
        
        for frame_index in range(len(dataset[serial_number]['color'].keys())):
            
            color_image = dataset[serial_number]['color'][str(frame_index)][()]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            depth_image = (dataset[serial_number]['depth'][str(frame_index)][()] * 0.001).astype(np.float32)
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image),
                o3d.geometry.Image(depth_image),
                depth_scale=1.0,
                depth_trunc=1.5,
                convert_rgb_to_intensity=False
            )
            
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        
            media_results = media_on_single_frame(dataset_path, serial_number, frame_index)
            if media_results.multi_hand_landmarks:
                wrist = media_results.multi_hand_landmarks[0].landmark[0]
                
                # Convert to pixel coordinates
                wrist_x_px = int(wrist.x * intrinsics_['width'])
                wrist_y_px = int(wrist.y * intrinsics_['height'])

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
            
            if frame_index == 0:
                pcd = new_pcd
                vis.add_geometry(pcd)

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

            view_control.set_front([0, -0.0, -1.0])  # Camera looks toward -Z
            view_control.set_lookat([0, 0, 0])       # Look at the origin
            view_control.set_up([0, -1.0, 0])        # Y points down (flip up) 

            vis.poll_events()
            vis.update_renderer()
        
        vis.destroy_window()

'''
Transformation matrix:

[  -0.8387982, -0.3911612, 0.3786957, -0.2170857; 0.3091259, 0.2304021, 0.9226896, -0.6857208; -0.4481726, 0.891015, -0.07234281, 0.5222885 ; 0, 0, 0, 1 ]
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

        transform_matrix = np.array([[-0.8387982, -0.3911612, 0.3786957, -0.2170857],
                                     [0.3091259, 0.2304021, 0.9226896, -0.6857208],
                                     [-0.4481726, 0.891015, -0.07234281, 0.5222885],
                                    [0, 0, 0, 1]], dtype=np.float32)

        transform_inv = np.linalg.inv(transform_matrix)  # Invert the transformation matrix for the slave camera
        
        for frame_index in range(min(len(dataset[serial_master]['color'].keys()), len(dataset[serial_slave]['color'].keys()))):
            color_image_master = dataset[serial_master]['color'][str(frame_index)][()]
            color_image_master = cv2.cvtColor(color_image_master, cv2.COLOR_BGR2RGB)
            
            depth_image_master = (dataset[serial_master]['depth'][str(frame_index)][()] * 0.001).astype(np.float32)
            
            rgbd_image_master = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image_master),
                o3d.geometry.Image(depth_image_master),
                depth_scale=1.0,
                depth_trunc=1.5,
                convert_rgb_to_intensity=False
            )
            
            new_pcd_master = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_master, 
                                                                    intrinsic_master)
            
            color_image_slave = dataset[serial_slave]['color'][str(frame_index)][()]
            color_image_slave = cv2.cvtColor(color_image_slave, cv2.COLOR_BGR2RGB)
            
            depth_image_slave = (dataset[serial_slave]['depth'][str(frame_index)][()] * 0.001).astype(np.float32)
            
            rgbd_image_slave = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image_slave),
                o3d.geometry.Image(depth_image_slave),
                depth_scale=1.0,
                depth_trunc=1.5,
                convert_rgb_to_intensity=False
            )
            
            new_pcd_slave = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_slave, 
                                                                    intrinsic_slave,
                                                                    extrinsic=transform_matrix)
            
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
                wrist_3d_transformed = transform_matrix @ wrist_3d_homogeneous  # Matrix-vector multiplication
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
    # for serial in serial_numbers:
    #     mediapipe_singleview_pcd(dataset_path, serial)
    mediapipe_combined_pcd(dataset_path)
