import h5py
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import json
from PIL import Image
import os

# import pyrealsense2 as rs

dataset_path = 'dataset/trial_dataset/check_video_data.h5'  # <- change this to your actual path if needed
serial_numbers = ['213522250729', '213622251272', '217222061083']  # <- update these to your recorded camera serial numbers

def convert_depth_data_to_png(dataset_path = dataset_path, serial_numbers = serial_numbers, output_dir="dataset/trial_dataset/depth_pngs"):
    # os.makedirs(output_dir, exist_ok=True)  # create output folder if needed

    with h5py.File(dataset_path, 'r') as dataset:
        for serial_number in serial_numbers:
            serial_output_dir = os.path.join(output_dir, serial_number)
            os.makedirs(serial_output_dir, exist_ok=True)

            # Load all depth frames to find global min and max
            all_depths = [
                dataset[serial_number]['depth'][frame_key][()]
                for frame_key in dataset[serial_number]['depth'].keys()
            ]
            global_min = min(np.min(frame) for frame in all_depths)
            global_max = max(np.max(frame) for frame in all_depths)

            # Save metadata for this serial_number
            metadata_path = os.path.join(serial_output_dir, "depth_meta.json")
            with open(metadata_path, "w") as f:
                json.dump({"min": float(global_min), "max": float(global_max)}, f)

            # Normalize, convert to uint8, save as PNG
            for frame_key in dataset[serial_number]['depth'].keys():
                depth = dataset[serial_number]['depth'][frame_key][()]
                depth_norm = (depth - global_min) / (global_max - global_min)
                depth_uint16 = (depth_norm * 65535).astype(np.uint16)

                # Save as PNG
                img = Image.fromarray(depth_uint16)
                img.save(os.path.join(serial_output_dir, f"{frame_key}.png"))

    
def media_on_single_frame(dataset_path, serial_number = serial_numbers[1], frame_index=0):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = dataset[serial_number]['color'][str(frame_index)][()]
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        return results

def pointcloud(dataset_path, serial_number = serial_numbers[0], frame_index=50, depth_dir="dataset/trial_dataset/depth_pngs"):
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = dataset[serial_number]['color'][str(frame_index)][()]
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # depth_image = (dataset[serial_number]['depth'][str(frame_index)][()] * 0.001).astype(np.float32)  # Convert depth to meters
        
        serial_input_dir = os.path.join(depth_dir, serial_number)

    with open(os.path.join(serial_input_dir, "depth_meta.json"), "r") as f:
        meta = json.load(f) 
    min_val, max_val = meta["min"], meta["max"]

    frame_path = os.path.join(serial_input_dir, "50.png")
    img = Image.open(frame_path)
    depth = np.array(img)
    
    depth_image_scaled = depth / 65535.0 * (max_val - min_val) + min_val
    depth_image = (depth_image_scaled * 0.001).astype(np.float32)

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
    
    points_float = np.asarray(pcd.points)
    points_int = (points_float).astype(np.uint8) 
    pcd_int = o3d.geometry.PointCloud()
    pcd_int.points = o3d.utility.Vector3dVector(points_int)
        
    
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

        # o3d.visualization.draw_geometries([pcd, sphere, sphere_x, sphere_y, sphere_z, sphere_origin])  
        o3d.io.write_point_cloud("pointclouds/pcd.ply", pcd)
        o3d.io.write_triangle_mesh("pointclouds/sphere.ply", sphere)
    else:
        #  o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud("pcd.ply", pcd_int)


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


if __name__ == "__main__":
    # convert_depth_data_to_png()
    pointcloud(dataset_path)
    # Example usage
    # for serial in serial_numbers:
    #     mediapipe_singleview_pcd(dataset_path, serial)
