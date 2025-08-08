import h5py
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs
import os

dataset_path = 'dataset/videos/2025-08-08T20:44:12.597771+00:00.h5'  # <- change this to your actual path if needed
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

def mediapipe_combined_pcd(dataset_path, t_matrices, serial_numbers):
    with h5py.File(dataset_path, 'r') as dataset:
        intrinsics = {}
        intrinsics_dict = {}
        for index, serial in serial_numbers.items():
            intrinsic_dict = np.load(os.path.join('data', serial + '_intrinsics.npy'), allow_pickle=True).item()
            intrin = o3d.camera.PinholeCameraIntrinsic(
                width=intrinsic_dict['width'],
                height=intrinsic_dict['height'],
                fx=intrinsic_dict['fx'],
                fy=intrinsic_dict['fy'],
                cx=intrinsic_dict['ppx'],
                cy=intrinsic_dict['ppy'])
            intrinsics[index] = intrin
            intrinsics_dict[index] = intrinsic_dict
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color([1, 0, 0])  # red sphere
        sphere_visible = False
        
        pcds = []
        for i in range(len(serial_numbers)):
            pcds.append(None)

        transform_matrices = [np.eye(4)]
        for t_matrix in t_matrices.values():
            transform_matrices.append(t_matrix @ transform_matrices[-1])
            
        min_frame_count = min(len(list(dataset[f'{serial}/frames/color'])) for serial in serial_numbers.values())

        for frame_index in range(min_frame_count):
            wrist_3d = None
            for i in serial_numbers.keys():
                # PCD of each camera
                serial_num = serial_numbers[i]
                intrinsic = intrinsics[i]
                t_matrix = transform_matrices[i]

                color_image = dataset[f'{serial_num}/frames/color'][str(frame_index)][()]
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                depth_image = (dataset[f'{serial_num}/frames/depth'][str(frame_index)][()] * 0.001).astype(np.float32)
                
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_image),
                    o3d.geometry.Image(depth_image),
                    depth_scale=1.0,
                    depth_trunc=1.5,
                    convert_rgb_to_intensity=False
                )
                
                pcds[i] = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic=t_matrix)

                # MediaPipe hand tracking
                if wrist_3d is None:
                    media_results = media_on_single_frame(dataset_path, serial_num, frame_index)
                    if media_results.multi_hand_landmarks:
                        wrist = media_results.multi_hand_landmarks[0].landmark[0]
                        # Convert to pixel coordinates
                        wrist_x_px = int(wrist.x * intrinsics_dict[i]['width'])
                        wrist_y_px = int(wrist.y * intrinsics_dict[i]['height'])

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
                        X = (wrist_x_px - intrinsics_dict[i]['ppx']) * wrist_depth / intrinsics_dict[i]['fx']
                        Y = (wrist_y_px - intrinsics_dict[i]['ppy']) * wrist_depth / intrinsics_dict[i]['fy']
                        Z = wrist_depth
                        wrist_3d_homogeneous = np.array([X, Y, Z, 1.0])  # Make it [X, Y, Z, 1]
                        wrist_3d_transformed = t_matrix @ wrist_3d_homogeneous  # Matrix-vector multiplication
                        wrist_3d = wrist_3d_transformed[:3]  # Drop the homogeneous component
                    else:
                        continue
            
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

    transform_matrices = {"213522250729-213622251272": [[ 0.3330974,  0.5598185, -0.7587156,  0.5890219],
                                                        [-0.6346248,  0.7282318,  0.258708,  -0.192262 ],
                                                        [ 0.6973503,  0.3953248,  0.5978469,  0.4151737],
                                                        [ 0.        ,  0.        ,  0.        ,  1.        ]], 
                        "213622251272-037522250789": [[-0.2002689, -0.6192168,  0.7592516, -0.4717122],
                                                    [ 0.7569423,  0.3942262,  0.5211756, -0.4153918],
                                                    [-0.6220376,  0.6790849,  0.3897601,  0.4593527],
                                                    [ 0.        ,  0.        ,  0.        ,  1.        ]]}
    serial_nums = {0: '213522250729', 1:'213622251272', 2: '037522250789'}

    mediapipe_combined_pcd('dataset/videos/2025-08-08T21:55:47.951669+00:00.h5', transform_matrices, serial_nums)
