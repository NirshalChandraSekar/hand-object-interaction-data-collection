import h5py
import cv2
import numpy as np
# import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs
import os

dataset_path = 'dataset/trial_dataset/check_video_data.h5'  # <- change this to your actual path if needed
serial_numbers = ['213522250729', '213622251272']  # <- update these to your recorded camera serial numbers

'''
Camera 213622251272 Intrinsics:
  Width: 1280
  Height: 800
  Focal Length (fx, fy): (645.9054565429688, 645.9054565429688)
  Principal Point (ppx, ppy): (646.1998901367188, 403.8904113769531)
  Distortion Model: distortion.brown_conrady
  Distortion Coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
Camera 213522250729 Intrinsics:
  Width: 1280
  Height: 720
  Focal Length (fx, fy): (649.6065063476562, 649.6065063476562)
  Principal Point (ppx, ppy): (650.9663696289062, 364.2262878417969)
  Distortion Model: distortion.brown_conrady
  Distortion Coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
'''

serial_numbers = ['213622251272', '213522250729']  # <- update these to your recorded camera serial numbers

def save_video_from_dataset(dataset_path, output_video_path, serial_numbers):
    with h5py.File(dataset_path, 'r') as dataset:
        num_frames = len(dataset[serial_numbers[0]]['color'])
        frame_size = (640, 480)  # Assuming all images are 640x480

        # Define output dimensions based on layout (RGB and depth side by side for each camera)
        output_width = frame_size[0] * 2  # Two columns (RGB + depth)
        output_height = frame_size[1] * len(serial_numbers)  # Rows = number of cameras

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (output_width, output_height))

        for frame_idx in range(num_frames):
            rows = []
            
            # Process each camera
            for serial in serial_numbers:
                camera_row = []
                
                # Get color and depth images
                color_image = dataset[serial]['color'][str(frame_idx)][()]
                depth_image = dataset[serial]['depth'][str(frame_idx)][()]

                # Normalize depth for visualization
                depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # Resize images to ensure they fit in the frame
                color_image_resized = cv2.resize(color_image, frame_size)
                depth_colored_resized = cv2.resize(depth_colored, frame_size)

                # Create a row with RGB and depth side by side
                camera_row = np.hstack([color_image_resized, depth_colored_resized])
                rows.append(camera_row)
            
            # Stack all camera rows vertically
            combined_frame = np.vstack(rows)
            out.write(combined_frame)
            
            # Optional: Display progress
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{num_frames} frames")

        out.release()
        print(f"Video saved to {output_video_path}")


def run_mediapipe_on_videos(dataset_path, output_video_path, serial_numbers):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    with h5py.File(dataset_path, 'r') as dataset:
        num_frames = len(dataset[serial_numbers[0]]['color'])
        frame_size = (640, 480)  # Assuming all images are 640x480

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


def view_recorded_stream(dataset_path, serial_numbers):
    with h5py.File(dataset_path, 'r') as dataset:
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


def get_camera_intrinsics(serial_numbers=None):
    # Create a context object
    ctx = rs.context()
    
    # If no serial numbers are provided, query all connected devices
    if serial_numbers is None:
        serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in ctx.query_devices()]
    
    for serial in serial_numbers:
        try:
            # Get the device by serial number
            device = next(dev for dev in ctx.query_devices() if dev.get_info(rs.camera_info.serial_number) == serial)
            depth_sensor = device.first_depth_sensor()
            
            # Get the intrinsics of the first depth stream
            intrinsics = depth_sensor.get_stream_profiles()[0].as_video_stream_profile().get_intrinsics()
            
            # Print the intrinsics
            print(f"Camera {serial} Intrinsics:")
            print(f"  Width: {intrinsics.width}")
            print(f"  Height: {intrinsics.height}")
            print(f"  Focal Length (fx, fy): ({intrinsics.fx}, {intrinsics.fy})")
            print(f"  Principal Point (ppx, ppy): ({intrinsics.ppx}, {intrinsics.ppy})")
            print(f"  Distortion Model: {intrinsics.model}")
            print(f"  Distortion Coefficients: {intrinsics.coeffs}")
        except StopIteration:
            print(f"Camera with serial {serial} not found.")


def view_point_cloud_single_camera(dataset_path = 'dataset/trial_dataset/check_video_data.h5', serial_number = serial_numbers[0]):
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = dataset[serial_number]['color'][str(20)][()]
        cv2.imshow('Color Image', color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Open3D
        depth_image = (dataset[serial_number]['depth'][str(20)][()] * 0.001).astype(np.float32)  # Convert depth to meters
        intrinsics_ = np.load('data/213622251272_intrinsics.npy', allow_pickle=True).item()
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


'''
Transformation matrix:

[ -0.986436, -0.1571164, -0.04752302, -0.04698686; 0.1640189, -0.9548615, -0.2476637, -0.07646833; -0.006465859, -0.2520991, 0.9676798, -0.07681673; 0, 0, 0, 1 ]
'''

def view_combined_point_cloud(dataset_path, serial_numbers = serial_numbers):
    with h5py.File(dataset_path, 'r') as dataset:
        
        color_image_master = cv2.imread(os.path.join('data', serial_numbers[0] + '_color.png'))
        color_image_master = cv2.cvtColor(color_image_master, cv2.COLOR_BGR2RGB)
        depth_image_master = (np.load(os.path.join('data', serial_numbers[0] + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
        intrinsic_master = np.load(os.path.join('data', serial_numbers[0] + '_intrinsics.npy'), allow_pickle=True).item()
        intrinsic_master = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic_master['width'],
            height=intrinsic_master['height'],
            fx=intrinsic_master['fx'],
            fy=intrinsic_master['fy'],
            cx=intrinsic_master['ppx'],
            cy=intrinsic_master['ppy'])
        rgbd_image_master = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image_master),
            o3d.geometry.Image(depth_image_master),
            depth_scale=1,  # Adjust based on your depth image scale
            depth_trunc=3.0,  # Truncate depth values beyond this distance
            convert_rgb_to_intensity=False)
        pcd_master = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_master, 
                                                                    intrinsic_master,
                                                                    extrinsic=np.eye(4))
        

        color_image_slave = cv2.imread(os.path.join('data', serial_numbers[1] + '_color.png'))
        color_image_slave = cv2.cvtColor(color_image_slave, cv2.COLOR_BGR2RGB)
        depth_image_slave = (np.load(os.path.join('data', serial_numbers[1] + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
        intrinsic_slave = np.load(os.path.join('data', serial_numbers[1] + '_intrinsics.npy'), allow_pickle=True).item()
        intrinsic_slave = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic_slave['width'],
            height=intrinsic_slave['height'],
            fx=intrinsic_slave['fx'],
            fy=intrinsic_slave['fy'],
            cx=intrinsic_slave['ppx'],
            cy=intrinsic_slave['ppy'])
        rgbd_image_slave = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image_slave),
            o3d.geometry.Image(depth_image_slave),
            depth_scale=1,  # Adjust based on your depth image scale
            depth_trunc=3.0,  # Truncate depth values beyond this distance
            convert_rgb_to_intensity=False)
        transform_matrix = np.array([
            [-0.986436, -0.1571164, -0.04752302, -0.04698686],
            [0.1640189, -0.9548615, -0.2476637, -0.07646833],
            [-0.006465859, -0.2520991, 0.9676798, -0.07681673],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # transform_matrix = np.linalg.inv(transform_matrix)  # Invert the transformation matrix for the slave camera

        pcd_slave = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_slave, 
                                                                    intrinsic_slave,
                                                                    extrinsic=transform_matrix)
        
        o3d.visualization.draw_geometries([pcd_master, pcd_slave])   



# view_point_cloud_single_camera(dataset_path)
view_combined_point_cloud(dataset_path)







