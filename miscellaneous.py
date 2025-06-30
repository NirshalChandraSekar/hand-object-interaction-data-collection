import h5py
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs
import os

dataset_path = 'dataset/trial_dataset/check_video_data.h5'  # <- change this to your actual path if needed
serial_numbers = ['213522250729', '213622251272', '217222061083']  # <- update these to your recorded camera serial numbers

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

# serial_numbers = ['213622251272', '213522250729']  # <- update these to your recorded camera serial numbers

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
        num_frames = len(dataset[f'{serial_numbers[1]}/frames/color']['0'][()])

        for frame_idx in range(num_frames):
            for serial in serial_numbers:
                # color_image = dataset[serial]['frames']['color'][str(frame_idx)][()]
                # depth_image = dataset[serial]['frames']['depth'][str(frame_idx)][()]

                # For the new dataset structure
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



def view_point_cloud_single_camera(dataset_path = 'dataset/trial_dataset/check_video_data.h5', serial_number = serial_numbers[1]):
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = (dataset[f'{serial_numbers[1]}/frames/color'][str(20)][()]) #.astype(np.float32) * 0.001
        cv2.imshow('Color Image', color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Open3D
        depth_image = (dataset[f'{serial_numbers[1]}/frames/depth'][str(20)][()]).astype(np.float32) * 0.001  # Convert depth to meters
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



'''
Transformation matrix:

 [ -0.9801651, -0.1587493, -0.1186386, -0.07690796; -0.01549144, -0.5354281, 0.8444387, -0.820531; -0.1975765, 0.8295272, 0.5223486, 0.3569262 ]
'''

def view_combined_point_cloud():
    color_image_72 = cv2.imread(os.path.join('data', serial_numbers[1] + '_color.png'))
    color_image_72 = cv2.cvtColor(color_image_72, cv2.COLOR_BGR2RGB)
    depth_image_72 = (np.load(os.path.join('data', serial_numbers[1] + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
    intrinsic_72 = np.load(os.path.join('data', serial_numbers[1] + '_intrinsics.npy'), allow_pickle=True).item()
    intrinsic_72 = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic_72['width'],
        height=intrinsic_72['height'],
        fx=intrinsic_72['fx'],
        fy=intrinsic_72['fy'],
        cx=intrinsic_72['ppx'],
        cy=intrinsic_72['ppy'])
    rgbd_image_72 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image_72),
        o3d.geometry.Image(depth_image_72),
        depth_scale=1,  # Adjust based on your depth image scale
        depth_trunc=1.5,  # Truncate depth values beyond this distance
        convert_rgb_to_intensity=False)
    pcd_72 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_72, 
                                                                intrinsic_72,
                                                                extrinsic=np.eye(4))
    
    # color_image_83 = cv2.imread(os.path.join('data', serial_numbers[2] + '_color.png'))
    # color_image_83 = cv2.cvtColor(color_image_83, cv2.COLOR_BGR2RGB)
    # depth_image_83 = (np.load(os.path.join('data', serial_numbers[2] + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
    # intrinsic_83 = np.load(os.path.join('data', serial_numbers[2] + '_intrinsics.npy'), allow_pickle=True).item()
    # intrinsic_83 = o3d.camera.PinholeCameraIntrinsic(
    #     width=intrinsic_83['width'],
    #     height=intrinsic_83['height'],
    #     fx=intrinsic_83['fx'],
    #     fy=intrinsic_83['fy'],
    #     cx=intrinsic_83['ppx'],
    #     cy=intrinsic_83['ppy'])
    # rgbd_image_83 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     o3d.geometry.Image(color_image_83),
    #     o3d.geometry.Image(depth_image_83),
    #     depth_scale=1,  # Adjust based on your depth image scale
    #     depth_trunc=1.5,  # Truncate depth values beyond this distance
    #     convert_rgb_to_intensity=False)

    # transform_matrix_83_72 = np.array([[-0.9798484, -0.1595772, -0.120134, -0.07619032],
    #                                   [-0.01650491, -0.5347027, 0.8448791, -0.8207434],
    #                                   [-0.1990594, 0.8298362, 0.5212938, 0.3569585],
    #                                 [0, 0, 0, 1]], dtype=np.float32)
    # transform_matrix_83_72 = np.linalg.inv(transform_matrix_83_72)  # Invert the transformation matrix for the slave camera
    
    # pcd_83 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_83, 
    #                                                         intrinsic_83,
    #                                                         extrinsic=transform_matrix_83_72)
    

    color_image_29 = cv2.imread(os.path.join('data', serial_numbers[0] + '_color.png'))
    color_image_29 = cv2.cvtColor(color_image_29, cv2.COLOR_BGR2RGB)
    depth_image_29 = (np.load(os.path.join('data', serial_numbers[0] + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
    intrinsic_29 = np.load(os.path.join('data', serial_numbers[0] + '_intrinsics.npy'), allow_pickle=True).item()
    intrinsic_29 = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic_29['width'],
        height=intrinsic_29['height'],
        fx=intrinsic_29['fx'],
        fy=intrinsic_29['fy'],
        cx=intrinsic_29['ppx'],
        cy=intrinsic_29['ppy'])
    rgbd_image_29 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image_29),
        o3d.geometry.Image(depth_image_29),
        depth_scale=1,  # Adjust based on your depth image scale
        depth_trunc=1.5,  # Truncate depth values beyond this distance
        convert_rgb_to_intensity=False)
    transform_matrix_29_83 = np.array([ [-0.9801651, -0.1587493, -0.1186386, -0.07690796],
                                     [-0.01549144, -0.5354281, 0.8444387, -0.820531],
                                     [-0.1975765, 0.8295272, 0.5223486, 0.3569262],
                                    [0, 0, 0, 1]], dtype=np.float32)

    # transform_matrix_29_83 = np.linalg.inv(transform_matrix_29_83) 
    # transform_29_72 = transform_matrix_29_83 @ transform_matrix_83_72

    pcd_29 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_29, 
                                                                intrinsic_29,
                                                                extrinsic=transform_matrix_29_83)
    
    o3d.io.write_point_cloud("pointclouds/pcd_72.ply", pcd_72)
    # o3d.io.write_point_cloud("pointclouds/pcd_83.ply", pcd_83)
    o3d.io.write_point_cloud("pointclouds/pcd_29.ply", pcd_29)
    # o3d.visualization.draw_geometries([pcd_72, pcd_29])

def three_camera_pointcloud():
    color_image_72 = cv2.imread(os.path.join('data', serial_numbers[1] + '_color.png'))
    color_image_72 = cv2.cvtColor(color_image_72, cv2.COLOR_BGR2RGB)
    depth_image_72 = (np.load(os.path.join('data', serial_numbers[1] + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
    intrinsic_72 = np.load(os.path.join('data', serial_numbers[1] + '_intrinsics.npy'), allow_pickle=True).item()
    intrinsic_72 = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic_72['width'],
        height=intrinsic_72['height'],
        fx=intrinsic_72['fx'],
        fy=intrinsic_72['fy'],
        cx=intrinsic_72['ppx'],
        cy=intrinsic_72['ppy'])
    rgbd_image_72 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image_72),
        o3d.geometry.Image(depth_image_72),
        depth_scale=1,  # Adjust based on your depth image scale
        depth_trunc=1.5,  # Truncate depth values beyond this distance
        convert_rgb_to_intensity=False)
    
    color_image_83 = cv2.imread(os.path.join('data', serial_numbers[2] + '_color.png'))
    color_image_83 = cv2.cvtColor(color_image_83, cv2.COLOR_BGR2RGB)
    depth_image_83 = (np.load(os.path.join('data', serial_numbers[2] + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
    intrinsic_83 = np.load(os.path.join('data', serial_numbers[2] + '_intrinsics.npy'), allow_pickle=True).item()
    intrinsic_83 = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic_83['width'],
        height=intrinsic_83['height'],
        fx=intrinsic_83['fx'],
        fy=intrinsic_83['fy'],
        cx=intrinsic_83['ppx'],
        cy=intrinsic_83['ppy'])
    rgbd_image_83 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image_83),
        o3d.geometry.Image(depth_image_83),
        depth_scale=1,  # Adjust based on your depth image scale
        depth_trunc=1.5,  # Truncate depth values beyond this distance
        convert_rgb_to_intensity=False)

    transform_matrix_83_72 = np.array([[ 0.05702207, -0.4745224, 0.8783945, -1.067303],
                                     [0.9192188, 0.3682892, 0.1392833, -0.4351598],
                                     [ -0.3895963, 0.7994946, 0.4571905, 0.3163233],
                                    [0, 0, 0, 1]], dtype=np.float32)
    # transform_matrix_83_72 = np.linalg.inv(transform_matrix_83_72)  # Invert the transformation matrix for the slave camera
    

    color_image_29 = cv2.imread(os.path.join('data', serial_numbers[0] + '_color.png'))
    color_image_29 = cv2.cvtColor(color_image_29, cv2.COLOR_BGR2RGB)
    depth_image_29 = (np.load(os.path.join('data', serial_numbers[0] + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
    intrinsic_29 = np.load(os.path.join('data', serial_numbers[0] + '_intrinsics.npy'), allow_pickle=True).item()
    intrinsic_29 = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic_29['width'],
        height=intrinsic_29['height'],
        fx=intrinsic_29['fx'],
        fy=intrinsic_29['fy'],
        cx=intrinsic_29['ppx'],
        cy=intrinsic_29['ppy'])
    rgbd_image_29 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image_29),
        o3d.geometry.Image(depth_image_29),
        depth_scale=1,  # Adjust based on your depth image scale
        depth_trunc=1.5,  # Truncate depth values beyond this distance
        convert_rgb_to_intensity=False)
    transform_matrix_29_83 = np.array([[ 0.05936317, -0.8555212, 0.5143535, -0.4373211],
                                     [0.5357799, 0.4620624, 0.7067095, -0.5153366],
                                     [-0.8422683, 0.2336278, 0.4858005, 0.8538319],
                                    [0, 0, 0, 1]], dtype=np.float32)

    transform_matrix_29_83 = np.linalg.inv(transform_matrix_29_83) 
    # transform_29_72 = transform_matrix_29_83 @ transform_matrix_83_72

    pcd_83 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_83, 
                                                            intrinsic_83,
                                                            extrinsic=np.eye(4))

    pcd_29 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_29, 
                                                                intrinsic_29,
                                                                extrinsic=transform_matrix_29_83)
    

    pcd_72 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_72, 
                                                                intrinsic_72,
                                                                extrinsic=transform_matrix_83_72)
    
    
    o3d.io.write_point_cloud("pointclouds/pcd_72.ply", pcd_72)
    o3d.io.write_point_cloud("pointclouds/pcd_83.ply", pcd_83)
    o3d.io.write_point_cloud("pointclouds/pcd_29.ply", pcd_29)
        # o3d.visualization.draw_geometries([pcd_master, pcd_slave])



def single_pcd(serial_number = serial_numbers[0]):
    color_image = cv2.imread(os.path.join('data', serial_number + '_color.png'))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    depth_image = (np.load(os.path.join('data', serial_number + '_depth.npy'), allow_pickle=True) * 0.001).astype(np.float32)  # Convert depth to meters
    intrinsic_ = np.load(os.path.join('data', serial_number + '_intrinsics.npy'), allow_pickle=True).item()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic_['width'],
        height=intrinsic_['height'],
        fx=intrinsic_['fx'],
        fy=intrinsic_['fy'],
        cx=intrinsic_['ppx'],
        cy=intrinsic_['ppy'])
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1,  # Adjust based on your depth image scale
        depth_trunc=1.5,  # Truncate depth values beyond this distance
        convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, 
                                                                intrinsic,
                                                                extrinsic=np.eye(4))
    
    o3d.io.write_point_cloud("pcd.ply", pcd)


# single_pcd()
view_combined_point_cloud()
# three_camera_pointcloud()