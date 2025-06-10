import h5py
import cv2
import numpy as np
# import mediapipe as mp
import open3d as o3d
import pyrealsense2 as rs

dataset_path = 'dataset/trial_dataset/check_video_data.h5'  # <- change this to your actual path if needed
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


def view_point_cloud(dataset_path, serial_number):
    with h5py.File(dataset_path, 'r') as dataset:
        color_image = dataset[serial_number]['color'][str(0)][()]
        depth_image = dataset[serial_number]['depth'][str(0)][()]

        # Create point cloud from depth image
        intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 320.0, 240.0)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_image), intrinsics)

        # Color the point cloud
        pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

        o3d.visualization.draw_geometries([pcd])


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

# save_video_from_dataset(dataset_path, combined_video.MP4, serial_numbers)
# run_mediapipe_on_videos(dataset_path, 'mediapipe_combined_video.MP4', serial_numbers)
# view_point_cloud(dataset_path, serial_numbers[0])

get_camera_intrinsics(serial_numbers)






