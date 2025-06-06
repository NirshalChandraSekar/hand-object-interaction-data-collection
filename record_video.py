'''
This file contains the code to record video streams form realsense cameras and store them in a h5 file.
'''

import pyrealsense2 as rs
import numpy as np
import h5py
import cv2


class Camera:
    def __init__(self):
        self.ctx = rs.context()


    def check_available_cameras(self):
        # Print the list of available RealSense cameras

        devices = self.ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense cameras found.")
        else:
            print("Available RealSense cameras:")
            for device in devices:
                print(f"  - {device.get_info(rs.camera_info.name)} (Serial: {device.get_info(rs.camera_info.serial_number)})")

        
    def view_camera_streams(self, serial_numbers=[device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()]):
        
        # Create a pipeline and config for each camera
        pipelines = []
        for serial in serial_numbers:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            pipeline.start(config)
            pipelines.append(pipeline)
        
        try:
            while True:
                # Process each camera
                for i, pipeline in enumerate(pipelines):
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                        
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Align depth to color
                    aligned_depth_frame = rs.align(rs.stream.color).process(frames).get_depth_frame()
                    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    
                    # Show camera-specific windows
                    cv2.imshow(f'Color Stream {serial_numbers[i]}', color_image)
                    cv2.imshow(f'Aligned Depth Stream {serial_numbers[i]}', aligned_depth_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            for pipeline in pipelines:
                pipeline.stop()
            cv2.destroyAllWindows()
        

    def record_video_streams(self, serial_numbers=[device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()], 
                       output_file='dataset/video_data.h5'):
        # Create a pipeline and config for each camera
        pipelines = []
        for serial in serial_numbers:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            pipeline.start(config)
            pipelines.append(pipeline)

        with h5py.File(output_file, 'w') as h5file:
            frame_count = 0
            try:
                while True:
                    # Create a group for this frame
                    frame_group = h5file.create_group(f'frame_{frame_count}')
                    # Create a video_streams group within the frame group
                    video_streams_group = frame_group.create_group('video_streams')
                    
                    for i, pipeline in enumerate(pipelines):
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        depth_frame = frames.get_depth_frame()

                        # if not color_frame or not depth_frame:
                        #     continue

                        color_image = np.asanyarray(color_frame.get_data())
                        
                        # Align depth to color
                        aligned_frames = rs.align(rs.stream.color).process(frames)
                        aligned_depth_frame = aligned_frames.get_depth_frame()
                        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

                        # Store images in the video_streams group for this frame
                        video_streams_group.create_dataset(f'color_{serial_numbers[i]}', data=color_image)
                        video_streams_group.create_dataset(f'depth_{serial_numbers[i]}', data=aligned_depth_image)
                    
                    frame_count += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                for pipeline in pipelines:
                    pipeline.stop()

if __name__ == "__main__":
    camera = Camera()
    # camera.check_available_cameras()
    # camera.view_camera_streams()
    camera.record_video_streams(output_file='dataset/trial_dataset/check_video_data.h5')
