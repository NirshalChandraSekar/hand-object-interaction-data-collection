'''
This file contains the code to record video streams form realsense cameras and store them in a h5 file.
'''

import pyrealsense2 as rs
import numpy as np
import h5py
import cv2
import time


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

    def record_video_streams(self, output_file='dataset/video_data.h5', duration=10, serial_numbers=None):
        if serial_numbers is None:
            serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in self.ctx.query_devices()]

        pipelines = []
        align = rs.align(rs.stream.color)

        for serial in serial_numbers:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            pipeline.start(config)
            pipelines.append(pipeline)

        try:
            with h5py.File(output_file, 'w') as h5file:
                color_groups = {serial: h5file.create_group(f"{serial}/color") for serial in serial_numbers}
                depth_groups = {serial: h5file.create_group(f"{serial}/depth") for serial in serial_numbers}
                frame_counters = {serial: 0 for serial in serial_numbers}

                print("[INFO] Recording started. Press Ctrl+C to stop early.")
                start_time = time.time()

                while time.time() - start_time < duration:
                    for i, pipeline in enumerate(pipelines):
                        frames = pipeline.wait_for_frames()
                        aligned_frames = align.process(frames)
                        color_frame = aligned_frames.get_color_frame()
                        depth_frame = aligned_frames.get_depth_frame()

                        if not color_frame or not depth_frame:
                            continue

                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(depth_frame.get_data())

                        serial = serial_numbers[i]
                        idx = frame_counters[serial]
                        color_groups[serial].create_dataset(str(idx), data=color_image, compression='gzip')
                        depth_groups[serial].create_dataset(str(idx), data=depth_image, compression='gzip')
                        frame_counters[serial] += 1

        except KeyboardInterrupt:
            print("[INFO] Recording interrupted by user.")
        finally:
            for pipeline in pipelines:
                pipeline.stop()
            print("[INFO] All pipelines stopped.")
    
if __name__ == "__main__":
    camera = Camera()
    # camera.check_available_cameras()
    # camera.view_camera_streams()
    camera.record_video_streams(output_file='dataset/trial_dataset/check_video_data.h5')
    
