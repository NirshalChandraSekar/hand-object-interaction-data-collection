'''
This file contains the code to record video streams form realsense cameras and store them in a h5 file.
'''

import pyrealsense2 as rs
import numpy as np
import hdf5plugin
import h5py
import cv2
from datetime import datetime, date, timezone


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
    def get_camera_intrinsics(self, serial_number):
            ctx = rs.context()
            device = next(dev for dev in ctx.query_devices() if dev.get_info(rs.camera_info.serial_number) == serial_number)
            depth_sensor = device.first_depth_sensor()
            intr = depth_sensor.get_stream_profiles()[0].as_video_stream_profile().get_intrinsics()
            return intr

    def record_video_streams(self, serial_numbers=None):
        if serial_numbers is None:
            serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in self.ctx.query_devices()]

        pipelines = []
        align = rs.align(rs.stream.color)

        start_time = datetime.now(timezone.utc).isoformat()
        output_file = f"dataset/trial_dataset/{start_time}.h5"


        try:
            with h5py.File(output_file, 'w') as h5file:
                color_groups = {serial: h5file.create_group(f"{serial}/frames/color") for serial in serial_numbers}
                depth_groups = {serial: h5file.create_group(f"{serial}/frames/depth") for serial in serial_numbers}
                timestamps = {serial: h5file.create_dataset(f"{serial}/frames/timestamps",shape=(0,), maxshape=(None,), dtype='float64')
                                                            for serial in serial_numbers}

                params_group = {serial: h5file.create_group(f"{serial}/params") for serial in serial_numbers}

                frame_counters = {serial: 0 for serial in serial_numbers}

                for serial in serial_numbers:
                    # Set up pipelines
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    pipeline.start(config)
                    pipelines.append(pipeline)

                    # Store params
                    intr = self.get_camera_intrinsics(serial)
                    intr_group = params_group[serial].create_group("intrinsics")
                    intr_group.create_dataset("width", data=intr.width)
                    intr_group.create_dataset("height", data=intr.height)
                    intr_group.create_dataset("ppx", data=intr.ppx)
                    intr_group.create_dataset("ppy", data=intr.ppy)
                    intr_group.create_dataset("fx", data=intr.fx)
                    intr_group.create_dataset("fy", data=intr.fy)
                    intr_group.create_dataset("model", data=str(intr.model))
                    intr_group.create_dataset("coeffs", data=intr.coeffs)

                    # 
                    params_group[serial].create_dataset("start_time", data=start_time, dtype=h5py.string_dtype(encoding='utf-8'))

                print("Press the 'space' key to start recording")

                cv2.namedWindow("Press 'space' to start recording", cv2.WINDOW_NORMAL)
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == 32:
                        break

                cv2.destroyAllWindows()
                
                print("[INFO] Recording started. Press 'Ctrl + c' to stop")

                recording_start = datetime.now(timezone.utc).timestamp()

                while True:
                    for i, pipeline in enumerate(pipelines):
                        frames = pipeline.wait_for_frames()
                        aligned_frames = align.process(frames)
                        color_frame = aligned_frames.get_color_frame()
                        depth_frame = aligned_frames.get_depth_frame()

                        if not color_frame or not depth_frame:
                            continue

                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

                        serial = serial_numbers[i]
                        utc_now = datetime.now(timezone.utc).timestamp() - recording_start
                        idx = frame_counters[serial]
                        color_groups[serial].create_dataset(str(idx), data=color_image, compression='gzip', compression_opts=9, chunks = True)
                        depth_groups[serial].create_dataset(str(idx), data=depth_image, compression='gzip', compression_opts=9, chunks=True)
                        timestamps[serial].resize((idx + 1,))
                        timestamps[serial][idx] = utc_now

                        frame_counters[serial] += 1
        except KeyboardInterrupt:
            print("[INFO] Recording ended by user.")
        finally:
            for pipeline in pipelines:
                pipeline.stop()
            print("[INFO] All pipelines stopped.")
    
if __name__ == "__main__":
    camera = Camera()
    # camera.check_available_cameras()
    # camera.view_camera_streams()
    camera.record_video_streams()
    
