import pyrealsense2 as rs
import numpy as np
import cv2


def view_streams(serial_numbers=[device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()]):
    pipelines = []
    for serial in serial_numbers:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
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

def calibrate_extrinsics():
    