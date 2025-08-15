import pyrealsense2 as rs
import numpy as np
import h5py
import cv2
# from datetime import datetime, timezone
from pynput import keyboard
import threading
import time
import os
from audio_utils import AudioRecorder

recording = False
stop_requested = False
exit_requested = False
task = None

class Camera:
    """
    Manages RealSense camera interactions: device queries, intrinsics, and video recording.
    """
    def __init__(self):
        self.ctx = rs.context()

    def check_available_cameras(self):
        """
        Manages RealSense camera interactions: device queries, intrinsics, and video recording.
        """
        devices = self.ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense cameras found.")
        else:
            print("Available RealSense cameras:")
            for device in devices:
                print(f"  - {device.get_info(rs.camera_info.name)} (Serial: {device.get_info(rs.camera_info.serial_number)})")

    def get_camera_intrinsics(self, pipeline):
        """
        Retrieves intrinsics from the color stream of the given pipeline.
        
        Args:
            pipeline (rs.pipeline): Started RealSense pipeline.

        Returns:
            rs.intrinsics: Camera intrinsics object.
        """
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        profile = color_frame.get_profile().as_video_stream_profile()
        intr = profile.get_intrinsics()
        return intr

    def record_video_streams(self, serial_numbers=None, t_matrices=None, task="1", stop_flag=lambda: False):
        ctx = self.ctx
        devices = ctx.query_devices()

        if serial_numbers is None:
            serial_numbers = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

        # Data saving logistics
        start_time = time.time()
        dir = f"dataset/task{task}"
        vid_dir = f"{dir}/videos"
        audio_dir = f"{dir}/audio"
        os.makedirs(vid_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        output_file = f"{vid_dir}/{start_time}.h5"
        audio_output_file = f"{audio_dir}/{start_time}.wav"

        pipelines = []
        aligners = {}

        for i, dev in enumerate(devices):
            serial_number = dev.get_info(rs.camera_info.serial_number)
            try:
                # Attempt to stop any existing pipeline for this device
                for pipeline in getattr(dev, 'pipelines', []):
                    pipeline.stop()
            except Exception:
                # If there’s no active pipeline, ignore
                pass
            pipeline = rs.pipeline()
            depth_sensor = dev.query_sensors()[0]
            targetSyncMode = 1 if i == 0 else 2  # Master/slave
            depth_sensor.set_option(rs.option.inter_cam_sync_mode, targetSyncMode)

            config = rs.config()
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

            pipeline.start(config)
            pipelines.append(pipeline)
            aligners[serial_number] = rs.align(rs.stream.color)

        master_serial = serial_numbers[0]
        
        # Audio thread
        audio_recorder = AudioRecorder(audio_output_file)
        audio_thread = threading.Thread(target=audio_recorder.record, args=())
        audio_thread.start()

        color_buffers = {s: [] for s in serial_numbers}
        depth_buffers = {s: [] for s in serial_numbers}
        timestamp_buffers = {s: [] for s in serial_numbers}

        print(f"[INFO] Recording started.")
        recording_start = time.time()

        try:
            while not stop_flag():
                master_serial = serial_numbers[0]
                master_frames = pipelines[0].wait_for_frames()
                master_aligned = aligners[master_serial].process(master_frames)
                synced_frames = {master_serial: master_aligned}
                frame_ts = {master_serial: time.time() - recording_start}

                for i, serial in enumerate(serial_numbers[1:], start=1):
                    frames = pipelines[i].wait_for_frames()
                    aligned = aligners[serial].process(frames)
                    synced_frames[serial] = aligned
                    frame_ts[serial] = time.time() - recording_start

                for serial in serial_numbers:
                    frames = synced_frames[serial]
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    color_img = np.asanyarray(color_frame.get_data())
                    depth_img = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

                    color_buffers[serial].append(color_img)
                    depth_buffers[serial].append(depth_img)
                    timestamp_buffers[serial].append(frame_ts[serial])

            print("[INFO] Stopping recording...")

            # Stop audio
            audio_recorder.stop_event.set()
            audio_thread.join()

            # Save to HDF5
            with h5py.File(output_file, 'w') as h5file:
                metadata_group = h5file.create_group("metadata")
                metadata_group.create_dataset("start_time", data=start_time)
                metadata_group.create_dataset("recording_start_time", data=recording_start)
                metadata_group.create_dataset("audio_start_time", data=audio_recorder.audio_start_time)

                if t_matrices is not None:
                    t_group = metadata_group.create_group("t_matrices")
                    for pair, matrix in t_matrices.items():
                        t_group.create_dataset(pair, data=matrix)

                for i, serial in enumerate(serial_numbers):
                    cam_group = h5file.create_group(f"cameras/{serial}")
                    frame_group = cam_group.create_group("frames")
                    frame_group.create_dataset("color", data=np.array(color_buffers[serial], dtype=np.uint8), compression="gzip")
                    frame_group.create_dataset("depth", data=np.array(depth_buffers[serial], dtype=np.uint16), compression="gzip")
                    frame_group.create_dataset("timestamps", data=np.array(timestamp_buffers[serial], dtype=np.float32))

                    # Intrinsics
                    intr = self.get_camera_intrinsics(pipelines[i])
                    intr_group = cam_group.create_group("params/intrinsics")
                    intr_group.create_dataset("width", data=intr.width)
                    intr_group.create_dataset("height", data=intr.height)
                    intr_group.create_dataset("ppx", data=intr.ppx)
                    intr_group.create_dataset("ppy", data=intr.ppy)
                    intr_group.create_dataset("fx", data=intr.fx)
                    intr_group.create_dataset("fy", data=intr.fy)
                    intr_group.create_dataset("model", data=str(intr.model))
                    intr_group.create_dataset("coeffs", data=intr.coeffs)

            print(f"[INFO] Recording saved: {output_file}, {audio_output_file}")
        except Exception as e:
            os.remove(output_file)
            os.remove(audio_output_file)
            print(f"[ERROR] Exception during recording: {e}")
        finally:

            # Stop pipelines
            for pipeline in pipelines:
                pipeline.stop()
            print(f"[INFO] Recording stopped. Files saved: {output_file}, {audio_output_file}")


def listen_for_keys():
    """
    Listens for keyboard inputs ('r' to start/stop recording, 'q' to quit) in a separate thread.
    """
    def on_press(key):
        global recording, stop_requested, exit_requested, task
        if hasattr(key, 'char'):
            if key.char in ('1', '2'):
                task = key.char
                if recording:
                    stop_requested = True
                    print("[INFO] Stopping recording.")
                else:
                    recording = True
                    print(f"[INFO] Starting new recording for task {task}.")
            elif key.char == 'r':
                if recording:
                    stop_requested = True
                    print("[INFO] Stopping recording.")
                else:
                    print("[WARNING] Press '1' or '2' to start a task-specific recording.")
            elif key.char == 'q':
                stop_requested = True
                exit_requested = True
                print("[INFO] Exiting program.")
                return False


    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def record_audio_video(serial_numbers=None, t_matrices=None):
    """
    Main entry point to start recording video and audio streams based on keyboard input.
    Starts listener thread to handle 'r' and 'q' keys.
    """

    global recording, stop_requested, exit_requested, task

    camera = Camera()
    camera.check_available_cameras()

    listener_thread = threading.Thread(target=listen_for_keys, daemon=True)
    listener_thread.start()

    print("Press '1' or '2' to start/stop recording, 'q' to quit.")

    while not exit_requested:
        if recording:
            stop_requested = False
            camera.record_video_streams(serial_numbers=serial_numbers, task=task, stop_flag=lambda: stop_requested)
            recording = False
            task = None 
        time.sleep(0.1)

    print("[INFO] Program exited.")


def view_live_camera_streams(serial_numbers=[device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()]):
    """
    Displays live color and aligned depth streams from specified RealSense cameras in OpenCV windows.
    
    Args:
        serial_numbers (List[str]): List of serial numbers to display streams for.
    """
    pipelines = []
    for serial in serial_numbers:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append(pipeline)
    
    try:
        while True:
            # Process each camera
            for i, pipeline in enumerate(pipelines):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()                
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                                
                # Show camera-specific windows
                cv2.imshow(f'Color Stream {serial_numbers[i]}', color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()

def save_images():
    """
    Captures and saves single color and depth frames from all connected RealSense cameras.
    Saves images and intrinsics data into the 'data' directory.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    os.makedirs('data', exist_ok=True)

    for device in devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        pipeline.start(config)

        align = rs.align(rs.stream.color)
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            # Get aligned color and depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print(f"Frames not captured for device {serial_number}")
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Save images with serial number as filename
            cv2.imwrite(f'data/{serial_number}_color.png', color_image)
            np.save(f'data/{serial_number}_depth.npy', depth_image)

            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            # Save intrinsic parameters
            intrinsics_data = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'model': intrinsics.model,
                'coeffs': intrinsics.coeffs
            }
            np.save(f'data/{serial_number}_intrinsics.npy', intrinsics_data)
        except Exception as e:
            print(f"Error capturing frames for device {serial_number}: {e}")
        finally:
            # Stop the pipeline
            pipeline.stop()    
