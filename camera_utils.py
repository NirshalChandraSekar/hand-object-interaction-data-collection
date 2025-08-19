import pyrealsense2 as rs
import numpy as np
import h5py
from datetime import datetime
from zoneinfo import ZoneInfo
from pynput import keyboard
import threading
import time
import os
import cv2
from audio_utils import AudioRecorder

class CameraRecorder:
    def __init__(self):
        self.frames_lock = threading.Lock()
        self.reset()
        self.ctx = rs.context()
        self.age = None
        self.dominant_hand = None

    def reset(self):
        self.camera_frames = {}
        self.task = None
        self.exit_requested = False
        self.stop_requested = False
        self.recording = False

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

    def listener_thread(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def on_press(self, key):
        try:
            if getattr(self, 'pause_listener', False):
                # Ignore keypresses during input phase
                return
            elif key.char in ('1', '2', '3') and not self.recording:
                self.recording = True
                self.task = key.char
                self.stop_requested = False
                print(f"----- Starting recording for task {self.task} -----")
            elif key.char == 's':
                if self.recording:
                    self.stop_requested = True
                    print("----- Stopping recording -----")
                else:
                    print("Press '1' or '2' or '3' to start recording.")
            elif key.char == 'c':
                self.age = None
                self.dominant_hand = None
                print("Resetting age and dominant hand. Please re-enter them next time.")
            elif key.char == 'q':
                self.stop_requested = True
                self.exit_requested = True
                print("Exiting program.")
                return False
        except AttributeError:
            pass

    def secondary_cam_thread(self, serial, pipeline, align):
        try:
            while not self.stop_requested:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    continue
                aligned = align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                timestamp = time.time()

                if not color or not depth:
                    continue

                color_img = np.asanyarray(color.get_data())
                depth_img = np.asanyarray(depth.get_data()).astype(np.uint16)

                with self.frames_lock:
                    self.camera_frames[serial] = (color_img, depth_img, timestamp)
        except Exception as e:
            print(f"Error with camera {serial}: {e}")
        finally:
            pipeline.stop()

    def record_video_streams(self, task, serial_numbers, t_matrices=None):
        print(serial_numbers)
        if serial_numbers is None:
            serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in self.ctx.query_devices()]
            if len(serial_numbers) < 1:
                print("[ERROR] No RealSense cameras found.")
                return
            
        self.pause_listener = True

        if(self.age is None or self.dominant_hand is None):

            self.dominant_hand = "Right" if (input("Enter dominant hand (l/r): ").lower() == 'r') else "Left"
            self.age = input("Enter age: ")

        self.pause_listener = False

        base_dir = f"datasets"
        central = ZoneInfo("America/Chicago")  # Central Time
        now = datetime.now(central)
        timestamp_str = now.strftime("%I-%M%p").lstrip("0").lower()

        # Define directory paths
        task_dir = os.path.join(base_dir, f"Task {task}")
        audio_dir = os.path.join(task_dir, "audio")
        video_dir = os.path.join(task_dir, "video")

        # Make sure directories exist
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        file_name = f"{timestamp_str}_{self.age}_{self.dominant_hand}"
        # Define file paths
        output_file = os.path.join(video_dir, f"{file_name}.h5")
        audio_output_file = os.path.join(audio_dir, f"{file_name}.wav")

        pipelines = {}
        alignments = {}

        try:
            with h5py.File(output_file, 'w') as h5file:
                metadata_group = h5file.create_group("metadata")
                metadata_group.create_dataset("dominant_hand", data=self.dominant_hand)
                metadata_group.create_dataset("age", data=self.age)
                if t_matrices:
                    t_group = metadata_group.create_group("t_matrices")
                    for k, v in t_matrices.items():
                        t_group.create_dataset(k, data=v)

                color_groups = {s: h5file.create_group(f"{s}/frames/color") for s in serial_numbers}
                depth_groups = {s: h5file.create_group(f"{s}/frames/depth") for s in serial_numbers}
                timestamps = {s: h5file.create_dataset(f"{s}/frames/timestamps", shape=(0,), maxshape=(None,), dtype='float64') for s in serial_numbers}
                params_group = {serial: h5file.create_group(f"{serial}/params") for serial in serial_numbers}
                frame_counters = {serial: 0 for serial in serial_numbers}

                for serial in serial_numbers:
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                    pipeline.start(config)
                    pipelines[serial] = pipeline

                    alignments[serial] = rs.align(rs.stream.color)

                    intr = self.get_camera_intrinsics(pipeline)
                    intr_group = params_group[serial].create_group("intrinsics")
                    intr_group.create_dataset("width", data=intr.width)
                    intr_group.create_dataset("height", data=intr.height)
                    intr_group.create_dataset("ppx", data=intr.ppx)
                    intr_group.create_dataset("ppy", data=intr.ppy)
                    intr_group.create_dataset("fx", data=intr.fx)
                    intr_group.create_dataset("fy", data=intr.fy)
                    intr_group.create_dataset("model", data=str(intr.model))
                    intr_group.create_dataset("coeffs", data=intr.coeffs)
            
                threads = []
                for serial in serial_numbers:
                    thread = threading.Thread(target=self.secondary_cam_thread, args=(serial, pipelines[serial], alignments[serial]))
                    thread.start()
                    threads.append(thread)

                audio_recorder = AudioRecorder(audio_output_file)
                audio_thread = threading.Thread(target=audio_recorder.record)
                audio_thread.start()
                threads.append(audio_thread)

                print("Recording started. Press 's' to stop.")
                recording_start = time.time()

                while not self.stop_requested:
                    time.sleep(0.03)  # Avoid CPU overload

                    with self.frames_lock:
                        data = self.camera_frames.copy()

                    for serial, (color_img, depth_img, ts) in data.items():
                        color_groups[serial].create_dataset(str(frame_counters[serial]), data=color_img, compression='gzip', compression_opts=9, chunks=True)
                        depth_groups[serial].create_dataset(str(frame_counters[serial]), data=depth_img, compression='gzip', compression_opts=9, chunks=True)
                        timestamps[serial].resize((frame_counters[serial] + 1,))
                        timestamps[serial][frame_counters[serial]] = ts
                        frame_counters[serial] += 1

                audio_recorder.stop_event.set()
                for thread in threads:
                    thread.join()

                metadata_group.create_dataset("recording_start_time", data=recording_start, dtype='float32')
                metadata_group.create_dataset("audio_start_time", data=audio_recorder.audio_start_time, dtype='float32')

        except Exception as e:
            print(f"ERROR during recording: {e}")
            os.remove(output_file)
            os.remove(audio_output_file)
        finally:
            for pipeline in pipelines.values():
                try:
                    pipeline.stop()
                except Exception as e:
                    pass
            if task == '3':
                print("Click 'c' to reset metadata info for the next participant.")
            print("Recording session finished. Press '1' or '2' or '3' to start again, or 'q' to quit.")

    def start_recording(self, serial_numbers, t_matrices=None):

        print("Press '1' or '2' or '3' to start recording. Press 's' to stop. Press 'q' to quit.")
        listener = threading.Thread(target=self.listener_thread, daemon=True)
        listener.start()

        while not self.exit_requested:
            if self.recording:
                try:    
                    self.record_video_streams(self.task, serial_numbers, t_matrices)
                except Exception as e:
                    print(f"TRY AGAIN: Error occurred while recording: {e}")
                    continue
                self.reset()
            time.sleep(0.1)

        listener.join()
        print("Program exited.")

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

