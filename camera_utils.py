import pyrealsense2 as rs
import numpy as np
import h5py
import cv2
from datetime import datetime, timezone
from pynput import keyboard
import threading
import time
import pyaudio
import wave
import os

recording = False
stop_requested = False
exit_requested = False


class AudioRecorder:
    """
    Handles audio recording using PyAudio in a separate thread.
    Records audio data and saves to a WAV file.
    """
    def __init__(self, output_file, rate=48000, channels=2, chunk=4096):
        self.output_file = output_file
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.frames = []
        self.stop_event = threading.Event()
        self.audio_start_time = None

    def record(self, start_time):
        """
        Start recording audio until stop_event is set.
        
        Args:
            start_time (float): The global recording start timestamp, used for synchronization.
        """
        self.frames = []
        self.stop_event.clear()
        self.audio_start_time = datetime.now(timezone.utc).timestamp() - start_time

        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,ine in enumerate(pipelines):
                        frames = pipeline.wait_for_frames()
                        aligned_frames = align.process(frames)
                        color_frame = aligned_frames.get_color_frame()
                        depth_frame = aligned_frames.get_depth_frame()

                        if not color_frame or not depth_f
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)

        print("[AUDIO] Recording started.")
        while not self.stop_event.is_set():
            data = stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        print("[AUDIO] Recording stopped.")

        wf = wave.open(self.output_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()


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

    def record_video_streams(self, t_matrices, serial_numbers=None, stop_flag=lambda: False):
        """
        Records synchronized color and depth streams from specified RealSense cameras,
        storing frames and timestamps in an HDF5 file. Also records audio in parallel.

        Args:
            serial_numbers (List[str], optional): List of serial numbers to record from.
                If None, records from all connected devices.
            stop_flag (callable): Function returning True to stop recording.
        """
        if serial_numbers is None:
            serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in self.ctx.query_devices()]
            if len(serial_numbers) < 1:
                print("[ERROR] No RealSense cameras found.")
                return

        pipelines = []
        align = rs.align(rs.stream.color)

        start_time = datetime.now(timezone.utc).isoformat()

        vid_dir = "dataset/videos"
        os.makedirs(vid_dir, exist_ok=True)

        audio_dir = "dataset/audio"
        os.makedirs(audio_dir, exist_ok=True)

        output_file = f"{vid_dir}/{start_time}.h5"
        audio_output_file = f"{audio_dir}/{start_time}.wav"

        print(f"[INFO] Saving video to: {output_file}")
        print(f"[INFO] Saving audio to: {audio_output_file}")

        try:
            with h5py.File(output_file, 'w') as h5file:
                t_matrices_group = h5file.create_group("t_matrices")
                t_matrices_group = h5file.create_group("t_matrices")
                for cam_pair in t_matrices:
                    t_matrices_group.create_dataset(cam_pair, data=t_matrices[cam_pair])
                color_groups = {serial: h5file.create_group(f"{serial}/frames/color") for serial in serial_numbers}
                depth_groups = {serial: h5file.create_group(f"{serial}/frames/depth") for serial in serial_numbers}
                timestamps = {serial: h5file.create_dataset(f"{serial}/frames/timestamps", shape=(0,), maxshape=(None,), dtype='float64')
                              for serial in serial_numbers}
                params_group = {serial: h5file.create_group(f"{serial}/params") for serial in serial_numbers}
                frame_counters = {serial: 0 for serial in serial_numbers}

                for serial in serial_numbers:
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_device(serial)
                    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                    pipeline.start(config)
                    pipelines.append(pipeline)

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
                    
                    params_group[serial].create_dataset("start_time", data=start_time, dtype=h5py.string_dtype(encoding='utf-8'))

                print("Recording started. Press 'r' again to stop.")
                recording_start = datetime.now(timezone.utc).timestamp()

                # Start audio thread
                # Comment this out when you don't need it
                audio_recorder = AudioRecorder(audio_output_file)
                audio_thread = threading.Thread(target=audio_recorder.record, args=(recording_start,))
                audio_thread.start()

                while not stop_flag():
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
                        idx = frame_counters[serial]
                        ts = datetime.now(timezone.utc).timestamp() - recording_start

                        color_groups[serial].create_dataset(str(idx), data=color_image, compression='gzip', compression_opts=9, chunks=True)
                        depth_groups[serial].create_dataset(str(idx), data=depth_image, compression='gzip', compression_opts=9, chunks=True)
                        timestamps[serial].resize((idx + 1,))
                        timestamps[serial][idx] = ts

                        frame_counters[serial] += 1

                audio_recorder.stop_event.set()
                audio_thread.join()
                for serial in serial_numbers:
                    params_group[serial].create_dataset("audio_start_time", data=audio_recorder.audio_start_time)

        except Exception as e:
            print(f"ERROR: {e}")

        finally:
            for pipeline in pipelines:
                pipeline.stop()
            print("Recording session finished. Press 'r' again to start.")


def listen_for_keys():
    """
    Listens for keyboard inputs ('r' to start/stop recording, 'q' to quit) in a separate thread.
    """
    def on_press(key):
        global recording, stop_requested, exit_requested
        if hasattr(key, 'char'):
            if key.char == 'r':
                if recording:
                    stop_requested = True
                    print("[INFO] Stopping recording.")
                else:
                    recording = True
                    print("[INFO] Starting new recording.")
            elif key.char == 'q':
                stop_requested = True
                exit_requested = True
                print("[INFO] Exiting program.")
                return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def record_audio_video(t_matrices):
    """
    Main entry point to start recording video and audio streams based on keyboard input.
    Starts listener thread to handle 'r' and 'q' keys.
    """

    global recording, stop_requested, exit_requested

    camera = Camera()
    camera.check_available_cameras()

    listener_thread = threading.Thread(target=listen_for_keys, daemon=True)
    listener_thread.start()

    print("Press 'r' to start/stop recording, 'q' to quit.")

    while not exit_requested:
        if recording:
            stop_requested = False
            camera.record_video_streams(t_matrices, stop_flag=lambda: stop_requested)
            recording = False
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

# Translate Later
# audio_start_absolute = start_time + audio_start_time  # global clock
# frame_absolute_time = start_time + ts_from_hdf5

# # audio sample index corresponding to video timestamp
# sample_index = int((frame_absolute_time - audio_start_absolute) * 44100)
