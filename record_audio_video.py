import pyrealsense2 as rs
import numpy as np
import h5py
import threading
import time
from datetime import datetime, timezone
from queue import Queue
import pyaudio
import wave
import os

class MultiCameraRecorder:
    def __init__(self, participant_num):
        self.ctx = rs.context()
        self.participant_num = participant_num
        self.align = rs.align(rs.stream.color)
        
        self.frame_queue = Queue()
        self.stop_event = threading.Event()

        self.audio_frames = []
        self.audio_rate = 44100
        self.audio_channels = 1
        self.audio_format = pyaudio.paInt16
        self.chunk_size = 1024

        self.vid_output_file = f"dataset/video/{participant_num}.h5"
        self.audio_output_file = f"dataset/audio/{participant_num}.wav"
        
        os.makedirs(os.path.dirname(self.audio_output_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.vid_output_file), exist_ok=True)

    def get_serial_numbers(self):
        return [dev.get_info(rs.camera_info.serial_number) for dev in self.ctx.query_devices()]

    def get_camera_intrinsics(self, pipeline):
        profile = pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        return color_stream.as_video_stream_profile().get_intrinsics()

    def record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_format,
                        channels=self.audio_channels,
                        rate=self.audio_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)

        print("Audio recording started.")
        while not self.stop_event.is_set():
            data = stream.read(self.chunk_size)
            self.audio_frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio recording stopped.")

        wav_path = self.audio_output_file
        wf = wave.open(wav_path, 'wb')
        wf.setnchannels(self.audio_channels)
        wf.setsampwidth(p.get_sample_size(self.audio_format))
        wf.setframerate(self.audio_rate)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
        print(f"Audio saved to {wav_path}")

    def record_video_streams(self):
        serial_numbers = self.get_serial_numbers()
        if not serial_numbers:
            print("No RealSense cameras found.")
            return

        print(f"Saving video to: {self.vid_output_file}")
        pipelines = []
        frame_counters = {serial: 0 for serial in serial_numbers}

        with h5py.File(self.vid_output_file, 'w') as h5file:
            color_groups = {s: h5file.create_group(f"{s}/frames/color") for s in serial_numbers}
            depth_groups = {s: h5file.create_group(f"{s}/frames/depth") for s in serial_numbers}
            timestamp_groups = {s: h5file.create_dataset(f"{s}/frames/timestamps", shape=(0,), maxshape=(None,), dtype='float32') for s in serial_numbers}
            params_group = {s: h5file.create_group(f"{s}/params") for s in serial_numbers}

            for serial in serial_numbers:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                pipeline.start(config)
                pipelines.append(pipeline)

                intr = self.get_camera_intrinsics(pipeline)
                intr_dict = {
                    "width": intr.width,
                    "height": intr.height,
                    "ppx": intr.ppx,
                    "ppy": intr.ppy,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "model": str(intr.model),
                    "coeffs": list(intr.coeffs)
                }
                params_group[serial].create_dataset("intrinsics", data=np.string_(str(intr_dict)))
                # params_group[serial].create_dataset("start_time", data=np.string_(datetime.now(timezone.utc).isoformat()), compression='gzip')

            audio_thread = threading.Thread(target=self.record_audio)
            audio_thread.start()

            print("Recording... Press Ctrl+C to stop.")
            start_time = datetime.now(timezone.utc).timestamp()
            try:
                while not self.stop_event.is_set():
                    for i, pipeline in enumerate(pipelines):
                        frames = pipeline.wait_for_frames()
                        aligned = self.align.process(frames)
                        color = aligned.get_color_frame()
                        depth = aligned.get_depth_frame()
                        if not color or not depth:
                            continue
                        serial = serial_numbers[i]
                        idx = frame_counters[serial]
                        ts = datetime.now(timezone.utc).timestamp()  - start_time

                        color_image = np.asanyarray(color.get_data()).astype(np.uint8)
                        depth_image = np.asanyarray(depth.get_data()).astype(np.uint16)

                        color_groups[serial].create_dataset(str(idx), data=color_image, compression='gzip', compression_opts=9, chunks=True)
                        depth_groups[serial].create_dataset(str(idx), data=depth_image, compression='gzip', compression_opts=9, chunks=True)
                        timestamp_groups[serial].resize((idx + 1,))
                        timestamp_groups[serial][idx] = ts

                        frame_counters[serial] += 1

            except KeyboardInterrupt:
                print("Stopping...")

            self.stop_event.set()
            audio_thread.join()
            for pipeline in pipelines:
                pipeline.stop()

        print("Recording complete.")

# Run it:
if __name__ == "__main__":
    id = input("Whats participant num\n")
    recorder = MultiCameraRecorder(participant_num=id)
    recorder.record_video_streams()
