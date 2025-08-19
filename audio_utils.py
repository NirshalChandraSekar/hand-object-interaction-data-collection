import pyaudio
import wave
import threading
import time
import matplotlib.pyplot as plt
import numpy as np


class AudioRecorder:
   """
   Handles audio recording using PyAudio in a separate thread.
   Records audio data and saves to a WAV file.
   """
   def __init__(self, output_file="", rate=48000, channels=2, chunk=4096):
       self.output_file = output_file
       self.rate = rate
       self.channels = channels
       self.chunk = chunk
       self.format = pyaudio.paInt16
       self.frames = []
       self.stop_event = threading.Event()
       self.audio_start_time = None


   def record(self):
       """
       Start recording audio until stop_event is set.
      
       """
       self.frames = []
       self.stop_event.clear()
       self.audio_start_time = time.time()


       p = pyaudio.PyAudio()
       stream = p.open(format=self.format,
                       channels=self.channels,
                       rate=self.rate,
                       input=True,
                        input_device_index=10,  # Adjust device name as needed
                       frames_per_buffer=self.chunk)


       while not self.stop_event.is_set():
           data = stream.read(self.chunk, exception_on_overflow=False)
           self.frames.append(data)


       stream.stop_stream()
       stream.close()
       p.terminate()


       wf = wave.open(self.output_file, 'wb')
       wf.setnchannels(self.channels)
       wf.setsampwidth(p.get_sample_size(self.format))
       wf.setframerate(self.rate)
       wf.writeframes(b''.join(self.frames))
       wf.close()


def check_input_device():
       """
       Checks all available audio input devices and prints their names.
       """


       p = pyaudio.PyAudio()


       print("Available audio input devices:")
       for i in range(p.get_device_count()):
           dev = p.get_device_info_by_index(i)
           if dev['maxInputChannels'] > 0:
               print(f"Index {i}: {dev['name']}: {dev['maxInputChannels']} channels")


       p.terminate()


def get_input_index_by_name(target_name='Yeti Nano: USB Audio'):
   p = pyaudio.PyAudio()
   for i in range(p.get_device_count()):
       dev = p.get_device_info_by_index(i)
       if dev['maxInputChannels'] > 0 and target_name.lower() in dev['name'].lower():
           p.terminate()
           return i
   p.terminate()
   return None


# This graphs realtime audio input devices
def graph_audio():
   frames = []
   p = pyaudio.PyAudio()
   stream = p.open(format=pyaudio.paInt16,
                   channels=2,
                   rate=44100,
                   input=True,
                   input_device_index=11,
                   frames_per_buffer=1024)
  
   # loop for 5 seconds and collect audio data
   for i in range(0, int(44100 / 1024 * 5)):
       data = stream.read(1024, exception_on_overflow=False)
       frames.append(data)


   stream.stop_stream()
   stream.close()
   p.terminate()


   # Convert frames to numpy array
   audio_data = b''.join(frames)
   audio_array = np.frombuffer(audio_data, dtype=np.int16)


   # If stereo, separate channels
   if stream._channels == 2:
       audio_array = audio_array.reshape(-1, 2)
       left = audio_array[:, 0]
       right = audio_array[:, 1]
   else:
       left = audio_array
       right = None


   # Plot waveform
   plt.figure(figsize=(12, 4))
   plt.plot(left, label="Left channel")
   if right is not None:
       plt.plot(right, label="Right channel", alpha=0.7)
   plt.title("Audio Waveform")
   plt.xlabel("Sample index")
   plt.ylabel("Amplitude")
   plt.legend()
   plt.show()



