import pyaudio
import numpy as np
import matplotlib.pyplot as plt

import wave
import time

FORMAT = pyaudio.paInt16  # 16-bit
CHANNELS = 1              # Mono channel
RATE = 44100     # Samples taken per second
CHUNK = 4096              # Buffer size
RECORD_SECONDS = 5        # Recording time (in seconds)

def check_inputs():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']} - Input Channels: {info['maxInputChannels']}")

    info = p.get_default_input_device_info()
    print("Default Device", info)

def record_audio(index = 0, filename ="videos/recorded_audio.wav"):
    # Initialize PyAudio
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    
    # Store audio data
    frames = []
    timestamps = []

    start_time = time.time()

    # Record audio data
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        timestamps.append(time.time() - start_time)
        frames.append(np.frombuffer(data, dtype=np.int16))


    print("Recording finished")
    # Stop and close the audio stream

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Saving as a .wav file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def plot_audio_data(frames):

    audio_data = np.hstack(frames)

    # Resample the audio from 48kHz to 16kHz using librosa.resample
    time_audio = np.linspace(0, len(audio_data) / RATE, num=len(audio_data))
    plt.plot(time_audio, audio_data)

def constant_stream(index):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=index,
                        frames_per_buffer=CHUNK)

    plt.ion()
    # Create two subplots - one for time domain, one for frequency domain
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time domain setup
    x_time = np.linspace(0, CHUNK / RATE, CHUNK)
    line_time, = ax1.plot(x_time, np.zeros(CHUNK))
    ax1.set_ylim(-500, 500)
    ax1.set_title('Audio Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Frequency domain setup
    x_freq = np.fft.rfftfreq(CHUNK, 1/RATE)
    line_fft, = ax2.plot(x_freq, np.zeros(len(x_freq)))
    ax2.set_xlim(0, RATE / 2)  # Frequency range up to Nyquist frequency
    ax2.set_title('FFT Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_yscale('log')  # Log scale for better visualization

    plt.tight_layout()
    
    try:
        for _ in range(1000):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Update time domain plot
            line_time.set_ydata(audio_data)
            
            # Compute and update FFT plot
            fft_data = np.abs(np.fft.rfft(audio_data))
            line_fft.set_ydata(fft_data)
            
            # Adjust y-axis limit for FFT to better visualize
            if _ % 10 == 0:  # Only adjust occasionally for performance
                ax2.set_ylim(max(1, np.min(fft_data)), max(10, np.max(fft_data)*1.1))
                
            fig.canvas.draw()
            fig.canvas.flush_events()
            
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


audio = pyaudio.PyAudio()
constant_stream(11)
audio.terminate()