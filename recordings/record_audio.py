import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_voice(filename="recorded_sample.wav", duration=4, fs=16000):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"Saved recording to: {filename}")

if __name__ == "__main__":
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gargi_happy.wav")
    record_voice(output_path)
