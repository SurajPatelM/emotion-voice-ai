import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.ticker as ticker
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

# Paths
MODEL_PATH = "models/cnn_spectrogram_combined_final.h5"
ENCODER_PATH = "encoders/cnn_spectrogram_classes.npy"
SAMPLE_RATE = 22050
DURATION = 4
N_MELS = 64
FIXED_FRAMES = 173

def extract_melspectrogram(audio, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS, fixed_frames=FIXED_FRAMES):
    target_len = sr * duration
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    if mel_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_db.shape[1] > fixed_frames:
        mel_db = mel_db[:, :fixed_frames]

    return mel_db

def plot_spectrogram(mel_db, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, x_axis='time', y_axis='log', cmap='coolwarm')
    plt.colorbar(format=ticker.FormatStrFormatter('+%.0f dB'))
    plt.title(f'Mel-spectrogram: {title}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Spectrogram saved to: {save_path}")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained CNN model not found.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_encoder():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("CNN label encoder not found.")
    return np.load(ENCODER_PATH)

def infer_on_file(wav_path):
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return

    print(f"Extracting features from: {wav_path}")
    try:
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        mel = extract_melspectrogram(audio)
        sample = mel[np.newaxis, ..., np.newaxis]  # shape: (1, 64, 173, 1)
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return

    # Visualize
    fname = os.path.basename(wav_path).replace(".wav", "")
    plot_spectrogram(mel, title=os.path.basename(wav_path), save_path=f"results/spectrogram_{fname}.png")

    print("Loading model and encoder...")
    model = load_model()
    classes = load_encoder()

    # Predict
    proba = model.predict(sample)[0]
    top_idx = np.argmax(proba)
    predicted_label = classes[top_idx]
    confidence = proba[top_idx]

    print("Prediction result:")
    print(f"Predicted emotion: {predicted_label}")
    print(f"Confidence score : {confidence * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_cnn.py path/to/audio.wav")
    else:
        infer_on_file(sys.argv[1])
