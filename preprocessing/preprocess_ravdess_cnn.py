import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import joblib
import tensorflow as tf

output_dir = "features_cnn/ravdess"
os.makedirs(output_dir, exist_ok=True)

emotion_map = {
    '01': 'neutral', '02': None, '03': 'happy',
    '04': 'sad', '05': 'angry', '06': None,
    '07': None, '08': None
}

def extract_melspec(audio, sr=22050, n_mels=64, duration=4):
    audio = librosa.util.fix_length(audio, size=sr * duration)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
    return mel_db   

def load_and_extract_features(data_dir, use_cached=True):
    if use_cached and os.path.exists(os.path.join(output_dir, "X.npy")):
        X = np.load(os.path.join(output_dir, "X.npy"))
        y = np.load(os.path.join(output_dir, "y.npy"))
        le = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))
        print("Loaded cached RAVDESS spectrogram features")
        return X, y, le

    features, labels = [], []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".wav"):
                parts = fname.split("-")
                emo_code = parts[2]
                label = emotion_map.get(emo_code)
                if label is None:
                    continue
                path = os.path.join(root, fname)
                try:
                    audio, _ = librosa.load(path, sr=22050)
                    mel = extract_melspec(audio)
                    features.append(mel)
                    labels.append(label)
                except Exception as e:
                    print(f"Failed on {path}: {e}")

    X = np.array(features)[..., np.newaxis]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    y = tf.keras.utils.to_categorical(y)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    print(f"Processed RAVDESS: {len(y)} samples")
    return X, y, le
