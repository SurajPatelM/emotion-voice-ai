import os
import numpy as np
import librosa
import tensorflow_datasets as tfds
from sklearn.preprocessing import LabelEncoder
import joblib

output_dir = "features_cnn/cremad"
os.makedirs(output_dir, exist_ok=True)

label_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
label_mapping = {
    'anger': 'angry',
    'disgust': None,
    'fear': None,
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad'
}

def extract_melspec(audio, sr=16000, n_mels=64, duration=4):
    audio = librosa.util.fix_length(audio, sr * duration)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
    return mel_db

def load_and_extract_features(use_cached=True):
    if use_cached and os.path.exists(os.path.join(output_dir, "X.npy")):
        X = np.load(os.path.join(output_dir, "X.npy"))
        y = np.load(os.path.join(output_dir, "y.npy"))
        le = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))
        print("Loaded cached CREMA-D spectrogram features")
        return X, y, le

    print("Loading CREMA-D from TFDS...")
    ds = tfds.load("crema_d", split="train+validation+test", shuffle_files=False)
    features, labels = [], []

    for example in tfds.as_numpy(ds):
        label_index = example['label']
        label = label_mapping.get(label_names[label_index])
        if label is None:
            continue
        audio = example['audio'].astype(np.float32)
        mel = extract_melspec(audio)
        if mel is not None:
            features.append(mel)
            labels.append(label)

    X = np.array(features)[..., np.newaxis]  # Add channel dim
    le = LabelEncoder()
    y = le.fit_transform(labels)
    y = tf.keras.utils.to_categorical(y)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    print(f"Processed CREMA-D: {len(y)} samples")
    return X, y, le
