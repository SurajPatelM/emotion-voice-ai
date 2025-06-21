import os
import numpy as np
import tensorflow_datasets as tfds
import librosa
from sklearn.preprocessing import LabelEncoder
import joblib

n_mfcc = 40
sr = 16000
output_dir = "features/cremad"
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

def extract_features(audio_array, n_mfcc=40):
    try:
        audio = audio_array.astype(np.float32)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        return np.concatenate((mfccs_mean, mfccs_std))
    except Exception as e:
        print(f"Error extracting MFCCs: {e}")
        return None

def load_and_extract_features(data_dir=None, use_cached=True):
    if use_cached:
        try:
            X = np.load(os.path.join(output_dir, "X.npy"))
            y = np.load(os.path.join(output_dir, "y.npy"))
            le = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))
            print(f"Loaded cached CREMA-D features from '{output_dir}/'")
            return X, y, le
        except Exception:
            print("Cache not found or corrupt. Reprocessing...")

    print("Loading CREMA-D dataset from TFDS...")
    ds = tfds.load("crema_d", split="train+validation+test", shuffle_files=False)
    features, labels = [], []

    for example in tfds.as_numpy(ds):
        label_index = example['label']
        raw_label = label_names[label_index]
        mapped_label = label_mapping.get(raw_label)

        if mapped_label is None:
            continue

        mfcc = extract_features(example['audio'], n_mfcc=n_mfcc)
        if mfcc is not None:
            features.append(mfcc)
            labels.append(mapped_label)

    X = np.array(features)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    print(f"Processed and saved {len(y)} samples to '{output_dir}/'")
    print(f"Label distribution: {dict(zip(le.classes_, np.bincount(y)))}")
    return X, y, le

if __name__ == "__main__":
    load_and_extract_features(use_cached=False)
