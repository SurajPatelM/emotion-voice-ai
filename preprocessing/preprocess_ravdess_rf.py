import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import joblib

n_mfcc = 40
output_dir = "features/ravdess"
os.makedirs(output_dir, exist_ok=True)

ravdess_emotion_map = {
    '01': 'neutral',
    '02': None,     # calm
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': None,     # fearful
    '07': None,     # disgust
    '08': None      # surprised
}

def extract_features(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        return np.concatenate((mfccs_mean, mfccs_std))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_and_extract_features(data_dir, use_cached=True):
    if use_cached:
        try:
            X = np.load(os.path.join(output_dir, "X.npy"))
            y = np.load(os.path.join(output_dir, "y.npy"))
            le = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))
            print(f"Loaded cached RAVDESS features from '{output_dir}/'")
            return X, y, le
        except Exception:
            print("Cache not found or corrupt. Reprocessing...")

    print(f"Extracting RAVDESS features from: {data_dir}")
    features, labels = [], []

    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith(".wav"):
                parts = file_name.split("-")
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion = ravdess_emotion_map.get(emotion_code)
                    if emotion is None:
                        continue

                    file_path = os.path.join(root, file_name)
                    mfcc = extract_features(file_path)
                    if mfcc is not None:
                        labels.append(emotion)
                        features.append(mfcc)

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
    load_and_extract_features("data/RAVDESS", use_cached=False)
