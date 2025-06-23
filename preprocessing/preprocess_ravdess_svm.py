import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import joblib

OUTPUT_DIR = "features_svm/ravdess"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_MFCC = 40

emotion_map = {
    '01': 'neutral', '02': None, '03': 'happy',
    '04': 'sad', '05': 'angry', '06': None,
    '07': None, '08': None
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        return np.concatenate([np.mean(mfcc.T, axis=0), np.std(mfcc.T, axis=0)])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_and_extract(data_dir, use_cached=True):
    if use_cached:
        try:
            return (
                np.load(f"{OUTPUT_DIR}/X.npy"),
                np.load(f"{OUTPUT_DIR}/y.npy"),
                joblib.load(f"{OUTPUT_DIR}/label_encoder.pkl")
            )
        except:
            print("Cache not found, reprocessing...")

    features, labels = [], []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".wav"):
                emo = emotion_map.get(f.split("-")[2])
                if emo is None: continue
                path = os.path.join(root, f)
                feat = extract_features(path)
                if feat is not None:
                    features.append(feat)
                    labels.append(emo)

    X = np.array(features)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    np.save(f"{OUTPUT_DIR}/X.npy", X)
    np.save(f"{OUTPUT_DIR}/y.npy", y)
    joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")

    return X, y, le

if __name__ == "__main__":
    load_and_extract("data/RAVDESS", use_cached=False)
