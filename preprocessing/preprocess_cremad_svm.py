import os
import numpy as np
import joblib
import tensorflow_datasets as tfds
import librosa
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = "features_svm/cremad"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_MFCC = 40

label_map = {
    'anger': 'angry', 'disgust': None, 'fear': None,
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'
}
label_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']

def extract_features(audio_array):
    try:
        mfcc = librosa.feature.mfcc(y=audio_array, sr=16000, n_mfcc=N_MFCC)
        return np.concatenate([np.mean(mfcc.T, axis=0), np.std(mfcc.T, axis=0)])
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def load_and_extract(use_cached=True):
    if use_cached:
        try:
            return (
                np.load(f"{OUTPUT_DIR}/X.npy"),
                np.load(f"{OUTPUT_DIR}/y.npy"),
                joblib.load(f"{OUTPUT_DIR}/label_encoder.pkl")
            )
        except:
            print("Cache not found, reprocessing...")

    ds = tfds.load("crema_d", split="train+validation+test", shuffle_files=False)
    X, y = [], []

    for example in tfds.as_numpy(ds):
        raw = label_names[example["label"]]
        mapped = label_map.get(raw)
        if mapped is None: continue
        feat = extract_features(example["audio"].astype(np.float32))
        if feat is not None:
            X.append(feat)
            y.append(mapped)

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    np.save(f"{OUTPUT_DIR}/X.npy", X)
    np.save(f"{OUTPUT_DIR}/y.npy", y_enc)
    joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")

    return X, y_enc, le

if __name__ == "__main__":
    load_and_extract(use_cached=False)
