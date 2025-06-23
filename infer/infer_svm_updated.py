import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import librosa
import joblib

# Constants
SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 40

# Paths
MODEL_PATH = "models/svm_model.pkl"
SCALER_PATH = "encoders/svm_scaler.pkl"
ENCODER_PATH = "encoders/svm_label_encoder.pkl"
PCA_PATH = "encoders/svm_pca.pkl"

def extract_features(audio, sr=SAMPLE_RATE, duration=DURATION, n_mfcc=N_MFCC):
    target_len = sr * duration
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def stats(x):
        return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])

    features = np.concatenate([stats(mfcc), stats(delta), stats(delta2)])
    return features

def load_model():
    return joblib.load(MODEL_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)

def load_encoder():
    return joblib.load(ENCODER_PATH)

def load_pca():
    return joblib.load(PCA_PATH)

def infer_on_file(wav_path):
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return

    print(f"Extracting features from: {wav_path}")
    try:
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        features = extract_features(audio)
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return

    sample = features.reshape(1, -1)

    # Load and apply transformations
    scaler = load_scaler()
    sample_scaled = scaler.transform(sample)

    pca = load_pca()
    sample_pca = pca.transform(sample_scaled)

    print("Loading model and encoder...")
    model = load_model()
    encoder = load_encoder()

    # Predict
    proba = model.predict_proba(sample_pca)[0]
    top_idx = np.argmax(proba)
    predicted_label = encoder.inverse_transform([top_idx])[0]
    confidence = proba[top_idx]

    print("Prediction result:")
    print(f"Predicted emotion: {predicted_label}")
    print(f"Confidence score : {confidence * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_svm.py path/to/audio.wav")
    else:
        infer_on_file(sys.argv[1])
