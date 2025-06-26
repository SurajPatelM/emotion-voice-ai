import os
import sys
import numpy as np
import joblib
import librosa
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 40
N_MELS = 64
FIXED_FRAMES = 173

SVM_MODEL = "models/svm_model.pkl"
SVM_SCALER = "encoders/svm_scaler.pkl"
SVM_ENCODER = "encoders/svm_label_encoder.pkl"
SVM_PCA = "encoders/svm_pca.pkl"

RF_MODEL = "models/random_forest_model.pkl"
RF_SCALER = "encoders/scaler_rf.pkl"
RF_ENCODER = "encoders/label_encoder.pkl"

CNN_MODEL = "models/cnn_spectrogram_combined_final.h5"
CNN_ENCODER = "encoders/cnn_spectrogram_classes.npy"


def extract_mfcc_stats(audio, sr=SAMPLE_RATE):
    target_len = sr * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def stats(x):
        return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])

    return np.concatenate([stats(mfcc), stats(delta), stats(delta2)])

def extract_rf_features(audio, sr=SAMPLE_RATE):
    try:
        target_len = sr * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0)
        return np.mean(features, axis=1)
    except Exception as e:
        print(f"Error extracting RF features: {e}")
        return None

def extract_melspectrogram(audio, sr=SAMPLE_RATE):
    target_len = sr * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    if mel_db.shape[1] < FIXED_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_FRAMES - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    return mel_db


def infer_svm(audio):
    model = joblib.load(SVM_MODEL)
    scaler = joblib.load(SVM_SCALER)
    pca = joblib.load(SVM_PCA)
    encoder = joblib.load(SVM_ENCODER)

    features = extract_mfcc_stats(audio)
    sample = features.reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    sample_pca = pca.transform(sample_scaled)

    proba = model.predict_proba(sample_pca)[0]
    idx = np.argmax(proba)
    return "SVM", encoder.inverse_transform([idx])[0], proba[idx]

def infer_randomforest(audio):
    model = joblib.load(RF_MODEL)
    scaler = joblib.load(RF_SCALER)
    encoder = joblib.load(RF_ENCODER)

    features = extract_rf_features(audio)
    if features is None:
        raise ValueError("Random Forest feature extraction failed.")
    
    sample = features.reshape(1, -1)
    sample_scaled = scaler.transform(sample)

    proba = model.predict_proba(sample_scaled)[0]
    idx = np.argmax(proba)
    return "Random Forest", encoder.inverse_transform([idx])[0], proba[idx]

def infer_cnn(audio, wav_path):
    model = tf.keras.models.load_model(CNN_MODEL, compile=False)
    classes = np.load(CNN_ENCODER)

    mel = extract_melspectrogram(audio)
    sample = mel[np.newaxis, ..., np.newaxis]

    fname = os.path.basename(wav_path).replace(".wav", "")
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=SAMPLE_RATE, x_axis='time', y_axis='log', cmap='coolwarm')
    plt.colorbar(format=ticker.FormatStrFormatter('+%.0f dB'))
    plt.title(f'Mel-spectrogram: {os.path.basename(wav_path)}')
    plt.tight_layout()
    plt.savefig(f"results/spectrogram_{fname}.png")
    plt.close()

    proba = model.predict(sample)[0]
    idx = np.argmax(proba)
    return "CNN", classes[idx], proba[idx]


def run_all_models(wav_path):
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return

    print(f"Processing file: {wav_path}")
    try:
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Failed to load audio: {e}")
        return

    results = []

    try:
        results.append(infer_svm(audio))
    except Exception as e:
        print(f"SVM failed: {e}")

    try:
        results.append(infer_randomforest(audio))
    except Exception as e:
        print(f"Random Forest failed: {e}")

    try:
        results.append(infer_cnn(audio, wav_path))
    except Exception as e:
        print(f"CNN failed: {e}")

    print("\n=== Inference Results ===")
    for model_name, label, conf in results:
        print(f"{model_name} → {label} ({conf * 100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_all.py path/to/audio.wav")
    else:
        run_all_models(sys.argv[1])
