import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import joblib
from preprocessing.preprocess_ravdess import extract_features  

# Constants for model and encoder paths
MODEL_PATH = "models/ensemble_model.pkl"
ENCODER_PATH = "encoders/label_encoder.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained ensemble model not found. Run training first.")
    return joblib.load(MODEL_PATH)

def load_encoder():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(" Label encoder not found.")
    return joblib.load(ENCODER_PATH)

def infer_on_file(wav_path):
    if not os.path.exists(wav_path):
        print(f" File not found: {wav_path}")
        return

    print(f" Extracting features from: {wav_path}")
    features = extract_features(wav_path)
    if features is None:
        print(" Feature extraction failed.")
        return

    sample = features.reshape(1, -1)

    print(" Loading model and encoder...")
    model = load_model()
    label_encoder = load_encoder()

    proba = model.predict_proba(sample)[0]
    top_idx = np.argmax(proba)
    predicted_label = label_encoder.inverse_transform([top_idx])[0]
    confidence = proba[top_idx]

    print(f" Predicted emotion: **{predicted_label}** with confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_ensemble.py path/to/audio.wav")
    else:
        infer_on_file(sys.argv[1])
