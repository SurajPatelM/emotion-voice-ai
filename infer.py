import os
import sys
import numpy as np
from preprocess import extract_features
from model import load_model, predict_emotion
from sklearn.preprocessing import LabelEncoder

# Define a fixed label encoder for decoding (same as training classes)
# EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
EMOTIONS = ['angry', 'happy', 'neutral', 'sad']
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)

def infer_on_file(wav_path):
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return
    features = extract_features(wav_path)
    if features is None:
        print("Failed to extract features from the file.")
        return
    sample = features.reshape(1, -1)
    model = load_model()
    prediction = predict_emotion(model, sample, label_encoder)
    print(f"Predicted emotion: {prediction}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer.py path/to/audio.wav")
    else:
        infer_on_file(sys.argv[1])
