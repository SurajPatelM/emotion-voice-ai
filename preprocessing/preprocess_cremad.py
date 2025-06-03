# This is for CREMA_D
import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

emotion_map = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

def extract_features(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_and_extract_features(data_dir):
    features, labels = [], []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".wav"):
            parts = file_name.split("_")
            emotion = emotion_map.get(parts[2])
            if emotion:
                file_path = os.path.join(data_dir, file_name)
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(emotion)

    features = np.array(features)
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return features, labels_encoded, label_encoder

