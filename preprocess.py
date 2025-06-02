#This is for CREMA_D
# import os
# import numpy as np
# import librosa
# from sklearn.preprocessing import LabelEncoder

# emotion_map = {
#     'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
#     'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
# }

# def extract_features(file_path, n_mfcc=40):
#     try:
#         audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
#         return np.mean(mfccs.T, axis=0)
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None

# def load_and_extract_features(data_dir):
#     features, labels = [], []
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".wav"):
#             parts = file_name.split("_")
#             emotion = emotion_map.get(parts[2])
#             if emotion:
#                 file_path = os.path.join(data_dir, file_name)
#                 mfccs = extract_features(file_path)
#                 if mfccs is not None:
#                     features.append(mfccs)
#                     labels.append(emotion)

#     features = np.array(features)
#     labels = np.array(labels)
#     label_encoder = LabelEncoder()
#     labels_encoded = label_encoder.fit_transform(labels)
#     return features, labels_encoded, label_encoder



#For RAVDESS
import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder

# Map RAVDESS emotion codes to labels
ravdess_emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
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

    # Walk through all subfolders (Actor_01, Actor_02, etc.)
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if file_name.endswith(".wav"):
                parts = file_name.split("-")
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion = ravdess_emotion_map.get(emotion_code)
                    if emotion in ['happy', 'sad', 'angry', 'neutral']:  # Use only core classes
                        file_path = os.path.join(root, file_name)
                        mfccs = extract_features(file_path)
                        if mfccs is not None:
                            features.append(mfccs)
                            labels.append(emotion)

    features = np.array(features)
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    print(f"Loaded {len(labels)} samples: {dict(zip(label_encoder.classes_, np.bincount(labels_encoded)))}")
    return features, labels_encoded, label_encoder

