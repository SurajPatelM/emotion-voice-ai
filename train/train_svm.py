import os
import numpy as np
import librosa
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
import warnings


warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 40
RAVDESS_PATH = "data/RAVDESS"
VALID_LABELS = {'angry', 'happy', 'neutral', 'sad'}

# Output paths
os.makedirs("models", exist_ok=True)
os.makedirs("encoders", exist_ok=True)
os.makedirs("results", exist_ok=True)

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

def load_ravdess_features():
    emotion_map = {
        '01': 'neutral', '02': None, '03': 'happy', '04': 'sad',
        '05': 'angry', '06': None, '07': None, '08': None
    }
    X, y = [], []
    for root, _, files in os.walk(RAVDESS_PATH):
        for fname in files:
            if fname.endswith(".wav"):
                parts = fname.split("-")
                emo_code = parts[2]
                label = emotion_map.get(emo_code)
                if label in VALID_LABELS:
                    path = os.path.join(root, fname)
                    try:
                        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
                        feat = extract_features(audio)
                        X.append(feat)
                        y.append(label)
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
    return X, y

def load_cremad_features():
    label_map = {
        'anger': 'angry', 'disgust': None, 'fear': None,
        'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'
    }
    label_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    ds = tfds.load("crema_d", split="train+validation+test", shuffle_files=False)
    X, y = [], []
    for example in tfds.as_numpy(ds):
        label_idx = example['label']
        raw_label = label_names[label_idx]
        label = label_map.get(raw_label)
        if label in VALID_LABELS:
            try:
                audio = example['audio'].astype(np.float32)
                feat = extract_features(audio, sr=16000)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"CREMA-D error: {e}")
    return X, y

def main():
    print("Extracting RAVDESS features...")
    X_rav, y_rav = load_ravdess_features()
    print("Extracting CREMA-D features...")
    X_crema, y_crema = load_cremad_features()

    X = np.array(X_rav + X_crema)
    y_raw = np.array(y_rav + y_crema)

    print("Label distribution:", dict(Counter(y_raw)))

    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    joblib.dump(label_encoder, "encoders/svm_label_encoder.pkl")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "encoders/svm_scaler.pkl")

    # Dimensionality reduction
    pca = PCA(n_components=50)
    X_reduced = pca.fit_transform(X_scaled)
    joblib.dump(pca, "encoders/svm_pca.pkl")  # Save PCA for inference

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
    )

    print("Training SVM model...")
    # model = SVC(kernel="poly", probability=True, class_weight='balanced')
    model = SVC(kernel="rbf", C=12, gamma=0.001, probability=True, class_weight='balanced')
    model.fit(X_train, y_train)
    joblib.dump(model, "models/svm_model.pkl")

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/svm_confusion_matrix.png")
    plt.show()
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    plt.figure(figsize=(6, 4))
    plt.bar(['Train Accuracy', 'Test Accuracy'], [train_acc, test_acc], color=['skyblue', 'orange'])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("SVM Accuracy Comparison")
    plt.tight_layout()
    plt.savefig("results/svm_accuracy_comparison.png")
    plt.show()

    print("Model, scaler, and label encoder saved.")


    print("Model, scaler, and label encoder saved.")

if __name__ == "__main__":
    main()
