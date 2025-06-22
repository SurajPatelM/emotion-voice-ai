import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow_datasets as tfds
import warnings

warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 22050
DURATION = 4
N_MELS = 64
FIXED_FRAMES = 173
VALID_LABELS = {'angry', 'happy', 'neutral', 'sad'}
RAVDESS_PATH = "data/RAVDESS"
os.makedirs("results", exist_ok=True)


def extract_melspectrogram(audio, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS, fixed_frames=FIXED_FRAMES, augment=False):
    target_len = sr * duration

    if augment:
        if np.random.rand() > 0.5:
            shift = np.random.randint(sr * -0.2, sr * 0.2)
            audio = np.roll(audio, shift)
        if np.random.rand() > 0.5:
            steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    if mel_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_db.shape[1] > fixed_frames:
        mel_db = mel_db[:, :fixed_frames]

    return mel_db


def load_ravdess():
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
                        for _ in range(3):  # Original + 2 augmentations
                            mel = extract_melspectrogram(audio, augment=_ > 0)
                            X.append(mel)
                            y.append(label)
                    except Exception as e:
                        print(f"RAVDESS error: {e} on {path}")
    return np.array(X)[..., np.newaxis], y


def load_cremad():
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
                for _ in range(3):
                    mel = extract_melspectrogram(audio, sr=16000, augment=_ > 0)
                    X.append(mel)
                    y.append(label)
            except Exception as e:
                print(f"CREMA-D error: {e}")
    return np.array(X)[..., np.newaxis], y


def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return focal_loss_fixed


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=focal_loss(gamma=2, alpha=0.25),
                  metrics=['accuracy'])
    return model


def calculate_class_weights(y):
    y_int = np.argmax(y, axis=1)
    class_counts = Counter(y_int)
    total = len(y_int)
    num_classes = y.shape[1]
    return {
        i: total / (num_classes * class_counts[i])
        for i in class_counts
    }


def plot_training(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/cnn_spectrogram_training.png")
    plt.show()


def evaluate_model(model, X_test, y_test, encoder):
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/cnn_spectrogram_confusion_matrix.png")
    plt.show()


def main():
    print("Loading RAVDESS...")
    X_rav, y_rav = load_ravdess()

    print("Loading CREMA-D...")
    X_crema, y_crema = load_cremad()

    X = np.concatenate([X_rav, X_crema], axis=0)
    y_raw = np.array(y_rav + y_crema)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    y_cat = to_categorical(y_encoded)

    print(f"Total samples: {X.shape[0]}, Classes: {encoder.classes_}")
    print(f"Class distribution: {dict(Counter(y_raw))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.15, stratify=y_encoded, random_state=42
    )

    model = build_model(X.shape[1:], num_classes=y_cat.shape[1])
    class_weight = calculate_class_weights(y_train)

    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy', verbose=1),
        ModelCheckpoint("cnn_spectrogram_best.h5", save_best_only=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    ]

    print("Training CNN...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    plot_training(history)
    evaluate_model(model, X_test, y_test, encoder)

    model.save("cnn_spectrogram_combined_final.h5")
    np.save("cnn_spectrogram_classes.npy", encoder.classes_)
    print("Model and label encoder saved.")


if __name__ == "__main__":
    main()
