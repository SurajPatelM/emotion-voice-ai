import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RAVDESSSpectrogramCNN:
    def __init__(self, data_path, sample_rate=22050, duration=4, n_mels=64):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.emotion_labels = {
            '01': 'neutral',
            '02': 'calm', 
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
    def extract_features(self, file_path, augment=False):
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Data augmentation during training
            if augment:
                # Random time shifting
                if np.random.random() > 0.5:
                    shift = np.random.randint(-len(audio)//10, len(audio)//10)
                    audio = np.roll(audio, shift)
                
                # Random noise addition
                if np.random.random() > 0.5:
                    noise = np.random.normal(0, 0.005, len(audio))
                    audio = audio + noise
                
                # Random pitch shifting
                if np.random.random() > 0.5:
                    n_steps = np.random.uniform(-2, 2)
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
            # Pad or truncate to fixed length
            target_length = self.sample_rate * self.duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            # Extract multiple features
            # 1. Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=self.n_mels,
                hop_length=512,
                n_fft=2048,
                fmin=0,
                fmax=sr//2
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def load_data(self, augment_data=True):
        features = []
        labels = []
        
        print("Loading RAVDESS dataset...")
        
        # First pass: collect all files
        all_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    parts = file.split('-')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        if emotion_code in self.emotion_labels:
                            all_files.append((file_path, emotion_code))
        
        print(f"Found {len(all_files)} audio files")
        
        # Process files
        for file_path, emotion_code in all_files:
            # Original sample
            spec = self.extract_features(file_path, augment=False)
            if spec is not None:
                features.append(spec)
                labels.append(self.emotion_labels[emotion_code])
                
                # Add augmented samples for training diversity
                if augment_data:
                    for _ in range(2):  # Add 2 augmented versions per sample
                        aug_spec = self.extract_features(file_path, augment=True)
                        if aug_spec is not None:
                            features.append(aug_spec)
                            labels.append(self.emotion_labels[emotion_code])
        
        print(f"Total samples after augmentation: {len(features)}")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Add channel dimension for CNN (height, width, channels)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for emotion, count in zip(unique, counts):
            print(f"{emotion}: {count}")
        
        return X, y_categorical, y
    
    def build_model(self, input_shape, num_classes):
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Global Average Pooling instead of Flatten
            GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # Use a lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=100, batch_size=16):
        print("Building model...")
        input_shape = X.shape[1:]  # (height, width, channels)
        num_classes = y.shape[1]
        
        self.model = self.build_model(input_shape, num_classes)
        print(self.model.summary())
        
        # Split data with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Improved callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_ravdess_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_format='h5'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint, reduce_lr],
            verbose=1,
            class_weight=self.calculate_class_weights(y_train)
        )
        
        return history
    
    def calculate_class_weights(self, y):
        y_integers = np.argmax(y, axis=1)
        class_weights = {}
        unique_classes = np.unique(y_integers)
        
        for cls in unique_classes:
            class_weights[cls] = len(y_integers) / (len(unique_classes) * np.sum(y_integers == cls))
        
        return class_weights
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, X_test, y_test, y_test_labels):
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return y_pred, y_pred_classes
    
    def predict_single_file(self, file_path):
        """Predict emotion for a single audio file"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Extract features
        spec = self.extract_features(file_path)
        if spec is None:
            print(f"Could not process file: {file_path}")
            return None
        
        # Reshape for prediction
        spec = spec.reshape(1, spec.shape[0], spec.shape[1], 1)
        
        # Predict
        prediction = self.model.predict(spec)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Get emotion label
        emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show all probabilities
        print("\nAll emotion probabilities:")
        for i, prob in enumerate(prediction[0]):
            emotion_name = self.label_encoder.inverse_transform([i])[0]
            print(f"{emotion_name}: {prob:.4f}")
        
        return emotion, confidence
    
    def visualize_spectrogram(self, file_path):
        """Visualize spectrogram of an audio file"""
        spec = self.extract_features(file_path)
        if spec is not None:
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                spec, 
                sr=self.sample_rate, 
                hop_length=512,
                x_axis='time', 
                y_axis='mel'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel-spectrogram: {os.path.basename(file_path)}')
            plt.tight_layout()
            plt.show()
    
    def save_model(self, filepath='ravdess_emotion_model.h5'):
        if self.model is not None:
            self.model.save(filepath, save_format='h5')
            # Also save label encoder
            np.save(filepath.replace('.h5', '_labels.npy'), self.label_encoder.classes_)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='ravdess_emotion_model.h5'):
        self.model = load_model(filepath)
        # Load label encoder classes
        label_file = filepath.replace('.h5', '_labels.npy')
        if os.path.exists(label_file):
            classes = np.load(label_file, allow_pickle=True)
            self.label_encoder.classes_ = classes
        print(f"Model loaded from {filepath}")

def main():
    # Initialize the classifier with improved parameters
    ravdess_path = "data/RAVDESS"  # Update this path
    classifier = RAVDESSSpectrogramCNN(ravdess_path, duration=4, n_mels=64)
    
    # Load and preprocess data with augmentation
    X, y, y_labels = classifier.load_data(augment_data=True)
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train the model with improved parameters
    history = classifier.train_model(X_train, y_train, epochs=50, batch_size=16)
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Evaluate on test set
    y_test_labels = classifier.label_encoder.inverse_transform(np.argmax(y_test, axis=1))
    classifier.evaluate_model(X_test, y_test, y_test_labels)
    
    # Save the model
    classifier.save_model()
    
    # Test inference on the specific file
    test_file = "/Users/gargi/Documents/gargikel/FAI/project/emotion-voice-ai/data/RAVDESS/Actor_01/03-01-03-02-01-01-01.wav"
    if os.path.exists(test_file):
        print(f"\n{'='*50}")
        print("TESTING SPECIFIC FILE 1:")
        print(f"{'='*50}")
        classifier.visualize_spectrogram(test_file)
        classifier.predict_single_file(test_file)
        
        # Also test with multiple predictions to see consistency
        print(f"\n{'='*30}")
        print("Multiple predictions (to check consistency):")
        print(f"{'='*30}")
        for i in range(3):
            print(f"Prediction {i+1}:")
            classifier.predict_single_file(test_file)
            print()

    # Test inference on the specific file
    test_file = "data/RAVDESS/Actor_01/03-01-03-02-01-01-01.wav"
    if os.path.exists(test_file):
        print(f"\n{'='*50}")
        print("TESTING SPECIFIC FILE 2:")
        print(f"{'='*50}")
        classifier.visualize_spectrogram(test_file)
        classifier.predict_single_file(test_file)
        
        # Also test with multiple predictions to see consistency
        print(f"\n{'='*30}")
        print("Multiple predictions (to check consistency):")
        print(f"{'='*30}")
        for i in range(3):
            print(f"Prediction {i+1}:")
            classifier.predict_single_file(test_file)
            print()

if __name__ == "__main__":
    main()