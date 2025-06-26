#!/usr/bin/env python3


import os
import sys
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class RAVDESSInference:
    def __init__(self, model_path='trained_models/cnn_model.h5', sample_rate=22050, duration=4, n_mels=64):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Load model and labels
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = load_model(self.model_path)
        
        # Load label encoder classes
        label_file = self.model_path.replace('.h5', '_labels.npy')
        if os.path.exists(label_file):
            classes = np.load(label_file, allow_pickle=True)
            self.label_encoder.classes_ = classes
            print(f"Loaded {len(classes)} emotion classes: {list(classes)}")
        else:
            # Fallback to default RAVDESS emotions if label file not found
            print("Warning: Label file not found, using default RAVDESS emotions")
            self.label_encoder.classes_ = np.array([
                'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
            ])
        
        print("Model loaded successfully!")
    
    def extract_features(self, file_path):

        try:
            print(f"Processing audio file: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to fixed length
            target_length = self.sample_rate * self.duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            # Extract mel-spectrogram
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
    
    def predict_single_file(self, file_path, show_probabilities=True):

        if self.model is None:
            print("Error: Model not loaded!")
            return None, None
        
        # Extract features
        spec = self.extract_features(file_path)
        if spec is None:
            print(f"Could not process file: {file_path}")
            return None, None
        
        # Reshape for prediction (add batch and channel dimensions)
        spec = spec.reshape(1, spec.shape[0], spec.shape[1], 1)
        
        # Predict
        print("Making prediction...")
        prediction = self.model.predict(spec, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Get emotion label
        emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Display results
        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"File: {os.path.basename(file_path)}")
        print(f"Predicted Emotion: {emotion.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        if show_probabilities:
            print(f"\n{'='*30}")
            print("All emotion probabilities:")
            print(f"{'='*30}")
            
            # Sort probabilities in descending order
            prob_pairs = [(self.label_encoder.inverse_transform([i])[0], prob) 
                         for i, prob in enumerate(prediction[0])]
            prob_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for emotion_name, prob in prob_pairs:
                bar = '█' * int(prob * 50)  # Visual bar
                print(f"{emotion_name:>10}: {prob:.4f} ({prob*100:5.2f}%) {bar}")
        
        return emotion, confidence
    
    def visualize_spectrogram(self, file_path, save_plot=False, output_dir='./'):

        spec = self.extract_features(file_path)
        if spec is not None:
            plt.figure(figsize=(14, 8))
            
            # Create spectrogram plot
            librosa.display.specshow(
                spec, 
                sr=self.sample_rate, 
                hop_length=512,
                x_axis='time', 
                y_axis='mel',
                cmap='viridis'
            )
            
            plt.colorbar(format='%+2.0f dB', label='Power (dB)')
            plt.title(f'Mel-spectrogram: {os.path.basename(file_path)}', fontsize=16)
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Mel Frequency', fontsize=12)
            plt.tight_layout()
            
            if save_plot:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                filename = os.path.splitext(os.path.basename(file_path))[0]
                save_path = os.path.join(output_dir, f"{filename}_spectrogram.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Spectrogram saved to: {save_path}")
            
            plt.show()
        else:
            print("Could not generate spectrogram")
    
    def batch_predict(self, audio_files, output_file=None):
 
        results = []
        
        print(f"\nProcessing {len(audio_files)} audio files...")
        print("="*60)
        
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(file_path)}")
            
            emotion, confidence = self.predict_single_file(file_path, show_probabilities=False)
            
            result = {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'predicted_emotion': emotion,
                'confidence': confidence
            }
            results.append(result)
            
            if emotion:
                print(f"Result: {emotion} (confidence: {confidence:.4f})")
            else:
                print("Result: Error processing file")
        
        # Save results to CSV if requested
        if output_file and results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='RAVDESS Emotion Recognition Inference')
    parser.add_argument('audio_path', help='Path to audio file or directory')
    parser.add_argument('--model', '-m', default='trained_models/cnn_model.h5', 
                       help='Path to trained model file (default: trained_models/cnn_model.h5)')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Show spectrogram visualization')
    parser.add_argument('--save-plot', '-s', action='store_true', 
                       help='Save spectrogram plot to file')
    parser.add_argument('--batch', '-b', action='store_true', 
                       help='Process all audio files in directory')
    parser.add_argument('--output', '-o', help='Output CSV file for batch processing')
    parser.add_argument('--no-probs', action='store_true', 
                       help='Hide probability breakdown')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        inference = RAVDESSInference(model_path=args.model)
        
        if os.path.isfile(args.audio_path):
            # Single file processing
            print(f"Processing single file: {args.audio_path}")
            
            if args.visualize:
                inference.visualize_spectrogram(args.audio_path, save_plot=args.save_plot)
            
            emotion, confidence = inference.predict_single_file(
                args.audio_path, 
                show_probabilities=not args.no_probs
            )
            
        elif os.path.isdir(args.audio_path) and args.batch:
            # Batch processing
            audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.aac')
            audio_files = []
            
            for root, dirs, files in os.walk(args.audio_path):
                for file in files:
                    if file.lower().endswith(audio_extensions):
                        audio_files.append(os.path.join(root, file))
            
            if not audio_files:
                print(f"No audio files found in {args.audio_path}")
                return
            
            results = inference.batch_predict(audio_files, args.output)
            
            # Summary
            print(f"\n{'='*60}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total files processed: {len(results)}")
            
            successful = [r for r in results if r['predicted_emotion'] is not None]
            if successful:
                print(f"Successful predictions: {len(successful)}")
                
                # Emotion distribution
                emotions = [r['predicted_emotion'] for r in successful]
                unique_emotions = list(set(emotions))
                
                print("\nEmotion distribution:")
                for emotion in unique_emotions:
                    count = emotions.count(emotion)
                    print(f"  {emotion}: {count} files")
        
        else:
            print("Error: Invalid path or missing --batch flag for directory processing")
            print("Use --help for usage information")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
