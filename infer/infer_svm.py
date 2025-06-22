#!/usr/bin/env python3
"""
SVM Emotion Recognition Inference Script
Loads a trained SVM model and performs inference on audio files
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import librosa
import joblib
from model import model_svm
import warnings
warnings.filterwarnings('ignore')

class SVMEmotionRecognition:
    def __init__(self, model_path='trained_models/svm_model.pkl', scaler_path='encoders/scaler.pkl', label_encoder_path='encoders/label_encoder.pkl', sample_rate=22050, duration=4):
        """
        Initialize the inference class
        
        Args:
            model_path (str): Path to the trained model file
            scaler_path (str): Path to the scaler file for feature scaling
            label_encoder_path (str): Path to the label encoder
            sample_rate (int): Sample rate for audio processing
            duration (int): Duration in seconds for audio clips
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.label_encoder_path = label_encoder_path
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Load model, scaler, and label encoder
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.label_encoder_path)
    
    def extract_features(self, file_path):
        try:
            print(f"Processing audio file: {file_path}")
        
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)

            # Extract only MFCC means (same as training)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            feature_vector = np.mean(mfcc.T, axis=0)  # shape: (40,)
            print(f"Feature vector shape: {feature_vector.shape}")
            return feature_vector

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def predict_single_file(self, file_path):
        """
        Predict emotion for a single audio file
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            tuple: (predicted_emotion, confidence) or (None, None) if error
        """
        features = self.extract_features(file_path)
        if features is None:
            return None, None

        # Scale features
        scaled_features = self.scaler.transform(features.reshape(1, -1))

        # Predict
        prediction = self.model.predict(scaled_features)
        predicted_emotion = self.label_encoder.inverse_transform(prediction)[0]
        return predicted_emotion
    
    def batch_predict(self, audio_files, output_file=None):
        """
        Predict emotions for multiple audio files
        
        Args:
            audio_files (list): List of audio file paths
            output_file (str): Optional CSV file to save results
            
        Returns:
            list: List of prediction results
        """
        results = []

        print(f"\nProcessing {len(audio_files)} audio files...")
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(file_path)}")

            emotion = self.predict_single_file(file_path)
            print(f"Predicted Emotion: {emotion} | Type: {type(emotion)}")  # <-- Add this line

            result = {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'predicted_emotion': emotion
            }
            results.append(result)


        # Save results to CSV if requested
        if output_file and results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")

        return results

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='SVM Emotion Recognition Inference')
    parser.add_argument('audio_path', help='Path to audio file or directory')
    parser.add_argument('--model', '-m', default='trained_models/svm_model.pkl', 
                        help='Path to trained model file (default: trained_models/svm_model.pkl)')
    parser.add_argument('--scaler', '-s', default='encoders/scaler.pkl', 
                        help='Path to scaler file (default: encoders/scaler.pkl)')
    parser.add_argument('--label-encoder', '-l', default='encoders/label_encoder.pkl', 
                        help='Path to label encoder file (default: encoders/label_encoder.pkl)')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all audio files in directory')
    parser.add_argument('--output', '-o', help='Output CSV file for batch processing')

    args = parser.parse_args()
    
    try:
        # Initialize inference
        inference = SVMEmotionRecognition(
            model_path=args.model, 
            scaler_path=args.scaler, 
            label_encoder_path=args.label_encoder
        )

        if os.path.isfile(args.audio_path):
            # Single file processing
            print(f"Processing single file: {args.audio_path}")
            emotion = inference.predict_single_file(args.audio_path)
            print(f"Predicted Emotion: {emotion}")
        
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
