# Emotion-Aware Voice Companion

A machine learning pipeline for detecting emotions in speech using RAVDESS audio dataset. Built with Python, librosa, and scikit-learn.

## Features
- Extract MFCCs from `.wav` files
- Train a Random Forest model
- Predict emotions from new voice samples

## Usage

python main.py          # Train the model
python infer.py path/to/audio.wav  # Predict emotion
