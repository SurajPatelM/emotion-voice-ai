#!/bin/bash

# Check if audio path is provided
if [ $# -ne 1 ]; then
  echo "Usage: ./run_all_inference.sh path/to/audio.wav"
  exit 1
fi

AUDIO_PATH="$1"

echo "Running SVM Inference..."
python infer/infer_svm_updated.py "$AUDIO_PATH"

echo "Running Random Forest Inference..."
python infer/infer_randomforest.py "$AUDIO_PATH"

echo "Running CNN Inference..."
python infer/infer_cnn.py "$AUDIO_PATH"
