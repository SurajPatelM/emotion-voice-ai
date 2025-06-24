# Emotion Voice Detector

A machine learning pipeline for detecting emotions in speech using two popular datasets — **RAVDESS** and **CREMA-D**. Built with **Python**, **librosa**, and **scikit-learn**, the system supports multiple model architectures for robust emotion recognition.

## Features

* Preprocessing and normalization of **RAVDESS** and **CREMA-D** audio datasets
* Extraction of **MFCCs** and **Spectrograms** for model training
* Support for **4 emotion classes**: `neutral`, `happy`, `angry`, `sad`
* Train and infer using three models:

  * **Random Forest**
  * **Support Vector Machine (SVM)**
  * **Convolutional Neural Network (CNN)**
* Spectrogram visualization for CNN-based predictions

## Demo

Watch the system in action:

[Click here to watch the demo video on YouTube](https://www.youtube.com/watch?v=L7qplengaVk)


## Model Training

Train each model with the respective script:


python train/train_randomforest.py     # Train Random Forest
python train/train_svm.py              # Train SVM
python train/train_cnn.py              # Train CNN


## Inference

Predict emotions from new `.wav` audio files using:


python infer/infer_randomforest.py path/to/audio.wav   # Predict with Random Forest
python infer/infer_svm.py path/to/audio.wav            # Predict with SVM 
python infer/infer_cnn.py path/to/audio.wav            # Predict with CNN

To predict the emotion from all the three models at once:

./run_all_inference.sh Path/to/audio.wav


## Datasets

* [RAVDESS](https://zenodo.org/record/1188976) – Ryerson Audio-Visual Database of Emotional Speech and Song
* [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) – Crowd-sourced Emotional Multimodal Actors Dataset

Both datasets undergo consistent preprocessing to enable seamless model training and evaluation.


