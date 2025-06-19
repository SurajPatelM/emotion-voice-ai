from model.model_randomforest import train_model, evaluate_model, predict_emotion
from preprocessing.preprocess_ravdess import load_and_extract_features
import os

if __name__ == "__main__":
    data_dir = "data/RAVDESS"  # Replace with your actual dataset path

    # Load features and labels
    print("Loading and extracting features...")
    X, y, label_encoder = load_and_extract_features(data_dir)

    # Train and evaluate model
    print("Training model...")
    model, X_test, y_test = train_model(X, y)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, label_encoder)

    # Simulate new sample prediction
    print("Predicting on new data sample...")
    sample = X_test[0].reshape(1, -1)
    predicted_label = predict_emotion(model, sample, label_encoder)
    print("Predicted emotion:", predicted_label)
