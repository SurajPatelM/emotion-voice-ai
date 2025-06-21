import joblib
import numpy as np

MODEL_PATH = "models/random_forest_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_emotion(model, sample, label_encoder):
    proba = model.predict_proba(sample)[0]  # Probabilities for all classes
    top_idx = np.argmax(proba)
    label = label_encoder.inverse_transform([top_idx])[0]
    confidence = proba[top_idx]
    return label, confidence