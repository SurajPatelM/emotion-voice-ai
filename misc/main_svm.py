from model import model_svm
from preprocessing.preprocess_ravdess import load_and_extract_features

# Load features and labels from RAVDESS
X, y, label_encoder = load_and_extract_features("data/RAVDESS")

# Train the SVM model (with grid search & class weights)
model, X_test, y_test = model_svm.train_model(X, y)

# Evaluate the model
model_svm.evaluate_model(model, X_test, y_test, label_encoder)

# Optional: Predict on one sample
sample = X_test[0].reshape(1, -1)
prediction = model_svm.predict_emotion(model, sample, label_encoder)
print("Predicted Emotion:", prediction)
