import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

MODEL_PATH = "trained_models/svm_model.pkl"
SCALER_PATH = "encoders/scaler.pkl"

def train_model(X, y):
    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, class_weight='balanced'))
    ])

    # Hyperparameter grid
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto', 0.01, 0.001],
        'svc__kernel': ['rbf', 'poly']
    }

    # Grid search with 5-fold cross-validation
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(grid.best_estimator_, MODEL_PATH)

    # Save scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    scaler = grid.best_estimator_.named_steps['scaler']
    joblib.dump(scaler, SCALER_PATH)

    return grid.best_estimator_, X_test, y_test


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def load_model():
    return joblib.load(MODEL_PATH)


def predict_emotion(model, sample, label_encoder):
    pred = model.predict(sample)
    return label_encoder.inverse_transform(pred)[0]
