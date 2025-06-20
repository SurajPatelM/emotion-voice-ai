import sys
import os
import numpy as np
import joblib
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.preprocess_cremad import load_and_extract_features as load_cremad
from preprocessing.preprocess_ravdess import load_and_extract_features as load_ravdess


def train_model_with_gridsearch(X, y):
    print("Starting GridSearchCV with scoring='f1_weighted'...")

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X, y)
    print("Best hyperparameters found:")
    print(grid_search.best_params_)
    return grid_search.best_estimator_


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix_rf.png")
    plt.close()
    print("Confusion matrix saved to results/confusion_matrix_rf.png")


if __name__ == "__main__":
    crema_dir = "data/CREMA-D"
    ravdess_dir = "data/RAVDESS"

    print("Loading features...")
    X_crema, y_crema, _ = load_cremad(crema_dir)
    X_ravdess, y_ravdess, _ = load_ravdess(ravdess_dir)

    print("Merging datasets...")
    X = np.concatenate((X_crema, X_ravdess), axis=0)
    y = np.concatenate((y_crema, y_ravdess), axis=0)

    print(f"Total samples: {len(y)}")

    label_names = ['angry', 'happy', 'neutral', 'sad']
    y_str = [label_names[i] if isinstance(i, (int, np.integer)) else i for i in y]

    valid_labels = {'angry', 'happy', 'neutral', 'sad'}
    X_filtered, y_filtered = [], []
    for xi, yi in zip(X, y_str):
        if yi in valid_labels:
            X_filtered.append(xi)
            y_filtered.append(yi)

    X = np.array(X_filtered)
    y = np.array(y_filtered)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Final label distribution:", dict(zip(label_encoder.classes_, np.bincount(y_encoded))))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print("Training model...")
    model = train_model_with_gridsearch(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)

    os.makedirs("models", exist_ok=True)
    os.makedirs("encoders", exist_ok=True)
    joblib.dump(model, "models/random_forest_model.pkl")
    joblib.dump(label_encoder, "encoders/label_encoder.pkl")
    joblib.dump(scaler, "encoders/scaler.pkl")

    print("Predicting a sample...")
    sample = X_test[0].reshape(1, -1)
    predicted_label = label_encoder.inverse_transform(model.predict(sample))[0]
    print(f"Predicted emotion: {predicted_label}")
