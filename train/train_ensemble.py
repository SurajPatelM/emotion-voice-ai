import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.preprocess_cremad import load_and_extract_features as load_cremad
from preprocessing.preprocess_ravdess import load_and_extract_features as load_ravdess


def plot_confusion(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    print("📊 Confusion matrix saved to results/confusion_matrix.png")


if __name__ == "__main__":
    crema_dir = "data/CREMA-D"
    ravdess_dir = "data/RAVDESS"

    print("📥 Loading features...")
    X_crema, y_crema, _ = load_cremad(crema_dir)
    X_ravdess, y_ravdess, _ = load_ravdess(ravdess_dir)

    print("🔗 Merging datasets...")
    X = np.concatenate((X_crema, X_ravdess), axis=0)
    y = np.concatenate((y_crema, y_ravdess), axis=0)

    label_names = ['angry', 'disgust', 'fearful', 'happy']
    y_str = [label_names[i] if isinstance(i, (int, np.integer)) else i for i in y]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)

    print(f"Label distribution: {dict(zip(label_encoder.classes_, np.bincount(y_encoded)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Model setup
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting='soft',  # use probabilities
        n_jobs=-1
    )

    print("🧠 Training ensemble model (VotingClassifier)...")
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Confusion matrix
    os.makedirs("results", exist_ok=True)
    plot_confusion(y_test, y_pred, label_encoder.classes_)

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(ensemble, "models/ensemble_model.pkl")
    joblib.dump(label_encoder, "encoders/label_encoder.pkl")

    print("✅ Ensemble model and encoder saved.")
