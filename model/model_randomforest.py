import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "trained_models/random_forest.pkl"

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model, X_test, y_test

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
