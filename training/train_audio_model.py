# train_audio_model.py

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from audio_utils import extract_features

DATASET_PATH = "data/audio"
MODEL_PATH = "models/emotion_audio.pkl"

X = []
y = []

print("[INFO] Loading and extracting audio features...")

# Loop through folders (each emotion label)
for emotion_label in os.listdir(DATASET_PATH):
    emotion_dir = os.path.join(DATASET_PATH, emotion_label)
    if os.path.isdir(emotion_dir):
        for file in os.listdir(emotion_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_dir, file)
                try:
                    features = extract_features(file_path)
                    if features.shape[0] == 128:  # Optional: sanity check
                        X.append(features)
                        y.append(emotion_label)
                except Exception as e:
                    print(f"[ERROR] Skipping {file}: {e}")

X = np.array(X)
y = np.array(y)

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("[INFO] Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")
