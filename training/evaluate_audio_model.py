import pickle
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load model
with open("models/emotion_audio.pkl", "rb") as f:
    model = pickle.load(f)

# Load metadata
metadata = pd.read_csv("data/audio_metadata.csv")

# Extract features
features = []
labels = []

for _, row in metadata.iterrows():
    try:
        y, sr = librosa.load(row["filepath"], sr=None)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfccs = np.mean(mel_db.T, axis=0)
        features.append(mfccs)
        labels.append(row["emotion"])
    except Exception as e:
        print(f"Error processing {row['filepath']}: {e}")

X = np.array(features)
y_true = np.array(labels)

# Encode true labels to match model output
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_true)
y_pred_encoded = model.predict(X)

# Decode predictions to strings
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluation
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
