# audio_emotion.py

import os
import numpy as np
import joblib
import tempfile
from audio_utils import record_audio, extract_features

# Path to the trained model
MODEL_PATH = "models/emotion_audio.pkl"

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"[ERROR] Model not found at {MODEL_PATH}. Please train it first.")

model = joblib.load(MODEL_PATH)

# Predict from saved file path (used internally)


def predict_audio_emotion(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Predict from uploaded file (Streamlit support)


def predict_emotion_from_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    return predict_audio_emotion(tmp_path)

# Predict from live microphone input (Streamlit or CLI)


def predict_live_audio_emotion(temp_path="data/live_audio.wav", duration=4):
    record_audio(temp_path, duration=duration)
    return predict_audio_emotion(temp_path)
