# text_emotion.py

import joblib
# Assumes a valid text cleaning function is available
from text_utils import clean_text

# Load the trained model and vectorizer
MODEL_PATH = "models/text_emotion.pkl"
VECTORIZER_PATH = "models/text_vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as e:
    raise FileNotFoundError(f"[ERROR] Could not load model/vectorizer: {e}")


def predict_emotion_from_text(text):
    """
    Predicts emotion from raw text using the trained model and vectorizer.
    """
    try:
        cleaned_text = clean_text(text)
        features = vectorizer.transform([cleaned_text]).toarray()
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        print(f"[ERROR] Text Emotion Prediction Failed: {e}")
        return None
