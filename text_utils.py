# text_utils.py

import re
import string
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=5000)
        self.label_encoder = LabelEncoder()

    def prepare_features(self, df):
        X = self.vectorizer.fit_transform(df["text"])
        y = self.label_encoder.fit_transform(df["label"])
        return X, y

    def transform_text(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts)

    def inverse_label(self, y):
        return self.label_encoder.inverse_transform(y)


def predict_text_emotion(text):
    """Loads the saved model and returns predicted emotion for input text."""
    model_path = "models/emotion_text.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    label_encoder_path = "models/label_encoder.pkl"

    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path)):
        raise FileNotFoundError(
            "Text emotion model not found. Please train it first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)

    cleaned = vectorizer.transform([text])
    pred = model.predict(cleaned)
    return label_encoder.inverse_transform(pred)[0]
