# evaluate_text_model.py

import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/text_emotion.csv")
df = df.rename(columns={"label": "emotion", "text": "text"})

# Load model and vectorizer
with open("text_emotion.pkl", "rb") as f:
    model = pickle.load(f)

with open("text_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Prepare data
X = vectorizer.transform(df['text']).toarray()
y = df['emotion']

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
