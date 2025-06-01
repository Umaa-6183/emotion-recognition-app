# train_text_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/text_emotion.csv")

# Rename columns for consistency (optional)
df = df.rename(columns={'label': 'emotion', 'text': 'text'})

# Check required columns exist
if 'text' not in df.columns or 'emotion' not in df.columns:
    raise ValueError("The CSV must contain 'text' and 'emotion' columns.")

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['emotion']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
with open("text_emotion.pkl", "wb") as f:
    pickle.dump(model, f)

with open("text_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("[INFO] Model and vectorizer saved as text_emotion.pkl and text_vectorizer.pkl.")
