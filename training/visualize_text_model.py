# visualize_text_model.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from text_utils import TextPreprocessor

# Load model, vectorizer, and label encoder from pickle
model_path = "models/emotion_text_model.pkl"
print("[INFO] Loading model from:", model_path)
model, vectorizer, label_encoder = joblib.load(model_path)

# Load dataset
df = pd.read_csv("data/text_emotion.csv")
df = df[["text", "label"]]

# Transform labels
y_true = label_encoder.transform(df["label"])

# Transform text features
X = vectorizer.transform(df["text"])

# Predict
y_pred = model.predict(X)

# Report
print("[INFO] Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap="YlGnBu")
plt.title("Text Emotion Model - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("models/text_model_confusion_matrix.png")
plt.show()
