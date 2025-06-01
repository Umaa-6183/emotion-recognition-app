# predict_text_emotion.py

import joblib

# Load the model, vectorizer, and label_encoder
print("[INFO] Loading model...")
model, vectorizer, label_encoder = joblib.load("models/emotion_text_model.pkl")


def predict_emotion(text):
    """Transform input text and return predicted emotion."""
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    emotion = label_encoder.inverse_transform(prediction)[0]  # decode label
    return emotion


# CLI
if __name__ == "__main__":
    print("[INFO] Model loaded. Enter text to analyze (type 'quit' to exit).")
    while True:
        user_input = input("\nEnter text: ")
        if user_input.strip().lower() == "quit":
            print("[INFO] Exiting.")
            break
        if not user_input.strip():
            print("[WARN] Please enter non-empty text.")
            continue
        emotion = predict_emotion(user_input)
        print(f"[PREDICTED EMOTION]: {emotion}")
