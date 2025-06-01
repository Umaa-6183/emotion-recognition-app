# facial_emotion.py

from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

# Webcam-based detection (for standalone use)


def detect_facial_emotion():
    cap = cv2.VideoCapture(0)
    print("Starting camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(
                frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            print("Emotion:", emotion)

            # Display emotion on the video
            cv2.putText(frame, emotion, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print("Error:", str(e))

        cv2.imshow('Real-Time Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Image upload support for Streamlit app


def predict_emotion_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        result = DeepFace.analyze(
            image_np, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"[ERROR] Facial Image Analysis Failed: {e}")
        return None
