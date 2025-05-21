import cv2
import tempfile
import streamlit as st
from fer import FER
import os


def detect_facial_emotion():
    uploaded_file = st.file_uploader(
        "📹 Upload a short video (max 10s) for facial emotion detection", type=["mp4", "mov", "avi"]
    )
    if not uploaded_file:
        return "No Video Uploaded", None

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_file.read())
        temp_video_path = tmpfile.name

    # Initialize FER detector
    detector = FER(mtcnn=True)

    # Open the video file
    cap = cv2.VideoCapture(temp_video_path)

    emotion_results = []
    frame_count = 0
    max_frames = 150  # limit to ~5 seconds to keep things fast

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        frame_count += 1

        # Resize to reduce processing load (optional)
        resized_frame = cv2.resize(frame, (480, 360))
        result = detector.top_emotion(resized_frame)
        if result:
            emotion_label, score = result
            if score > 0.5:  # filter out weak predictions
                emotion_results.append(emotion_label)

    cap.release()
    os.remove(temp_video_path)

    if emotion_results:
        # Return the most common emotion
        final_emotion = max(set(emotion_results), key=emotion_results.count)
        return final_emotion, emotion_results
    else:
        return "No Clear Emotion Detected", None
