# app.py

import streamlit as st
from text_emotion import predict_emotion_from_text
from audio_emotion import predict_emotion_from_audio, predict_live_audio_emotion
from facial_emotion import predict_emotion_from_image
from utils import majority_vote

st.set_page_config(
    page_title="Multi-Modal Emotion Detector", layout="centered")
st.markdown("<h1 style='text-align: center;'>😊 Emotion Recognition App</h1>",
            unsafe_allow_html=True)
st.markdown("---")

# Initialize results
text_result = None
audio_result = None
facial_result = None

# Tabs for input methods
tabs = st.tabs(["📝 Text", "🔊 Audio", "📷 Facial"])

# -------------------- TEXT TAB --------------------
with tabs[0]:
    st.subheader("📝 Text Emotion Detection")
    text_input = st.text_area("Enter any text here:")
    if st.button("Analyze Text"):
        if text_input.strip():
            text_result = predict_emotion_from_text(text_input)
            if text_result:
                st.success(f"Emotion Detected: **{text_result}**")
            else:
                st.error("Failed to detect emotion from text.")
        else:
            st.warning("Please type some text to analyze.")

# -------------------- AUDIO TAB --------------------
with tabs[1]:
    st.subheader("🎤 Audio Emotion Detection")
    audio_mode = st.radio("Choose audio input method:", [
                          "Upload .wav", "Record Live"])

    if audio_mode == "Upload .wav":
        uploaded_audio = st.file_uploader("Upload a .wav file", type=["wav"])
        if st.button("Analyze Uploaded Audio"):
            if uploaded_audio:
                audio_result = predict_emotion_from_audio(uploaded_audio)
                st.success(f"Emotion Detected: **{audio_result}**")
            else:
                st.warning("Upload a .wav file before analyzing.")

    elif audio_mode == "Record Live":
        if st.button("Record & Analyze Live Audio"):
            audio_result = predict_live_audio_emotion()
            if audio_result:
                st.success(f"Emotion Detected (Live): **{audio_result}**")
            else:
                st.error("Failed to detect emotion from live audio.")

# -------------------- FACIAL TAB --------------------
with tabs[2]:
    st.subheader("📸 Facial Emotion Detection")
    face_mode = st.radio("Select input method:", [
                         "Upload Image", "Use Webcam"])

    if face_mode == "Upload Image":
        uploaded_image = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"])
        if st.button("Analyze Image"):
            if uploaded_image:
                facial_result = predict_emotion_from_image(uploaded_image)
                st.success(f"Emotion Detected: **{facial_result}**")
            else:
                st.warning("Upload an image before analyzing.")
    elif face_mode == "Use Webcam":
        st.info("🚧 Webcam support coming soon! Please upload an image for now.")

# -------------------- MAJORITY VOTE --------------------
st.markdown("---")
if st.button("🧠 Get Final Emotion (Majority Vote)"):
    predictions = [e for e in [text_result, audio_result, facial_result] if e]
    if len(predictions) >= 2:
        final_emotion = majority_vote(predictions)
        st.success(f"✅ Final Emotion: **{final_emotion}** (via Majority Vote)")
    else:
        st.warning(
            "At least two emotion predictions are required for majority voting.")
