import streamlit as st
from facial_emotion import detect_facial_emotion
from audio_emotion import detect_audio_emotion
from text_emotion import detect_text_emotion
from context_profile import get_user_context
from fusion_engine import fuse_emotions

st.title("🧠 Real-Time Multi-Modal Emotion Recognition")

user_text = st.text_input("Enter how you're feeling:")

if st.button("Detect Emotion"):
    with st.spinner("Detecting emotions..."):
        facial, _ = detect_facial_emotion()
        audio = detect_audio_emotion()
        text = detect_text_emotion(user_text)
        context = get_user_context()

        st.write(f"Facial Emotion: {facial}")
        st.write(f"Audio Emotion: {audio}")
        st.write(f"Text Emotion: {text}")
        st.write(f"User Context: {context}")

        final_emotion = fuse_emotions(facial, audio, text, context)
        st.success(f"🧠 Final Emotion (Context-Aware): {final_emotion}")
