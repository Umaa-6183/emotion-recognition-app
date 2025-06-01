import os
import pandas as pd

# Emotion code to label mapping (RAVDESS format)
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Base path to the audio files
base_dir = "data/audio_speech_actors_01-24"
metadata = []

print("[INFO] Scanning audio files...")

# Walk through all subdirectories and collect .wav files
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            try:
                parts = file.split("-")
                emotion_code = parts[2]  # 03-01-**08**-01-02-02-01.wav
                emotion = emotion_map.get(emotion_code)
                if emotion:
                    metadata.append({
                        "filepath": file_path.replace("\\", "/"),
                        "emotion": emotion
                    })
            except Exception as e:
                print(f"[WARNING] Failed to parse filename: {file} - {e}")

# Save to CSV
df = pd.DataFrame(metadata)
df.to_csv("data/audio_metadata.csv", index=False)
print(
    f"[DONE] Metadata saved to data/audio_metadata.csv with {len(df)} entries.")
