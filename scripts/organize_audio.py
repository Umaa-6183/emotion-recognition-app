import os
import shutil

# Folder where all unorganized .wav files are present
SOURCE_DIR = "data/audio_speech_actors_01-24"

# Target folder where you'll save organized files
DEST_DIR = "data/audio"

# Mapping of emotion code to folder name
EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",   # or change to 'calm' if needed
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

# Create subfolders for each emotion
for emotion in set(EMOTION_MAP.values()):
    os.makedirs(os.path.join(DEST_DIR, emotion), exist_ok=True)

# Go through each Actor folder and file inside them
for root, dirs, files in os.walk(SOURCE_DIR):
    for filename in files:
        if filename.endswith(".wav"):
            parts = filename.split("-")
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = EMOTION_MAP.get(emotion_code)
                if emotion:
                    source = os.path.join(root, filename)
                    dest = os.path.join(DEST_DIR, emotion, filename)
                    shutil.copy(source, dest)
                    print(f"Copied {filename} to {emotion}")
