# audio_utils.py

import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np


def record_audio(filename, duration=3, samplerate=44100):
    print(f"[INFO] Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"[INFO] Audio saved to {filename}")


def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Pad or truncate MFCCs to ensure same shape
        if mfccs.shape[1] < 174:
            pad_width = 174 - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=(
                (0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :174]

        return mfccs.flatten()[:128]  # Return only first 128 features
    except Exception as e:
        print(f"[ERROR] Extracting features failed: {e}")
        return np.zeros(128)  # Return fallback
