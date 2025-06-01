import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import pickle
import os

DURATION = 4  # seconds
SAMPLE_RATE = 44100
AUDIO_FILENAME = "live_audio.wav"


def record_audio(filename=AUDIO_FILENAME, duration=DURATION, samplerate=SAMPLE_RATE):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate),
                       samplerate=samplerate, channels=1)
    sd.wait()
    write(filename, samplerate, recording)
    print("Recording finished")


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs


def predict_emotion_live_audio(model_path='models/emotion_audio.pkl'):
    record_audio()
    features = extract_features(AUDIO_FILENAME).reshape(1, -1)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(features)
    return prediction[0]
