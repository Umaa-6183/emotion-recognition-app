import sounddevice as sd
import librosa
import soundfile as sf
import numpy as np
from joblib import load

def detect_audio_emotion(model_path='models/audio_emotion_model.joblib'):
    duration = 3
    fs = 44100
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write('record.wav', audio, fs)

    y, sr = librosa.load('record.wav')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)

    model = load(model_path)
    prediction = model.predict(mfccs_scaled)[0]
    return prediction
