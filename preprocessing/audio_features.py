import librosa
import numpy as np

def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Shape: (time_steps, n_mfcc)
