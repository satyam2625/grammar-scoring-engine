# src/audio_utils.py
import soundfile as sf
import librosa
import numpy as np

def load_audio(path, sr=16000):
    y, orig_sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y, sr

def pause_features(y, sr=16000, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)
    voiced_samples = sum((end - start) for start, end in intervals)
    total = len(y)
    pause_total = total - voiced_samples
    return {
        "duration_sec": total / sr,
        "voiced_sec": voiced_samples / sr,
        "pause_sec": pause_total / sr,
        "pause_ratio": pause_total / total if total > 0 else 0.0
    }
