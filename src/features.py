# src/features.py
import numpy as np
import librosa

def mfcc_features(y, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    feat = {}
    for i, v in enumerate(mfcc_mean):
        feat[f"mfcc_mean_{i}"] = float(v)
    for i, v in enumerate(mfcc_std):
        feat[f"mfcc_std_{i}"] = float(v)
    return feat
