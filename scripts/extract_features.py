# scripts/extract_features.py
import pandas as pd
import os
from src.audio_utils import load_audio, pause_features
from src.features import mfcc_features
# optional: from src.asr_wrapper import transcribe
import joblib
from tqdm import tqdm

def extract_for_row(row, audio_root):
    path = os.path.join(audio_root, row['file_path'])
    y, sr = load_audio(path)
    pf = pause_features(y, sr)
    mf = mfcc_features(y, sr)
    # optionally: text = transcribe(y, sr); compute grammar errors...
    feat = {}
    feat.update(pf)
    feat.update(mf)
    feat['sample_id'] = row.get('sample_id', row.get('id', None))
    return feat

def main():
    metadata_path = "data/metadata/train.csv"   # change as needed
    audio_root = "data/raw"
    df = pd.read_csv(metadata_path)
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            feat = extract_for_row(row, audio_root)
            features.append(feat)
        except Exception as e:
            print("error", row, e)
    feat_df = pd.DataFrame(features)
    feat_df.to_pickle("outputs/features.pkl")
    print("Saved features to outputs/features.pkl")

if __name__ == "__main__":
    main()
