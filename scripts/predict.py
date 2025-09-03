# scripts/predict.py
import pandas as pd
import joblib

def main():
    feat_df = pd.read_pickle("outputs/test_features.pkl")  # produced by extract_features on test set
    mdl = joblib.load("models/rf_baseline.joblib")
    model = mdl['model']
    cols = mdl['columns']
    X = feat_df[cols].fillna(0)
    preds = model.predict(X)
    out = pd.DataFrame({"sample_id": feat_df["sample_id"], "grammar_score": preds})
    out.to_csv("outputs/submission.csv", index=False)
    print("Saved outputs/submission.csv")

if __name__ == "__main__":
    main()
