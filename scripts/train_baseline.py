# scripts/train_baseline.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def main():
    feat_df = pd.read_pickle("outputs/features.pkl")
    meta = pd.read_csv("data/metadata/train.csv")  # must align sample_id
    # assume labels in meta: columns sample_id, score
    df = feat_df.merge(meta, on="sample_id", how="left")
    y = df['score'].values
    X = df.drop(columns=['sample_id','score']).fillna(0)
    X_cols = X.columns

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr, val in kf.split(X):
        m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        m.fit(X.iloc[tr], y[tr])
        oof[val] = m.predict(X.iloc[val])
    rmse = mean_squared_error(y, oof, squared=False)
    print("CV RMSE:", rmse)

    # Train final on full data
    final = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    final.fit(X, y)
    joblib.dump({"model": final, "columns": list(X_cols)}, "models/rf_baseline.joblib")
    print("Saved models/rf_baseline.joblib")

if __name__ == "__main__":
    main()
