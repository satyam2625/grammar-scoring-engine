import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# 1. Load the data (adjust paths if your CSVs are not inside "data/")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Select simple features
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Convert categorical "Sex" to numbers (safe mapping)
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"]  = test_df["Sex"].map({"male": 0, "female": 1})

# If any NaN appear after mapping, fill with 0 (or median)
train_df["Sex"].fillna(0, inplace=True)
test_df["Sex"].fillna(0, inplace=True)

X = train_df[features]
y = train_df["Survived"]
X_test = test_df[features]

# 3. Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Make predictions
preds = model.predict(X_test)

# 5. Save submission file
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": preds
})
submission.to_csv("outputs/submission.csv", index=False)

print("✅ Submission file created: outputs/submission.csv")
