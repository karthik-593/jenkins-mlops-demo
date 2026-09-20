"""Train a classifier and persist the model + training metadata."""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    os.makedirs(a.model_dir, exist_ok=True)
    train = pd.read_csv(f"{a.data_dir}/train.csv")
    X = train.drop(columns=["target"])
    y = train["target"]

    model = RandomForestClassifier(
        n_estimators=a.n_estimators, random_state=a.seed, n_jobs=-1
    )
    model.fit(X, y)

    joblib.dump(model, f"{a.model_dir}/model.pkl")
    with open(f"{a.model_dir}/train_meta.json", "w") as f:
        json.dump(
            {"n_estimators": a.n_estimators,
             "n_train_rows": len(train),
             "features": list(X.columns)},
            f, indent=2,
        )
    print(f"Trained on {len(train)} rows -> {a.model_dir}/model.pkl")


if __name__ == "__main__":
    main()
