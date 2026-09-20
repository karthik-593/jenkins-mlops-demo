"""Evaluate the model and enforce the quality gate.

Writes metrics.json + a confusion-matrix PNG, then exits non-zero if the chosen
metric is below --threshold. That non-zero exit is what fails the Jenkins build.
"""
import argparse
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # headless: no display in CI
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--metric", default="accuracy",
                   choices=["accuracy", "f1", "roc_auc"])
    a = p.parse_args()

    test = pd.read_csv(f"{a.data_dir}/test.csv")
    X = test.drop(columns=["target"])
    y = test["target"]

    model = joblib.load(f"{a.model_dir}/model.pkl")
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }
    with open(f"{a.model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    ConfusionMatrixDisplay(confusion_matrix(y, pred)).plot()
    plt.savefig(f"{a.model_dir}/confusion_matrix.png", bbox_inches="tight")

    primary = metrics[a.metric]
    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"Quality gate: {a.metric}={primary:.4f} vs threshold={a.threshold}")

    if primary < a.threshold:
        print(f"QUALITY GATE FAILED: {a.metric} {primary:.4f} < {a.threshold}")
        sys.exit(1)
    print("QUALITY GATE PASSED.")


if __name__ == "__main__":
    main()
