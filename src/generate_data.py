"""Generate a synthetic binary-classification dataset and split into train/test CSVs.

`--class-sep` controls how separable the classes are (lower = harder = lower
accuracy). Use it in the demo to push the model below the quality-gate threshold.
"""
import argparse
import os

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--class-sep", type=float, default=1.0,
                   help="Lower = harder classes = lower accuracy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    X, y = make_classification(
        n_samples=a.n_samples,
        n_features=a.n_features,
        n_informative=max(2, a.n_features // 2),
        n_redundant=2,
        n_classes=2,
        class_sep=a.class_sep,
        random_state=a.seed,
    )
    cols = [f"f{i}" for i in range(a.n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    train, test = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=a.seed
    )
    train.to_csv(os.path.join(a.out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(a.out_dir, "test.csv"), index=False)

    print(f"Wrote {len(train)} train / {len(test)} test rows to {a.out_dir}/ "
          f"(class_sep={a.class_sep})")


if __name__ == "__main__":
    main()
