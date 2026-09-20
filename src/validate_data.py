"""Data validation gate. Exits non-zero if any check fails, which stops the
Jenkins build before wasting compute on training bad data."""
import argparse
import sys

import pandas as pd

TARGET = "target"


def check(df, name, min_rows):
    errors = []
    if len(df) < min_rows:
        errors.append(f"{name}: only {len(df)} rows (< {min_rows})")
    if TARGET not in df.columns:
        errors.append(f"{name}: missing '{TARGET}' column")
        return errors
    if df.isnull().any().any():
        errors.append(f"{name}: contains null values")
    if df[TARGET].nunique() < 2:
        errors.append(f"{name}: target has < 2 classes")
    else:
        frac = df[TARGET].value_counts(normalize=True).min()
        if frac < 0.1:
            errors.append(f"{name}: severe class imbalance (min class {frac:.1%})")
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--min-rows", type=int, default=100)
    a = p.parse_args()

    errors = []
    for split in ["train", "test"]:
        df = pd.read_csv(f"{a.data_dir}/{split}.csv")
        errors += check(df, split, a.min_rows)

    if errors:
        print("DATA VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("Data validation passed.")


if __name__ == "__main__":
    main()
