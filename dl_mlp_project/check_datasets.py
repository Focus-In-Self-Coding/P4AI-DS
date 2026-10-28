"""Check the two Kaggle CSVs before starting the longer MLP experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def inspect_csv(label: str, path: Path) -> bool:
    print(f"\n{'=' * 72}\n{label}\nFile: {path}\n{'=' * 72}")
    if not path.exists():
        print("ERROR: file not found.")
        return False
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR: could not read CSV: {exc}")
        return False

    df.columns = [str(c).strip().lower() for c in df.columns]
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("Column names:", ", ".join(df.columns))

    if "price" not in df.columns:
        print("ERROR: expected a target column named 'price' (case-insensitive).")
        return False

    price = pd.to_numeric(df["price"], errors="coerce")
    missing_or_invalid = int(price.isna().sum())
    nonpositive = int((price.dropna() <= 0).sum())
    print(f"Invalid/missing target rows (the training script drops these): {missing_or_invalid}")
    print(f"Non-positive target values: {nonpositive}")
    if nonpositive:
        print("ERROR: log1p(price) is enabled, so all usable prices must be positive.")
        return False
    if len(df) < 25:
        print("WARNING: fewer than 25 rows; 5-fold CV and validation splits may be unstable.")

    usable = df.loc[price.notna()].copy()
    print(f"Usable target rows: {len(usable):,}")
    print(f"Median price: {price.median():,.3f}")
    print(f"Missing cells (all columns): {int(df.isna().sum().sum()):,}")
    # This is informational only; the experiment's preprocessing is fitted within train folds.
    cat_cols = [c for c in usable.columns if c != "price" and (
        pd.api.types.is_object_dtype(usable[c])
        or pd.api.types.is_string_dtype(usable[c])
        or pd.api.types.is_bool_dtype(usable[c])
    )]
    print(f"Columns currently inferred as categorical: {len(cat_cols)}")
    print("First 3 rows:")
    print(df.head(3).to_string(index=False))
    print("STATUS: CSV loaded and target check passed.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MLP experiment input CSV files.")
    parser.add_argument("--housing_csv", default="data/Housing.csv")
    parser.add_argument("--king_county_csv", default="data/kc_house_data.csv")
    args = parser.parse_args()

    a = inspect_csv("Housing Prices dataset", Path(args.housing_csv))
    b = inspect_csv("King County dataset", Path(args.king_county_csv))
    if a and b:
        print("\nBoth CSV files passed the basic checks. You can run run_mlp.bat.")
        return 0
    print("\nOne or more checks failed. See data/README.md and README.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
