"""Predict the five 90-day GHSA counts for one npm package."""
from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd

from features import TARGETS, make_package_dataset


def main():
    parser = argparse.ArgumentParser(description="Predict 90-day GHSA counts for one npm package.")
    parser.add_argument("input", help="CSV containing version records for exactly one npm package")
    parser.add_argument("output_pos", nargs="?", default=None,
                        help="optional output CSV (same as --output)")
    parser.add_argument("--model", default="pcom_models.joblib")
    parser.add_argument("--output", default="prediction.csv")
    args = parser.parse_args()
    out_path = args.output_pos or args.output

    bundle = joblib.load(args.model)
    # the input of a prediction does not need target columns
    package_data, _ = make_package_dataset(args.input, require_targets=False)
    if len(package_data) != 1:
        raise ValueError(
            f"Prediction input must contain exactly one package; found {len(package_data)}. "
            "Create a CSV containing versions of only one npm package."
        )

    feature_columns = bundle["feature_columns"]
    X = package_data.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    output = {}
    for target in bundle["target_columns"]:
        saved = bundle["models"][target]
        if saved["type"] == "constant":
            value = saved["value"]
        else:
            value = float(saved["estimator"].predict(X)[0])
        output[target] = round(max(0.0, float(value)), 4)   # float, not rounded to an integer

    result = pd.DataFrame([output], columns=TARGETS)
    result.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Prediction saved to {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
