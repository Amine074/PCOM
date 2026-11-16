#!/usr/bin/env python3
"""PCOM Homework 5 - prediction function (XGBoost).

Usage:
    python predict.py input.csv output.csv
    python predict.py input.csv              (prints the output CSV to the screen)

Input : CSV with the records (package versions) of ONE package, in the record schema.
        Missing columns, empty cells and NULL are ok.
Output: CSV with ONE record: the five target_ghsa_*_count_90d columns
        (floats >= 0, for example 0.4286 = expected number of new advisories).

The XGBoost models are in the folder  model/  (made by train.py).
If XGBoost or the models cannot be used, a simple rule is used instead and a
warning is printed, so there is always a valid output file.
"""
import csv
import json
import os
import sys

from features import (FEATURE_NAMES, SEVERITIES, TARGET_COLUMNS, DEFAULT_META, feature_vector,
                      merge_rows, read_rows, rule_prediction)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")


def load_meta():
    """Read model/meta.json; fall back to defaults if it is missing or broken."""
    try:
        with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
            m = json.load(f)
        return {
            "weight_recent": min(max(float(m["weight_recent"]), 0.0), 1.0),
            "global_mean": {s: float(m["global_mean"].get(s, 0.0)) for s in SEVERITIES},
            "constant_zero": list(m.get("constant_zero", [])),
            "feature_names": m.get("feature_names"),
        }
    except Exception:
        return dict(DEFAULT_META, constant_zero=[], feature_names=None)


def xgboost_prediction(rec, meta):
    """Expected counts from the XGBoost models. Raises an error if anything is wrong."""
    if os.environ.get("PCOM_NO_XGBOOST") == "1":  # only used by the tests
        raise RuntimeError("XGBoost switched off by PCOM_NO_XGBOOST")
    import numpy as np
    import xgboost as xgb

    if meta["feature_names"] != FEATURE_NAMES:
        raise ValueError("models were trained with other features, run train.py again")
    x = xgb.DMatrix(np.array([feature_vector(rec)], dtype=float))
    result = {}
    for sev in SEVERITIES:
        if sev in meta["constant_zero"]:
            result[sev] = 0.0
            continue
        booster = xgb.Booster()
        booster.load_model(os.path.join(MODEL_DIR, "xgb_%s.json" % sev))
        value = float(booster.predict(x)[0])
        if not value < float("inf"):  # NaN or infinity
            raise ValueError("model for %s gave %r" % (sev, value))
        result[sev] = max(value, 0.0)
    return result


def write_output(values, output_path=None):
    """Write the one-record CSV to a file, or to the screen if no path is given."""
    out = open(output_path, "w", newline="", encoding="utf-8") if output_path else sys.stdout
    try:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(TARGET_COLUMNS)
        writer.writerow(["%.4f" % values[s] for s in SEVERITIES])  # always a float, e.g. 0.0000
    finally:
        if output_path:
            out.close()


def predict(input_path, output_path=None):
    """The prediction function: input CSV file -> output CSV file (one record)."""
    try:
        rows = read_rows(input_path)
    except Exception as exc:  # still write a valid output file
        print("WARNING: could not read %s (%s). Using empty data." % (input_path, exc),
              file=sys.stderr)
        rows = []
    rec = merge_rows(rows)
    meta = load_meta()
    try:
        values = xgboost_prediction(rec, meta)
    except Exception as exc:
        print("WARNING: XGBoost prediction not possible (%s: %s). Using the simple rule."
              % (type(exc).__name__, exc), file=sys.stderr)
        values = rule_prediction(rec, meta)
    values = {s: round(values[s], 4) for s in SEVERITIES}
    if output_path and os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_output(values, output_path)
    return values


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(2)
    predict(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)