#!/usr/bin/env python3
"""Train the XGBoost models.

Usage:
    python train.py dataset_initial.csv [more_data.csv ...]

- Needs:  pip install xgboost numpy
- One training example per package (data of its newest version).
- One XGBoost model per severity (low, moderate, high, critical, unknown).
  Objective: count:poisson. The model predicts the expected number of new
  advisories in 90 days (a float).
- Prints a cross-validation table: XGBoost against two simple baselines.
- Saves the models in the folder  model/  (next to this file).
"""
import datetime
import json
import math
import os
import sys

try:
    import numpy as np
    import xgboost as xgb
except ImportError:
    sys.exit("Please install the packages first:  pip install xgboost numpy")

from features import (DEFAULT_META, FEATURE_NAMES, SEVERITIES, TARGET_COLUMNS, feature_vector,
                      merge_rows, package_name, read_rows, recent_rate, to_number)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

# Small trees and strong regularization, because we have few training packages.
PARAMS = {"objective": "count:poisson", "max_depth": 2, "eta": 0.05, "min_child_weight": 0.5,
          "lambda": 5.0, "subsample": 1.0, "seed": 0, "nthread": 1, "verbosity": 0}
ROUNDS = 100


def load_packages(paths):
    """{package name: data of the package as one dict}"""
    packages = {}
    for path in paths:
        by_package = {}
        for row in read_rows(path):
            by_package.setdefault(package_name(row.get("purl")), []).append(row)
        for name, rows in by_package.items():
            if name and name not in packages:
                packages[name] = merge_rows(rows)
    return packages


def make_dataset(packages):
    X, Y, R, names, skipped = [], [], [], [], 0
    for name, rec in sorted(packages.items()):
        targets = [to_number(rec.get(c)) for c in TARGET_COLUMNS]
        if any(t is None for t in targets):
            skipped += 1
            continue
        X.append(feature_vector(rec))
        Y.append(targets)
        R.append([recent_rate(rec, s) for s in SEVERITIES])
        names.append(name)
    return (np.array(X, dtype=float), np.array(Y, dtype=float),
            np.array(R, dtype=float), names, skipped)


def fit_booster(X, y):
    params = dict(PARAMS)
    params["base_score"] = max(float(y.mean()), 0.001)  # start at the average count
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=ROUNDS)


def booster_predict(booster, X):
    return np.clip(booster.predict(xgb.DMatrix(X)), 0.0, None)


def poisson_log_score(Y, L):
    """Average (over packages) of the log probability of the true counts. Higher is better."""
    L = np.clip(L, 0.01, None)
    log_factorial = np.array([[math.lgamma(v + 1) for v in row] for row in Y])
    return float(np.mean(np.sum(Y * np.log(L) - L - log_factorial, axis=1)))


def cross_validate(X, Y, R, folds=5):
    """Cross-validation by package (every package is in the test part exactly once)."""
    n = len(X)
    order = np.random.RandomState(0).permutation(n)
    pred = {k: np.zeros_like(Y) for k in ("constant", "rule", "xgboost")}
    w = DEFAULT_META["weight_recent"]
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        mean = Y[train].mean(axis=0)
        for j in range(Y.shape[1]):
            pred["constant"][test, j] = mean[j]
            pred["rule"][test, j] = w * R[test, j] + (1 - w) * mean[j]
            y = Y[train, j]
            pred["xgboost"][test, j] = (0.0 if y.sum() == 0 else
                                        booster_predict(fit_booster(X[train], y), X[test]))
    print("\nCross-validation by package (%d packages, %d folds)" % (n, folds))
    print("%-34s %14s %18s" % ("method", "log score", "abs. error/package"))
    print("%-34s %14s %18s" % ("", "(higher=better)", "(lower=better)"))
    for key, label in (("constant", "constant (training average)"),
                       ("rule", "simple rule (last 12 months)"),
                       ("xgboost", "XGBoost")):
        print("%-34s %14.3f %18.3f" % (label, poisson_log_score(Y, pred[key]),
                                       float(np.mean(np.sum(np.abs(pred[key] - Y), axis=1)))))


def main(paths):
    packages = load_packages(paths)
    X, Y, R, names, skipped = make_dataset(packages)
    print("Packages used for training: %d (skipped without target values: %d)"
          % (len(names), skipped))
    if len(names) < 2:
        sys.exit("Not enough packages with target values to train.")
    print("Packages with at least one new advisory in 90 days: %d" % int((Y.sum(axis=1) > 0).sum()))

    if len(names) >= 10:
        cross_validate(X, Y, R, folds=min(5, len(names)))
    else:
        print("\n(Fewer than 10 packages: cross-validation skipped, it would not mean much.)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    constant_zero = []
    print()
    for j, sev in enumerate(SEVERITIES):
        path = os.path.join(MODEL_DIR, "xgb_%s.json" % sev)
        if os.path.exists(path):
            os.remove(path)
        if Y[:, j].sum() == 0:
            constant_zero.append(sev)
            print("%-9s no training package has such an advisory -> always predicts 0.0" % sev)
            continue
        booster = fit_booster(X, Y[:, j])
        booster.save_model(path)
        gain = booster.get_score(importance_type="gain")  # keys look like "f12"
        top = sorted(gain.items(), key=lambda kv: -kv[1])[:3]
        shown = ", ".join(FEATURE_NAMES[int(k[1:])] for k, _ in top) if top else "(no splits)"
        print("%-9s model saved. Most used features: %s" % (sev, shown))

    meta = {
        "feature_names": FEATURE_NAMES,
        "constant_zero": constant_zero,
        "global_mean": {s: round(float(Y[:, j].mean()), 4) for j, s in enumerate(SEVERITIES)},
        "weight_recent": DEFAULT_META["weight_recent"],
        "trained_on_packages": len(names),
        "trained_on_date": datetime.date.today().isoformat(),
        "xgboost_version": xgb.__version__,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved the models in", MODEL_DIR)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])