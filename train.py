"""Train one Poisson Gradient Boosting model for each 90-day GHSA target."""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from features import TARGETS, get_feature_columns, make_package_dataset

RANDOM_STATE = 42
MODEL_PARAMS = {
    "loss": "poisson",
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
}


def build_regressor() -> HistGradientBoostingRegressor:
    """Create a Poisson gradient boosting regressor for count targets."""
    return HistGradientBoostingRegressor(**MODEL_PARAMS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="training_data.csv")
    parser.add_argument("--model", default="pcom_models.joblib")
    parser.add_argument("--package-csv", default="package_training_data.csv")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    data, _ = make_package_dataset(args.input)
    data.to_csv(args.package_csv, index=False)
    features = get_feature_columns(data)
    if not features:
        raise ValueError("No usable numeric feature columns found.")

    X = data[features].apply(pd.to_numeric, errors="coerce")
    y = data[TARGETS].apply(pd.to_numeric, errors="coerce").fillna(0)

    if len(data) < 10:
        raise ValueError(f"Only {len(data)} packages available; need at least 10 for a holdout test.")

    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=args.test_size, random_state=RANDOM_STATE
    )
    models = {}
    report = {}

    for target in TARGETS:
        y_train = y.iloc[train_idx][target]
        y_test = y.iloc[test_idx][target]

        # 1. Holdout evaluation on the 80/20 split
        if y_train.nunique() <= 1:
            predictions = np.full(len(test_idx), float(y_train.iloc[0]))
        else:
            eval_model = build_regressor()
            eval_model.fit(X.iloc[train_idx], y_train)
            predictions = eval_model.predict(X.iloc[test_idx])

        predictions = np.maximum(0.0, predictions)
        report[target] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "test_actual_positive": int((y_test > 0).sum()),
            "test_rows": int(len(y_test)),
            "model_type": "constant" if y[target].nunique() <= 1 else "model",
        }

        # 2. Fit final model on 100% of packages for prediction rounds
        if y[target].nunique() <= 1:
            models[target] = {"type": "constant", "value": float(y[target].iloc[0])}
        else:
            final_model = build_regressor()
            final_model.fit(X, y[target])
            models[target] = {"type": "model", "estimator": final_model}

    bundle = {
        "models": models,
        "feature_columns": features,
        "target_columns": TARGETS,
        "parameters": MODEL_PARAMS,
        "as_of_date": "2026-07-01",
        "package_count": int(len(data)),
    }
    joblib.dump(bundle, args.model)

    print(f"Packages: {len(data)}")
    print(f"Features used: {len(features)}")
    print(f"Saved model bundle: {args.model}")
    print("Holdout results (MAE / RMSE):")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()