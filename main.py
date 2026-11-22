from pathlib import Path
import io

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
import uvicorn

from features import (
    TARGETS,
    add_date_features,
    newest_row_without_purl,
    normalize_columns,
    stable_version_from_purl,
)


app = FastAPI(title="ML Prediction API")

MODEL_PATH = Path(__file__).resolve().parent / "pcom_models.joblib"
MODEL_BUNDLE = joblib.load(MODEL_PATH)


class EvaluationBody(BaseModel):
    testdata_download_endpoint: str
    prediction_upload_endpoint: str


def prepare_single_package_row(df: pd.DataFrame) -> pd.DataFrame:
    """Reduziert die mehrzeilige Versions-Historie eines Pakets auf eine Feature-Zeile."""
    df = normalize_columns(df)
    if "purl" not in df.columns:
        return newest_row_without_purl(df)

    parsed = df["purl"].apply(stable_version_from_purl)
    df["_package"] = parsed.apply(lambda x: x[0])
    df["_version"] = parsed.apply(lambda x: x[1])
    df = df[df["_version"].notna()].copy()
    if df.empty:
        raise ValueError("No stable npm package versions found in input CSV.")
    if df["_package"].nunique() != 1:
        raise ValueError(f"Expected exactly 1 package; found {df['_package'].nunique()}.")

    latest_idx = max(df.index, key=lambda idx: df.loc[idx, "_version"])
    row = df.loc[latest_idx].drop(labels=["_package", "_version"]).to_dict()
    row["package_name"] = df.loc[latest_idx, "_package"]
    return add_date_features(pd.DataFrame([row]))


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    package_row = prepare_single_package_row(df)
    feature_columns = MODEL_BUNDLE["feature_columns"]
    X = package_row.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")

    output = {}
    for target in TARGETS:
        saved = MODEL_BUNDLE["models"][target]
        if saved["type"] == "constant":
            value = saved["value"]
        else:
            value = float(saved["estimator"].predict(X)[0])

        val = round(max(0.0, float(value)), 4)
        if pd.isna(val):
            val = 0.0
        output[target] = val

    return pd.DataFrame([output], columns=TARGETS)


def start_prediction(download_url: str, upload_url: str):
    try:
        result = requests.get(download_url, timeout=30)
        result.raise_for_status()

        # Read input time-series CSV (comma-separated, US float format)
        input_data = pd.read_csv(io.StringIO(result.text), sep=",", decimal=".")

        prediction_data = run_model(input_data)

        assert len(prediction_data) == 1, "Prediction DataFrame must contain exactly 1 row"
        output_buffer = io.StringIO()
        prediction_data.to_csv(output_buffer, sep=",", decimal=".", header=True, index=False)
        output_buffer.seek(0)

        file = {"predictions": ("prediction.csv", output_buffer.getvalue(), "text/csv")}

        upload_res = requests.post(upload_url, files=file, timeout=30)
        upload_res.raise_for_status()
        print("Evaluation Result:", upload_res.json())

    except Exception as e:
        print("Error:", e)


@app.post("/predict", status_code=200)
async def prediction(payload: EvaluationBody, background_tasks: BackgroundTasks):
    # Start prediction in background
    background_tasks.add_task(
        start_prediction,
        payload.testdata_download_endpoint,
        payload.prediction_upload_endpoint,
    )
    return {"status": "prediction started"}


@app.get("/health", status_code=200)
async def health():
    return {"health": "up"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
