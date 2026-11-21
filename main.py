from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import requests
import pandas as pd
import io
import uvicorn


app = FastAPI(title="ML Prediction API")


class EvaluationBody(BaseModel):
    testdata_download_endpoint: str
    prediction_upload_endpoint: str


def run_model(df: pd.DataFrame) -> pd.DataFrame:
    # ==== EXAMPLE: ADD YOUR PREDICTIONS OF TRAINED MODEL HERE ====
    # df contains the input time-series metrics for a package (multiple rows).
    # Return a single-row DataFrame containing the target columns required by the competition:
    prediction_df = pd.DataFrame(
        [
            {
                "cve_high": 0.0,
            }
        ]
    )
    # =============================================================

    return prediction_df


def start_prediction(download_url: str, upload_url: str):
    try:
        result = requests.get(download_url, timeout=30)
        result.raise_for_status()

        # Read input time-series CSV (comma-separated, US float format)
        input_data = pd.read_csv(io.StringIO(result.text), sep=",", decimal=".")

        # ADD YOUR PREDICTION MODEL HERE TO PREDICT A SINGLE ROW WITH THE TARGET HEADERS
        # ======== Example =========
        prediction_data = run_model(input_data)
        # ======== END Example =====

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
