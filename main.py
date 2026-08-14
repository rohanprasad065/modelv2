"""
main.py
-------
FastAPI service exposing ONE endpoint:

  POST /process-week
    - Accepts raw readings as a JSON body:
        {
          "fetch_weather": true,
          "readings": [
            {"date": "2026-08-06", "time": "00:00:00", "neck_temp": 36.96,
             "heart_rate": 61, "region_temp": 24.3, "activity": "Walking",
             "location": "(28.474832, 77.504403)"},
            ...
          ]
        }
      (irregular ~30-min intervals, any field except date/time may be null.
       activity must be one of: Standing, Walking, Eating, Ruminating)

    - Internal flow (all in memory for the duration of this request only --
      nothing is written to disk, nothing persists after the response is
      sent):
        1. Clean + resample raw readings into the 12-column hourly result
           (region_temperature_C, THI, heart_rate_bpm, Standing, Walking,
           Eating, Ruminating, neck_temperature_C, hour_sin, hour_cos,
           day_sin, day_cos). Held in a local variable ("saved" just long
           enough to be reused in step 4).
        2. Build Model 1's 36-feature set from that same 12-column data.
        3. Feed the 36 features into the trained Model 1 XGBoost model ->
           predicted_body_temp_C, one per hour.
        4. Append predicted_body_temp_C onto the 12-column data (13 cols
           total).
        5. Build 30 estrus-detection features (model2_features.py) from
           that same 13-column data and append them (43 cols total) and
           return it. The 36-feature Model 1 table, the 30-feature
           Model 2 table, and everything else is discarded once the
           response is built -- no database, no file on disk, nothing
           kept after the response is sent.

    - Response shape (168 rows, one per hour, 43 columns each):
        {
          "rows": 168,
          "columns": [...],
          "data": [
            {
              "region_temperature_C": 16.1,
              "THI": 60.01805,
              "heart_rate_bpm": 83.5,
              "Standing": false,
              "Walking": false,
              "Eating": false,
              "Ruminating": true,
              "neck_temperature_C": 33.345,
              "hour_sin": 0.0,
              "hour_cos": 1.0,
              "day_sin": -0.5737722679043243,
              "day_cos": -0.8190148866680802,
              "predicted_body_temp_C": 39.20598602294922,
              "body_temp_zscore_24h": 1.44,
              "body_temp_zscore_6h": 1.44,
              "body_temp_delta_6h": null,
              "neck_temp_zscore_24h": -1.59,
              "thermal_gradient_body_neck": 18.66,
              "thi_adjusted_body_temp_residual": 0.81,
              "cardiothermal_index": 807.86,
              "...": "... (30 model2 features total, see model2_features.FEATURE_COLUMNS)"
            },
            ...
          ]
        }

Run locally:
    uvicorn main:app --reload --port 8000

Interactive docs: http://127.0.0.1:8000/docs
"""

from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from processing import clean_pipeline
from weather import add_humidity
from features import build_features
from model1_features import build_model1_features
from model1_predict import predict_body_temp
from model2_features import build_model2_features

app = FastAPI(title="Wearable Weekly Data Pipeline")


class Reading(BaseModel):
    date: str                              # "2026-08-06"
    time: str                              # "00:00:00"
    neck_temp: Optional[float] = None      # neck-collar sensor temp, °C
    heart_rate: Optional[float] = None
    region_temp: Optional[float] = None    # regional/ambient temp, °C
    activity: Optional[str] = None         # Standing | Walking | Eating | Ruminating
    location: Optional[str] = None         # "(lat, lon)" e.g. "(28.4748, 77.5040)"


class WeeklyDataRequest(BaseModel):
    readings: List[Reading]
    fetch_weather: bool = True


@app.get("/")
def root():
    return {"status": "ok", "endpoint": "POST /process-week (JSON body)"}


@app.post("/process-week")
async def process_week(payload: WeeklyDataRequest):
    if not payload.readings:
        raise HTTPException(status_code=400, detail="`readings` cannot be empty.")

    df_raw = pd.DataFrame([r.model_dump() for r in payload.readings])

    # Step 1: clean + resample -> 12-column hourly result. Held only in
    # this local variable for the rest of the request; nothing is written
    # to disk and nothing survives past this function returning.
    try:
        hourly = clean_pipeline(df_raw, days=7)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if payload.fetch_weather:
        hourly = add_humidity(hourly)

    try:
        base_12col = build_features(hourly)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 2: build the 36-feature set from the 12-column data (separate,
    # throwaway intermediate -- never returned, never persisted)
    try:
        model1_feats = build_model1_features(base_12col)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 3: predict
    try:
        predictions = predict_body_temp(model1_feats)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Step 4: append prediction onto the 12-column data (13 columns).
    result_df = base_12col.copy()
    result_df["predicted_body_temp_C"] = predictions.values

    # Step 5: build the 30 estrus-detection features from that 13-column
    # table and append them too. Same in-memory, throwaway pattern as
    # everything above -- nothing here is persisted.
    try:
        model2_feats = build_model2_features(result_df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result_df = pd.concat([result_df, model2_feats], axis=1)

    # Several model2 features are structurally undefined for the first
    # few hours of the window (e.g. body_temp_delta_6h needs t>=6,
    # lag_24h_walking needs t>=24). Those show up as NaN, which Python's
    # json module serializes as a bare `NaN` token -- not valid JSON.
    # Convert to None so the response is strict-JSON-parseable (null).
    result_df = result_df.astype(object).where(pd.notnull(result_df), None)

    # After this response is sent, base_12col / model1_feats / predictions /
    # model2_feats are all garbage-collected -- nothing about this request
    # is retained.
    records = result_df.to_dict(orient="records")
    return JSONResponse({
        "rows": len(records),
        "columns": list(result_df.columns),
        "data": records,
    })
