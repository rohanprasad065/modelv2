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
           that same 13-column data and append them (43 cols total).
           These are built per-hour because the rolling/lag/z-score
           formulas need the trailing history at each hour -- but only
           the LAST row (t=167) ends up with features drawn from the
           *complete* 7-day window; every earlier row's features are
           partial (e.g. a 24h rolling z-score at hour 5 only sees 5
           hours of history). That last row is therefore the one row
           that actually represents "this animal's state given the
           whole week" -- the only one worth handing to the estrus
           models.
        6. Feed ONLY that last row's 43 columns into the two estrus
           XGBoost models (estrus_predict.py) to get a SINGLE:
             - predicted_days_to_estrus (estrus_xgb_model.json, regressor)
             - estrus_probability_24h (estrus_24hour_xgb_model_phase1.json,
               classifier -- P(estrus within next 24h))
           predicted_days_to_estrus is also converted into a calendar
           date (predicted_estrus_date), anchored to the timestamp of
           that last row.

           The response has two parts:
             - "week_data": the 168-row hourly history (the 8 raw
               sensor/activity columns + predicted_body_temp_C, plus
               plain date/time) -- this is DESCRIPTIVE, not a
               prediction, so it does not repeat the estrus outputs on
               every row.
             - "prediction": the single days-to-estrus / date /
               probability result for the week.
           The 30 model2 engineered features and the hour_sin/hour_cos/
           day_sin/day_cos cyclical encoding are computed and used
           internally as required input to Model 1, Model 2, and both
           estrus models, but are not part of either response section.
           All intermediate tables are discarded once the response is
           built -- no database, no file on disk, nothing kept after
           the response is sent.

    - Response shape:
        {
          "week_data": {
            "rows": 168,
            "columns": [
              "date", "time", "region_temperature_C", "THI",
              "heart_rate_bpm", "Standing", "Walking", "Eating",
              "Ruminating", "neck_temperature_C", "predicted_body_temp_C"
            ],
            "data": [
              {
                "date": "2026-08-06",
                "time": "00:00:00",
                "region_temperature_C": 16.1,
                "THI": 60.01805,
                "heart_rate_bpm": 83.5,
                "Standing": false,
                "Walking": false,
                "Eating": false,
                "Ruminating": true,
                "neck_temperature_C": 33.345,
                "predicted_body_temp_C": 39.20598602294922
              },
              ... (168 rows total)
            ]
          },
          "prediction": {
            "as_of_date": "2026-08-12",
            "as_of_time": "23:00:00",
            "predicted_days_to_estrus": 18.26,
            "predicted_estrus_date": "2026-08-31",
            "estrus_probability_24h": 0.073
          }
        }

Run locally:
    uvicorn main:app --reload --port 8000

Interactive docs: http://127.0.0.1:8000/docs
"""

from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import timedelta

from processing import clean_pipeline
from weather import add_humidity
from features import build_features
from model1_features import build_model1_features
from model1_predict import predict_body_temp
from model2_features import build_model2_features
from estrus_predict import predict_days_to_estrus, predict_estrus_probability_24h

app = FastAPI(title="Wearable Weekly Data Pipeline")

# CORS: the frontend runs on a different origin (different port, or a
# different host entirely) than this API, so the browser sends a
# preflight OPTIONS request before every real POST. Without this
# middleware FastAPI has no OPTIONS handler registered for /process-week
# at all, so that preflight gets a 405 and the browser blocks the actual
# POST from ever firing -- that's the "OPTIONS ... 405" you're seeing.
#
# allow_origins=["*"] is fine for local development / a demo. Before
# shipping this to production, replace "*" with the exact origin(s) your
# frontend is served from (e.g. "https://your-app.example.com") --
# wildcard origins can't be combined with allow_credentials=True per the
# CORS spec, and browsers will reject that combination outright.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # build_features() consumes 'datetime' only to derive hour_sin/cos and
    # day_sin/cos, then drops the column entirely -- capture it here so we
    # can put plain date/time back in the response later (row order is
    # preserved end-to-end, so this aligns positionally with base_12col).
    timestamps = pd.to_datetime(hourly["datetime"]).reset_index(drop=True)

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

    # Step 6: feed the 43-column table into both estrus models -- but
    # only for the LAST row of the window. Every row's rolling/lag/
    # z-score features are computed over "however much history is
    # available up to that hour", so only the final row (t=167) has
    # features built from the *complete* 7-day window; that's the one
    # row that actually represents "given everything we know about this
    # animal this week, what's the estrus outlook right now" -- the
    # thing we actually want one answer for, not 168 partial-window
    # guesses re-asked every hour.
    last_row = result_df.iloc[[-1]]

    try:
        days_to_estrus = predict_days_to_estrus(last_row)
        estrus_prob_24h = predict_estrus_probability_24h(last_row)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=500, detail=f"Estrus prediction failed: {e}")

    predicted_days_to_estrus = float(days_to_estrus.iloc[0])
    estrus_probability_24h = float(estrus_prob_24h.iloc[0])

    # Anchor the "days to estrus" figure to a real calendar date, taken
    # from the most recent hour in the window (the point the prediction
    # is made "as of").
    last_timestamp = timestamps.iloc[-1]
    predicted_estrus_date = (
        last_timestamp + timedelta(days=predicted_days_to_estrus)
    ).strftime("%Y-%m-%d")

    # week_data: the 7-day base data + Model 1's per-hour body-temp
    # prediction -- one row per hour, 168 rows total. This is descriptive
    # history, not a prediction, so it does NOT carry the estrus outputs.
    # Drop the 30 model2 engineered features and the cyclical encodings --
    # those were only needed internally to build inputs for Model 1 /
    # Model 2 / the estrus models above.
    result_df = result_df.drop(columns=list(model2_feats.columns))
    result_df["date"] = timestamps.dt.strftime("%Y-%m-%d").values
    result_df["time"] = timestamps.dt.strftime("%H:%M:%S").values
    result_df = result_df.drop(columns=["hour_sin", "hour_cos", "day_sin", "day_cos"])

    WEEK_DATA_COLUMNS = [
        "date", "time",
        "region_temperature_C", "THI", "heart_rate_bpm",
        "Standing", "Walking", "Eating", "Ruminating",
        "neck_temperature_C", "predicted_body_temp_C",
        "humidity", "state",
    ]
    result_df = result_df[WEEK_DATA_COLUMNS]

    # predicted_body_temp_C never contains NaN, but keep this NaN->None
    # safety pass anyway in case a future raw input column ever does
    # (JSON has no NaN literal, only null).
    result_df = result_df.astype(object).where(pd.notnull(result_df), None)

    # After this response is sent, base_12col / model1_feats / predictions /
    # model2_feats are all garbage-collected -- nothing about this request
    # is retained.
    records = result_df.to_dict(orient="records")
    return JSONResponse({
        "week_data": {
            "rows": len(records),
            "columns": list(result_df.columns),
            "data": records,
        },
        "prediction": {
            "as_of_date": last_timestamp.strftime("%Y-%m-%d"),
            "as_of_time": last_timestamp.strftime("%H:%M:%S"),
            "predicted_days_to_estrus": predicted_days_to_estrus,
            "predicted_estrus_date": predicted_estrus_date,
            "estrus_probability_24h": estrus_probability_24h,
        },
    })
