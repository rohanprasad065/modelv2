"""
main.py
-------
FastAPI service exposing one endpoint:

  POST /process-week
    - Accepts raw readings as a JSON body (not a file/form upload):
        {
          "fetch_weather": true,
          "readings": [
            {"date": "2026-08-06", "time": "00:00:00", "temp_c": 36.96,
             "heart_rate": 61, "ambient_temp": 24.3, "activity": "Walking",
             "location": "(28.474832, 77.504403)"},
            ...
          ]
        }
      (irregular ~30-min intervals, any field except date/time may be null)
    - Returns clean 7-day x 1-hour (168-row) feature table as JSON,
      with humidity fetched per-location and heat index / one-hot /
      cyclical time features attached.

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

app = FastAPI(title="Wearable Weekly Data Pipeline")


class Reading(BaseModel):
    date: str                              # "2026-08-06"
    time: str                              # "00:00:00"
    temp_c: Optional[float] = None
    heart_rate: Optional[float] = None
    ambient_temp: Optional[float] = None
    activity: Optional[str] = None
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

    try:
        hourly = clean_pipeline(df_raw, days=7)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if payload.fetch_weather:
        hourly = add_humidity(hourly)

    featurized = build_features(hourly)

    featurized["datetime"] = featurized["datetime"].astype(str)
    records = featurized.to_dict(orient="records")

    return JSONResponse({
        "rows": len(records),
        "columns": list(featurized.columns),
        "data": records,
    })