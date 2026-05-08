"""
api/main.py
===========
FastAPI REST API for the State Sales Forecasting System.

Endpoints
---------
GET  /                            Health check
GET  /states                      List available states
GET  /forecast/{state}            Get forecast for a state
GET  /forecast/{state}/compare    Compare all model forecasts
POST /forecast/batch              Batch forecasts for multiple states
GET  /models/leaderboard          Global model performance leaderboard
GET  /forecast/{state}/history    Historical sales + forecast (chart data)
"""

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path so we can import modules
SRC_DIR = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import (
    load_raw_data, clean_data, build_monthly_panel, get_state_series
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="State Sales Forecasting API",
    description=(
        "Production-ready time-series forecasting service. "
        "Trains SARIMA, Prophet, XGBoost, and LSTM models per US state, "
        "auto-selects the best, and serves 8-week ahead predictions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state: load forecasts at startup
# ---------------------------------------------------------------------------

FORECASTS_PATH = Path(__file__).parents[1] / "outputs" / "forecasts" / "all_forecasts.json"
_forecasts: Dict = {}
_panel: Optional[pd.DataFrame] = None


def _load_forecasts():
    global _forecasts
    if FORECASTS_PATH.exists():
        with open(FORECASTS_PATH) as f:
            _forecasts = json.load(f)
        logger.info(f"Loaded forecasts for {len(_forecasts)} states.")
    else:
        logger.warning(f"Forecasts file not found at {FORECASTS_PATH}. "
                       "Run the training pipeline first.")
        _forecasts = {}


def _load_panel():
    global _panel
    try:
        raw   = load_raw_data()
        clean = clean_data(raw)
        _panel = build_monthly_panel(clean)
        logger.info(f"Historical panel loaded: {len(_panel):,} rows.")
    except Exception as e:
        logger.error(f"Could not load historical panel: {e}")
        _panel = None


@app.on_event("startup")
async def startup_event():
    _load_forecasts()
    _load_panel()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    date:  str
    value: float

class ForecastResponse(BaseModel):
    state:          str
    best_model:     str
    forecast_steps: int
    unit:           str = "USD (monthly sales)"
    forecast:       List[ForecastPoint]
    generated_at:   str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ModelMetrics(BaseModel):
    model: str
    MAE:   float
    RMSE:  float
    MAPE:  float
    sMAPE: float

class CompareResponse(BaseModel):
    state:              str
    best_model:         str
    metrics:            List[ModelMetrics]
    all_forecasts:      Dict[str, List[ForecastPoint]]

class BatchRequest(BaseModel):
    states: List[str] = Field(..., description="List of state names to forecast")

class HistoryForecastResponse(BaseModel):
    state:   str
    history: List[ForecastPoint]
    forecast: List[ForecastPoint]
    best_model: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_state_data(state: str) -> Dict:
    """Retrieve forecast data for a state; raise 404 if not found."""
    # Try exact match first, then case-insensitive
    if state in _forecasts:
        return _forecasts[state]
    for k, v in _forecasts.items():
        if k.lower() == state.lower():
            return v
    raise HTTPException(
        status_code=404,
        detail=f"State '{state}' not found. "
               f"Use GET /states to see available states."
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {
        "status":        "ok",
        "service":       "State Sales Forecasting API",
        "version":       "1.0.0",
        "states_loaded": len(_forecasts),
        "docs":          "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/states", response_model=List[str], tags=["Data"])
def list_states():
    """Return all states for which forecasts are available."""
    if not _forecasts:
        raise HTTPException(status_code=503, detail="Forecasts not yet loaded.")
    return sorted(_forecasts.keys())


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(state: str):
    """
    Return the best-model 8-week (≈2-month) sales forecast for a given state.
    """
    data = _get_state_data(state)
    forecast_points = [
        ForecastPoint(date=d, value=round(v, 2))
        for d, v in zip(data["forecast_dates"], data["forecast_values"])
    ]
    return ForecastResponse(
        state=state,
        best_model=data["best_model"],
        forecast_steps=len(forecast_points),
        forecast=forecast_points,
    )


@app.get("/forecast/{state}/compare", response_model=CompareResponse, tags=["Forecast"])
def compare_models(state: str):
    """
    Return validation metrics and forecasts for ALL models for a state,
    plus the name of the automatically selected best model.
    """
    data = _get_state_data(state)
    dates = data["forecast_dates"]

    all_forecasts: Dict[str, List[ForecastPoint]] = {}
    for model_name, vals in data.get("all_model_forecasts", {}).items():
        all_forecasts[model_name] = [
            ForecastPoint(date=d, value=round(v, 2))
            for d, v in zip(dates, vals)
        ]

    metrics = [ModelMetrics(**m) for m in data.get("metrics", [])]

    return CompareResponse(
        state=state,
        best_model=data["best_model"],
        metrics=metrics,
        all_forecasts=all_forecasts,
    )


@app.post("/forecast/batch", tags=["Forecast"])
def batch_forecast(request: BatchRequest):
    """
    Return forecasts for multiple states in a single request.

    Body example::

        { "states": ["California", "Texas", "Florida"] }
    """
    results = {}
    errors  = {}
    for state in request.states:
        try:
            data = _get_state_data(state)
            results[state] = {
                "best_model":     data["best_model"],
                "forecast_dates": data["forecast_dates"],
                "forecast":       data["forecast_values"],
            }
        except HTTPException as e:
            errors[state] = e.detail

    return {"forecasts": results, "errors": errors}


@app.get("/models/leaderboard", tags=["Models"])
def model_leaderboard():
    """
    Aggregate model win counts across all states.
    Returns how often each model was chosen as the best.
    """
    if not _forecasts:
        raise HTTPException(status_code=503, detail="No forecasts available.")

    win_counts: Dict[str, int] = {}
    avg_mape:   Dict[str, List[float]] = {}

    for state_data in _forecasts.values():
        bm = state_data.get("best_model", "Unknown")
        win_counts[bm] = win_counts.get(bm, 0) + 1

        for m in state_data.get("metrics", []):
            name = m.get("model", "?")
            mape = m.get("MAPE", None)
            if mape is not None:
                avg_mape.setdefault(name, []).append(mape)

    leaderboard = []
    for model_name, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        leaderboard.append({
            "model":      model_name,
            "wins":       wins,
            "win_pct":    round(wins / len(_forecasts) * 100, 1),
            "avg_MAPE":   round(float(np.mean(avg_mape.get(model_name, [0]))), 2),
        })

    return {"total_states": len(_forecasts), "leaderboard": leaderboard}


@app.get("/forecast/{state}/history", response_model=HistoryForecastResponse,
         tags=["Forecast"])
def get_history_and_forecast(
    state: str,
    last_n_months: int = Query(default=24, ge=3, le=120,
                               description="How many historical months to include")
):
    """
    Return historical sales + future forecast for charting.
    """
    data = _get_state_data(state)

    history_points: List[ForecastPoint] = []
    if _panel is not None:
        state_hist = _panel[_panel["state"].str.lower() == state.lower()].copy()
        state_hist = state_hist.sort_values("date").tail(last_n_months)
        for _, row in state_hist.iterrows():
            history_points.append(ForecastPoint(
                date=str(row["date"].date()),
                value=round(float(row["total"]), 2)
            ))

    forecast_points = [
        ForecastPoint(date=d, value=round(v, 2))
        for d, v in zip(data["forecast_dates"], data["forecast_values"])
    ]

    return HistoryForecastResponse(
        state=state,
        history=history_points,
        forecast=forecast_points,
        best_model=data["best_model"],
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True,
                log_level="info")
