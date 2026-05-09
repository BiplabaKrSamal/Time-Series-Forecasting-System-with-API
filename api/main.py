"""
api/main.py
===========
Production FastAPI — State Sales Forecasting System

Start locally:
    uvicorn api.main:app --reload --port 8000

Production:
    gunicorn api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
"""

import json, logging, os, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path resolution (works locally AND in Docker/Railway) ─────────────────────
# Try: env var → sibling of api/ → current dir
def _resolve_forecasts_path() -> Path:
    if os.getenv("FORECASTS_PATH"):
        return Path(os.getenv("FORECASTS_PATH"))
    candidates = [
        Path(__file__).parents[1] / "outputs" / "forecasts" / "all_forecasts.json",
        Path("/app/outputs/forecasts/all_forecasts.json"),
        Path("outputs/forecasts/all_forecasts.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]   # will warn at startup

def _resolve_src_path() -> Path:
    candidates = [
        Path(__file__).parents[1] / "src",
        Path("/app/src"),
        Path("src"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

SRC_PATH = _resolve_src_path()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="State Sales Forecasting API",
    description=(
        "Production time-series forecasting service. "
        "SARIMA · Prophet · XGBoost · LSTM — auto-selects best model per state. "
        "43 US states · 8-week forecast horizon."
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

# ── Global state ──────────────────────────────────────────────────────────────
_forecasts: Dict = {}
_panel: Optional[pd.DataFrame] = None
FORECASTS_PATH = _resolve_forecasts_path()

def _load_forecasts():
    global _forecasts
    if FORECASTS_PATH.exists():
        with open(FORECASTS_PATH) as f:
            _forecasts = json.load(f)
        logger.info(f"Loaded forecasts for {len(_forecasts)} states from {FORECASTS_PATH}")
    else:
        logger.warning(f"Forecasts file not found: {FORECASTS_PATH} — run python run_pipeline.py first")

def _load_panel():
    global _panel
    try:
        from preprocessing import load_raw_data, clean_data, build_monthly_panel
        raw = load_raw_data(); clean = clean_data(raw); _panel = build_monthly_panel(clean)
        logger.info(f"Historical panel loaded: {len(_panel):,} rows")
    except Exception as e:
        logger.warning(f"Could not load historical panel: {e}")

@app.on_event("startup")
async def startup_event():
    _load_forecasts()
    _load_panel()

# ── Schemas ───────────────────────────────────────────────────────────────────
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
    state:         str
    best_model:    str
    metrics:       List[ModelMetrics]
    all_forecasts: Dict[str, List[ForecastPoint]]

class BatchRequest(BaseModel):
    states: List[str] = Field(..., description="List of state names")

class HistoryForecastResponse(BaseModel):
    state:      str
    best_model: str
    history:    List[ForecastPoint]
    forecast:   List[ForecastPoint]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_state(state: str) -> Dict:
    if state in _forecasts:
        return _forecasts[state]
    for k, v in _forecasts.items():
        if k.lower() == state.lower():
            return v
    available = list(_forecasts.keys())[:5]
    raise HTTPException(
        status_code=404,
        detail=f"State '{state}' not found. Available: {available}… Use GET /states for full list."
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":        "ok",
        "service":       "State Sales Forecasting API",
        "version":       "1.0.0",
        "states_loaded": len(_forecasts),
        "docs":          "/docs",
        "endpoints": [
            "GET  /states",
            "GET  /forecast/{state}",
            "GET  /forecast/{state}/compare",
            "GET  /forecast/{state}/history",
            "POST /forecast/batch",
            "GET  /models/leaderboard",
        ]
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "states_loaded": len(_forecasts),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/states", response_model=List[str], tags=["Data"])
def list_states():
    if not _forecasts:
        raise HTTPException(status_code=503, detail="Run python run_pipeline.py first.")
    return sorted(_forecasts.keys())

@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(state: str):
    """Best-model 8-week sales forecast for one state."""
    data = _get_state(state)
    return ForecastResponse(
        state=state,
        best_model=data["best_model"],
        forecast_steps=len(data["forecast_dates"]),
        forecast=[ForecastPoint(date=d, value=round(v, 2))
                  for d, v in zip(data["forecast_dates"], data["forecast_values"])],
    )

@app.get("/forecast/{state}/compare", response_model=CompareResponse, tags=["Forecast"])
def compare_models(state: str):
    """All-model validation metrics + individual forecasts for a state."""
    data = _get_state(state)
    dates = data["forecast_dates"]
    all_fc = {
        m: [ForecastPoint(date=d, value=round(v, 2)) for d, v in zip(dates, vals)]
        for m, vals in data.get("all_model_forecasts", {}).items()
    }
    return CompareResponse(
        state=state,
        best_model=data["best_model"],
        metrics=[ModelMetrics(**m) for m in data.get("metrics", [])],
        all_forecasts=all_fc,
    )

@app.post("/forecast/batch", tags=["Forecast"])
def batch_forecast(request: BatchRequest):
    """Forecasts for multiple states in one request."""
    results, errors = {}, {}
    for state in request.states:
        try:
            data = _get_state(state)
            results[state] = {
                "best_model":     data["best_model"],
                "forecast_dates": data["forecast_dates"],
                "forecast":       [round(v, 2) for v in data["forecast_values"]],
            }
        except HTTPException as e:
            errors[state] = e.detail
    return {"forecasts": results, "errors": errors}

@app.get("/models/leaderboard", tags=["Models"])
def leaderboard():
    """Global model win count across all states."""
    if not _forecasts:
        raise HTTPException(status_code=503, detail="No forecasts loaded.")
    wins:     Dict[str, int]        = {}
    avg_mape: Dict[str, List[float]] = {}
    for d in _forecasts.values():
        bm = d.get("best_model", "?")
        wins[bm] = wins.get(bm, 0) + 1
        for m in d.get("metrics", []):
            avg_mape.setdefault(m["model"], []).append(m["MAPE"])
    return {
        "total_states": len(_forecasts),
        "leaderboard": sorted([
            {"model": m, "wins": w,
             "win_pct": round(w / len(_forecasts) * 100, 1),
             "avg_val_MAPE": round(float(np.mean(avg_mape.get(m, [0]))), 2)}
            for m, w in wins.items()
        ], key=lambda x: -x["wins"])
    }

@app.get("/forecast/{state}/history", response_model=HistoryForecastResponse,
         tags=["Forecast"])
def history_and_forecast(
    state: str,
    last_n_months: int = Query(default=36, ge=3, le=120)
):
    """Historical sales + forecast — chart-ready data."""
    data = _get_state(state)
    history = []
    if _panel is not None:
        hist_df = (_panel[_panel["state"].str.lower() == state.lower()]
                   .sort_values("date").tail(last_n_months))
        history = [ForecastPoint(date=str(r["date"].date()), value=round(float(r["total"]), 2))
                   for _, r in hist_df.iterrows()]
    return HistoryForecastResponse(
        state=state,
        best_model=data["best_model"],
        history=history,
        forecast=[ForecastPoint(date=d, value=round(v, 2))
                  for d, v in zip(data["forecast_dates"], data["forecast_values"])],
    )

# ── Dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)),
                reload=False, log_level="info")
