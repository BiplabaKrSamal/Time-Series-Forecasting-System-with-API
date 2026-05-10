"""
api/main.py
===========
Production FastAPI — State Sales Forecasting System

Start locally:   cd api && uvicorn main:app --reload --port 8000
Production:      gunicorn api.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
"""

import json, logging, os, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ── Path resolution (works locally AND in Docker/Railway/Render) ──────────────
def _find(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return Path(candidates[0])

SRC_PATH = _find([
    Path(__file__).parents[1] / "src",
    "/app/src", "src",
])
FORECASTS_PATH = _find([
    os.getenv("FORECASTS_PATH", ""),
    Path(__file__).parents[1] / "outputs" / "forecasts" / "all_forecasts.json",
    "/app/outputs/forecasts/all_forecasts.json",
    "outputs/forecasts/all_forecasts.json",
])
DASHBOARD_PATH = _find([
    Path(__file__).parent / "dashboard.html",
    "/app/api/dashboard.html",
    "api/dashboard.html",
])

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="State Sales Forecasting API",
    description="SARIMA · Prophet · XGBoost · LSTM — 43 US states · 8-week forecast",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Global state ──────────────────────────────────────────────────────────────
_forecasts: Dict = {}
_panel: Optional[pd.DataFrame] = None

def _load_forecasts():
    global _forecasts
    if FORECASTS_PATH.exists():
        with open(FORECASTS_PATH) as f:
            _forecasts = json.load(f)
        logger.info(f"Loaded {len(_forecasts)} state forecasts")
    else:
        logger.warning(f"No forecasts at {FORECASTS_PATH} — run python run_pipeline.py first")

def _load_panel():
    global _panel
    try:
        from preprocessing import load_raw_data, clean_data, build_monthly_panel
        _panel = build_monthly_panel(clean_data(load_raw_data()))
        logger.info(f"Panel loaded: {len(_panel):,} rows")
    except Exception as e:
        logger.warning(f"Panel not loaded: {e}")

@app.on_event("startup")
async def startup():
    _load_forecasts()
    _load_panel()

# ── Dashboard (root) ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Serve the interactive dashboard at root URL."""
    if DASHBOARD_PATH.exists():
        return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))
    # Fallback: plain JSON status
    return HTMLResponse(f"""<!DOCTYPE html><html><body style="font-family:monospace;background:#07080f;color:#e8eaf6;padding:40px">
    <h2>🔮 State Sales Forecasting API</h2>
    <p>States loaded: <strong>{len(_forecasts)}</strong></p>
    <p><a href="/docs" style="color:#7c6dfa">→ Swagger UI</a> &nbsp;
       <a href="/states" style="color:#06d6a0">→ /states</a> &nbsp;
       <a href="/models/leaderboard" style="color:#06d6a0">→ /leaderboard</a></p>
    </body></html>""")

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
    states: List[str]

class HistoryResponse(BaseModel):
    state:      str
    best_model: str
    history:    List[ForecastPoint]
    forecast:   List[ForecastPoint]

# ── Helper ────────────────────────────────────────────────────────────────────
def _get(state: str) -> Dict:
    if state in _forecasts:
        return _forecasts[state]
    for k, v in _forecasts.items():
        if k.lower() == state.lower():
            return v
    raise HTTPException(404, f"'{state}' not found. GET /states for full list.")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "states_loaded": len(_forecasts),
            "timestamp": datetime.utcnow().isoformat()}

@app.get("/states", response_model=List[str], tags=["Data"])
def list_states():
    if not _forecasts:
        raise HTTPException(503, "Run python run_pipeline.py first.")
    return sorted(_forecasts.keys())

@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(state: str):
    """Best-model 8-week sales forecast for one state."""
    d = _get(state)
    return ForecastResponse(
        state=state, best_model=d["best_model"],
        forecast_steps=len(d["forecast_dates"]),
        forecast=[ForecastPoint(date=dt, value=round(v, 2))
                  for dt, v in zip(d["forecast_dates"], d["forecast_values"])],
    )

@app.get("/forecast/{state}/compare", response_model=CompareResponse, tags=["Forecast"])
def compare(state: str):
    """All-model validation metrics + forecasts for a state."""
    d = _get(state)
    return CompareResponse(
        state=state, best_model=d["best_model"],
        metrics=[ModelMetrics(**m) for m in d.get("metrics", [])],
        all_forecasts={
            m: [ForecastPoint(date=dt, value=round(v, 2))
                for dt, v in zip(d["forecast_dates"], vals)]
            for m, vals in d.get("all_model_forecasts", {}).items()
        },
    )

@app.post("/forecast/batch", tags=["Forecast"])
def batch(req: BatchRequest):
    """Forecasts for multiple states in one request."""
    results, errors = {}, {}
    for s in req.states:
        try:
            d = _get(s)
            results[s] = {"best_model": d["best_model"],
                          "forecast_dates": d["forecast_dates"],
                          "forecast": [round(v, 2) for v in d["forecast_values"]]}
        except HTTPException as e:
            errors[s] = e.detail
    return {"forecasts": results, "errors": errors}

@app.get("/models/leaderboard", tags=["Models"])
def leaderboard():
    """Global model win counts across all 43 states."""
    if not _forecasts:
        raise HTTPException(503, "No forecasts loaded.")
    wins, avg_mape = {}, {}
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

@app.get("/forecast/{state}/history", response_model=HistoryResponse, tags=["Forecast"])
def history(state: str, last_n_months: int = Query(default=36, ge=3, le=120)):
    """Historical + forecast — chart-ready data."""
    d = _get(state)
    hist = []
    if _panel is not None:
        rows = _panel[_panel["state"].str.lower() == state.lower()].sort_values("date").tail(last_n_months)
        hist = [ForecastPoint(date=str(r["date"].date()), value=round(float(r["total"]), 2))
                for _, r in rows.iterrows()]
    return HistoryResponse(
        state=state, best_model=d["best_model"], history=hist,
        forecast=[ForecastPoint(date=dt, value=round(v, 2))
                  for dt, v in zip(d["forecast_dates"], d["forecast_values"])],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
