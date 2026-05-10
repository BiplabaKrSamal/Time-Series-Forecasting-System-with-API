"""
api/main.py  —  State Sales Forecasting API
============================================
Start via wsgi.py (recommended):
    uvicorn wsgi:app --reload --port 8000

Or directly from repo root:
    PYTHONPATH=src:api uvicorn api.main:app --port 8000
"""

import json, logging, os, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ── Reliable path resolution ──────────────────────────────────────────────────
# __file__ is always the absolute path to this file regardless of how it's run
_API_DIR  = Path(__file__).parent          # .../api/
_ROOT     = _API_DIR.parent               # repo root

_SRC      = _ROOT / "src"
_FC_JSON  = _ROOT / "outputs" / "forecasts" / "all_forecasts.json"
_DASH     = _API_DIR / "dashboard.html"

# Add src/ to path so preprocessing.py etc. are importable
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="State Sales Forecasting API",
    description="SARIMA · Prophet · XGBoost · LSTM — 43 US states · 8-week forecast horizon",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Global state ──────────────────────────────────────────────────────────────
_fc:    Dict = {}
_panel: Optional[pd.DataFrame] = None

def _load():
    global _fc, _panel

    # Forecasts JSON (mandatory)
    fc_path = Path(os.getenv("FORECASTS_PATH", "")) if os.getenv("FORECASTS_PATH") else _FC_JSON
    if fc_path.is_file():
        with open(fc_path) as f:
            _fc = json.load(f)
        log.info(f"✅ Loaded {len(_fc)} state forecasts from {fc_path}")
    else:
        log.error(f"❌ Forecasts not found at {fc_path}. Run: python run_pipeline.py")

    # Historical panel (optional — only used by /history endpoint)
    try:
        from preprocessing import load_raw_data, clean_data, build_monthly_panel
        _panel = build_monthly_panel(clean_data(load_raw_data()))
        log.info(f"✅ Panel: {len(_panel):,} rows")
    except Exception as e:
        log.warning(f"Panel not loaded (optional): {e}")

@app.on_event("startup")
async def startup():
    _load()

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    if _DASH.is_file():
        return HTMLResponse(_DASH.read_text(encoding="utf-8"))
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>body{{font-family:monospace;background:#07080f;color:#e8eaf6;padding:48px;line-height:2}}</style></head>
<body><h2>🔮 State Sales Forecasting API</h2>
<p>States loaded: <b>{len(_fc)}</b></p>
<p><a href='/docs' style='color:#7c6dfa'>→ Swagger UI</a> &nbsp;
   <a href='/forecast/California' style='color:#06d6a0'>→ /forecast/California</a> &nbsp;
   <a href='/models/leaderboard' style='color:#06d6a0'>→ /leaderboard</a></p>
</body></html>""")

# ── Schemas ───────────────────────────────────────────────────────────────────
class ForecastPoint(BaseModel):
    date: str; value: float

class ForecastResponse(BaseModel):
    state: str; best_model: str; forecast_steps: int
    unit: str = "USD (monthly sales)"
    forecast: List[ForecastPoint]
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ModelMetrics(BaseModel):
    model: str; MAE: float; RMSE: float; MAPE: float; sMAPE: float

class CompareResponse(BaseModel):
    state: str; best_model: str
    metrics: List[ModelMetrics]
    all_forecasts: Dict[str, List[ForecastPoint]]

class BatchRequest(BaseModel):
    states: List[str]

class HistoryResponse(BaseModel):
    state: str; best_model: str
    history: List[ForecastPoint]; forecast: List[ForecastPoint]

# ── Helper ────────────────────────────────────────────────────────────────────
def _get(state: str) -> Dict:
    if state in _fc: return _fc[state]
    for k, v in _fc.items():
        if k.lower() == state.lower(): return v
    raise HTTPException(404, f"'{state}' not found. GET /states for list.")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "states_loaded": len(_fc),
            "timestamp": datetime.utcnow().isoformat()}

@app.get("/states", response_model=List[str], tags=["Data"])
def states():
    if not _fc: raise HTTPException(503, "Run python run_pipeline.py first.")
    return sorted(_fc.keys())

@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def forecast(state: str):
    d = _get(state)
    return ForecastResponse(
        state=state, best_model=d["best_model"],
        forecast_steps=len(d["forecast_dates"]),
        forecast=[ForecastPoint(date=dt, value=round(v, 2))
                  for dt, v in zip(d["forecast_dates"], d["forecast_values"])])

@app.get("/forecast/{state}/compare", response_model=CompareResponse, tags=["Forecast"])
def compare(state: str):
    d = _get(state)
    return CompareResponse(
        state=state, best_model=d["best_model"],
        metrics=[ModelMetrics(**m) for m in d.get("metrics", [])],
        all_forecasts={
            m: [ForecastPoint(date=dt, value=round(v, 2))
                for dt, v in zip(d["forecast_dates"], vals)]
            for m, vals in d.get("all_model_forecasts", {}).items()})

@app.post("/forecast/batch", tags=["Forecast"])
def batch(req: BatchRequest):
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
    if not _fc: raise HTTPException(503, "No forecasts loaded.")
    wins, mapes = {}, {}
    for d in _fc.values():
        bm = d.get("best_model", "?")
        wins[bm] = wins.get(bm, 0) + 1
        for m in d.get("metrics", []):
            mapes.setdefault(m["model"], []).append(m["MAPE"])
    return {
        "total_states": len(_fc),
        "leaderboard": sorted([
            {"model": m, "wins": w,
             "win_pct": round(w / len(_fc) * 100, 1),
             "avg_val_MAPE": round(float(np.mean(mapes.get(m, [0]))), 2)}
            for m, w in wins.items()], key=lambda x: -x["wins"])}

@app.get("/forecast/{state}/history", response_model=HistoryResponse, tags=["Forecast"])
def history(state: str, last_n_months: int = Query(default=36, ge=3, le=120)):
    d = _get(state)
    hist = []
    if _panel is not None:
        rows = (_panel[_panel["state"].str.lower() == state.lower()]
                .sort_values("date").tail(last_n_months))
        hist = [ForecastPoint(date=str(r["date"].date()), value=round(float(r["total"]), 2))
                for _, r in rows.iterrows()]
    return HistoryResponse(
        state=state, best_model=d["best_model"], history=hist,
        forecast=[ForecastPoint(date=dt, value=round(v, 2))
                  for dt, v in zip(d["forecast_dates"], d["forecast_values"])])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=False)
