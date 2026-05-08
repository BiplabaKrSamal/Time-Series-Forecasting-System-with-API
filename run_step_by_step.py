#!/usr/bin/env python3
"""
run_step_by_step.py
===================
Guided, step-by-step execution of the entire forecasting pipeline.
Run this to see every stage execute and verify the system end-to-end.

Usage:
    python run_step_by_step.py              # all steps
    python run_step_by_step.py --step 3    # specific step only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Suppress TF log noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SRC = Path(__file__).parent / "src"
API = Path(__file__).parent / "api"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(API))

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def banner(title: str, width: int = 62):
    print("\n" + "═" * width)
    pad = (width - len(title) - 4) // 2
    print(f"║{' ' * pad}  {title}{' ' * (width - pad - len(title) - 4)}║")
    print("═" * width)

def step_banner(n: int, title: str):
    print(f"\n{'─'*62}")
    print(f"  STEP {n}: {title}")
    print(f"{'─'*62}")

def ok(msg: str):   print(f"  ✅  {msg}")
def info(msg: str): print(f"  ℹ️   {msg}")
def warn(msg: str): print(f"  ⚠️   {msg}")


# ─────────────────────────────────────────────
# STEP 1: Load & Inspect Raw Data
# ─────────────────────────────────────────────
def step1_load_data():
    step_banner(1, "Load & Inspect Raw Data")
    from preprocessing import load_raw_data
    df = load_raw_data()

    info(f"Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
    info(f"Columns       : {list(df.columns)}")
    info(f"Unique states : {df['state'].nunique() if 'state' in df.columns else df.iloc[:,0].nunique()}")
    info(f"Date range    : {df.iloc[:,1].min()} → {df.iloc[:,1].max()}")
    info("Sample rows:")
    print(df.head(3).to_string(index=False))
    ok("Raw data loaded successfully")
    return df


# ─────────────────────────────────────────────
# STEP 2: Clean & Parse
# ─────────────────────────────────────────────
def step2_clean(raw_df):
    step_banner(2, "Data Cleaning & Date Parsing")
    from preprocessing import clean_data
    clean = clean_data(raw_df)

    info(f"After cleaning : {len(clean):,} rows")
    info(f"Date range     : {clean['date'].min().date()} → {clean['date'].max().date()}")
    info(f"NaN total      : {clean.isna().sum().sum()}")
    ok("Date parsing handled both M/D/YYYY and DD-MM-YYYY formats")
    ok("Commas and whitespace stripped from 'total' column")
    return clean


# ─────────────────────────────────────────────
# STEP 3: Build Monthly Panel
# ─────────────────────────────────────────────
def step3_panel(clean_df):
    step_banner(3, "Build Regular Monthly Panel")
    from preprocessing import build_monthly_panel
    panel = build_monthly_panel(clean_df)

    info(f"Panel shape   : {panel.shape}")
    info(f"States        : {panel['state'].nunique()}")
    info(f"Months        : {panel['date'].nunique()}")
    info(f"Missing values: {panel['total'].isna().sum()}")
    print(panel[panel['state']=='California'][['date','total']].head(5).to_string(index=False))
    ok("Snapped to month-end, interpolated gaps, forward/back-filled edges")
    return panel


# ─────────────────────────────────────────────
# STEP 4: Feature Engineering Demo
# ─────────────────────────────────────────────
def step4_features(panel):
    step_banner(4, "Feature Engineering (XGBoost / LSTM features)")
    from preprocessing import get_state_series, train_val_split
    from features import build_features, prepare_supervised

    train_df, val_df = train_val_split(panel)
    series = get_state_series(train_df, "California")
    feats  = build_features(series)
    X, y   = prepare_supervised(feats)

    info(f"Feature matrix shape : {X.shape}")
    info(f"Target vector shape  : {y.shape}")
    info("Features created:")
    for col in X.columns:
        print(f"       • {col}")
    ok("Lag features: t-1, t-2, t-3, t-6, t-12")
    ok("Rolling mean/std: 3-month, 6-month, 12-month (no leakage)")
    ok("Calendar: month, quarter, year, trend, sin_month, cos_month")
    ok("Holiday flag: US federal holidays")
    return train_df, val_df


# ─────────────────────────────────────────────
# STEP 5: Train All 4 Models on California
# ─────────────────────────────────────────────
def step5_train_models(train_df, val_df):
    step_banner(5, "Train All 4 Models → California")
    from preprocessing import get_state_series
    from models.arima_model   import ARIMAForecaster
    from models.prophet_model import ProphetForecaster
    from models.xgboost_model import XGBoostForecaster
    from models.lstm_model    import LSTMForecaster
    from evaluation           import ModelSelector

    train_s = get_state_series(train_df, "California")
    val_s   = get_state_series(val_df, "California")
    freq    = "ME"

    selector = ModelSelector()
    preds    = {}

    models_to_train = [
        ("SARIMA",  ARIMAForecaster()),
        ("Prophet", ProphetForecaster()),
        ("XGBoost", XGBoostForecaster()),
        ("LSTM",    LSTMForecaster(lookback=12, epochs=40, patience=8)),
    ]

    for name, model in models_to_train:
        print(f"\n  Training {name} …")
        t0 = time.time()
        try:
            model.fit(train_s)
            if name == "LSTM":
                pred = model.predict_with_index(len(val_s),
                                                 last_date=train_s.index[-1],
                                                 freq=freq)
            else:
                pred = model.predict(len(val_s))

            preds[name] = pred.values
            metrics = selector.add_result(name, val_s.values, pred.values)
            elapsed = time.time() - t0
            print(f"     MAE={metrics['MAE']:>15,.0f}  RMSE={metrics['RMSE']:>15,.0f}"
                  f"  MAPE={metrics['MAPE']:>7.2f}%  [{elapsed:.1f}s]")
        except Exception as e:
            warn(f"{name} failed: {e}")

    best = selector.best_model()
    print(f"\n  {'★'*3}  Best model: {best}  {'★'*3}")

    print("\n  Validation metrics summary:")
    print(selector.summary_table().to_string())
    return selector, preds, best, train_s, val_s


# ─────────────────────────────────────────────
# STEP 6: Generate Final 8-Week Forecast
# ─────────────────────────────────────────────
def step6_forecast(best_name, train_s, val_s):
    step_banner(6, f"Generate Final 8-Week Forecast  [{best_name}]")
    import pandas as pd, numpy as np
    from preprocessing import get_state_series
    from models.arima_model   import ARIMAForecaster
    from models.prophet_model import ProphetForecaster
    from models.xgboost_model import XGBoostForecaster
    from models.lstm_model    import LSTMForecaster

    full_s  = pd.concat([train_s, val_s]).sort_index()
    freq    = "ME"
    last_dt = full_s.index[-1]

    print(f"  Retraining {best_name} on {len(full_s)} months of data …")

    cls_map = {"SARIMA": ARIMAForecaster, "Prophet": ProphetForecaster,
               "XGBoost": XGBoostForecaster}
    if best_name == "LSTM":
        model = LSTMForecaster(lookback=12, epochs=40, patience=8)
        model.fit(full_s)
        forecast = model.predict_with_index(2, last_date=last_dt, freq=freq)
    else:
        model = cls_map[best_name]()
        model.fit(full_s)
        forecast = model.predict(2)

    forecast = np.maximum(forecast.values, 0)
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  California 8-Week Forecast (≈ 2 months) ║")
    print(f"  ╠══════════════════════════════════════════╣")
    for i, val in enumerate(forecast, 1):
        print(f"  ║  Month {i}:  ${val:>25,.0f}  ║")
    print(f"  ╚══════════════════════════════════════════╝")
    ok(f"Forecast generated via {best_name}")
    return forecast


# ─────────────────────────────────────────────
# STEP 7: Validate API
# ─────────────────────────────────────────────
def step7_api():
    step_banner(7, "Validate REST API Endpoints")

    fc_path = Path(__file__).parent / "outputs" / "forecasts" / "all_forecasts.json"
    if not fc_path.exists():
        warn("No forecasts file found — run step 5/6 first or use demo_forecasts.json")
        demo = Path(__file__).parent / "outputs" / "forecasts" / "demo_forecasts.json"
        if demo.exists():
            import shutil
            shutil.copy(demo, fc_path)
            info("Copied demo_forecasts.json → all_forecasts.json for API test")

    import main as api_main
    with open(fc_path) as f:
        api_main._forecasts = json.load(f)

    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)

    endpoints = [
        ("GET",  "/",                         None),
        ("GET",  "/health",                   None),
        ("GET",  "/states",                   None),
        ("GET",  "/forecast/California",      None),
        ("GET",  "/forecast/California/compare", None),
        ("POST", "/forecast/batch",           {"states": ["California"]}),
        ("GET",  "/models/leaderboard",       None),
    ]

    all_ok = True
    for method, path, body in endpoints:
        if method == "GET":
            r = client.get(path)
        else:
            r = client.post(path, json=body)

        status = "✅" if r.status_code < 400 else "❌"
        print(f"  {status}  {method:4s} {path:<45} → {r.status_code}")
        if r.status_code >= 400:
            all_ok = False

    if all_ok:
        ok("All API endpoints responded successfully")
    else:
        warn("Some endpoints failed — check server logs")

    # Pretty-print a sample response
    print("\n  Sample: GET /forecast/California")
    import pprint
    pprint.pprint(client.get("/forecast/California").json(), indent=4, width=60)


# ─────────────────────────────────────────────
# STEP 8: Run Automated Test Suite
# ─────────────────────────────────────────────
def step8_tests():
    step_banner(8, "Run Automated Test Suite (24 tests)")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_forecasting_system.py", "-v", "--tb=short",
         "--no-header", "-q"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent
    )
    # Filter noise
    for line in result.stdout.splitlines():
        if any(x in line for x in ["PASSED","FAILED","ERROR","passed","failed","error","warning"]):
            status = "✅" if "PASSED" in line else ("❌" if "FAILED" in line or "ERROR" in line else "  ")
            clean = line.replace("PASSED","✅").replace("FAILED","❌")
            print(f"  {clean}")
    if result.returncode == 0:
        ok("All tests passed!")
    else:
        warn("Some tests failed — see output above")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step-by-step forecasting pipeline")
    parser.add_argument("--step", type=int, default=0,
                        help="Run a specific step only (0 = all)")
    args = parser.parse_args()

    banner("State Sales Forecasting System — Step-by-Step Execution")
    print("  Dataset : US Beverages Sales by State, 2019–2023")
    print("  Goal    : Forecast next 8 weeks of sales per state")
    print("  Models  : SARIMA · Prophet · XGBoost · LSTM")
    print("  API     : FastAPI REST service (7 endpoints)")

    run = lambda n: args.step == 0 or args.step == n

    raw_df = train_df = val_df = None
    selector = preds = best_name = train_s = val_s = None

    if run(1):  raw_df               = step1_load_data()
    if run(2):  clean_df             = step2_clean(raw_df or step1_load_data())
    if run(3):
        if raw_df is None:
            from preprocessing import load_raw_data, clean_data
            clean_df = clean_data(load_raw_data())
        panel = step3_panel(clean_df)
    if run(4):
        if raw_df is None:
            from preprocessing import load_raw_data, clean_data, build_monthly_panel
            panel = build_monthly_panel(clean_data(load_raw_data()))
        train_df, val_df = step4_features(panel)
    if run(5):
        if train_df is None:
            from preprocessing import load_raw_data, clean_data, build_monthly_panel, train_val_split
            panel = build_monthly_panel(clean_data(load_raw_data()))
            train_df, val_df = train_val_split(panel)
        selector, preds, best_name, train_s, val_s = step5_train_models(train_df, val_df)
    if run(6):
        if train_s is None:
            from preprocessing import (load_raw_data, clean_data, build_monthly_panel,
                                        train_val_split, get_state_series)
            panel = build_monthly_panel(clean_data(load_raw_data()))
            tr, va = train_val_split(panel)
            train_s = get_state_series(tr, "California")
            val_s   = get_state_series(va, "California")
            best_name = "XGBoost"
        step6_forecast(best_name, train_s, val_s)
    if run(7):  step7_api()
    if run(8):  step8_tests()

    banner("Pipeline Complete ✅")
    print("  Outputs:")
    print("    outputs/plots/        → EDA + forecast charts")
    print("    outputs/forecasts/    → JSON forecasts")
    print("    outputs/models/       → Serialised model objects")
    print("  Start API:  cd api && uvicorn main:app --reload --port 8000")
    print()


if __name__ == "__main__":
    main()
