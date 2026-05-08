"""
forecaster.py
=============
Master orchestrator: trains all four models per state, evaluates on the
validation set, selects the best model, and produces the final 8-week
(2-month) forecast.

This module is the single entry-point for the training pipeline.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from preprocessing import (
    load_raw_data, clean_data, build_monthly_panel,
    train_val_split, get_state_series, FORECAST_HORIZON
)
from features import build_features, prepare_supervised
from models.arima_model   import ARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model    import LSTMForecaster
from evaluation import ModelSelector

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parents[1] / "outputs"


# ---------------------------------------------------------------------------
# Per-state training helper
# ---------------------------------------------------------------------------

def _train_single_state(state: str,
                         train_series: pd.Series,
                         val_series:   pd.Series,
                         forecast_steps: int = 2   # 8 weeks ≈ 2 months
                         ) -> Dict:
    """
    Train all four models on `train_series`, evaluate on `val_series`,
    select the best, then retrain on train+val and return final forecast.

    Returns
    -------
    dict with keys:
        state, best_model, metrics, forecast_dates, forecast_values,
        all_model_forecasts, model_objects
    """
    logger.info(f"\n{'='*60}")
    logger.info(f" STATE: {state}")
    logger.info(f"  Train: {len(train_series)} pts | Val: {len(val_series)} pts")

    selector = ModelSelector(primary_metric="MAPE")
    val_steps = len(val_series)
    model_objects: Dict = {}
    val_forecasts: Dict = {}

    # --- 1. SARIMA --------------------------------------------------------
    try:
        arima = ARIMAForecaster()
        arima.fit(train_series)
        arima_pred = arima.predict(val_steps)
        val_forecasts["SARIMA"] = arima_pred.values
        selector.add_result("SARIMA", val_series.values, arima_pred.values)
        model_objects["SARIMA"] = arima
    except Exception as e:
        logger.warning(f"  [SARIMA] FAILED: {e}")

    # --- 2. Prophet -------------------------------------------------------
    try:
        prophet = ProphetForecaster()
        prophet.fit(train_series)
        prophet_pred = prophet.predict(val_steps)
        val_forecasts["Prophet"] = prophet_pred.values
        selector.add_result("Prophet", val_series.values, prophet_pred.values)
        model_objects["Prophet"] = prophet
    except Exception as e:
        logger.warning(f"  [Prophet] FAILED: {e}")

    # --- 3. XGBoost -------------------------------------------------------
    try:
        xgb_model = XGBoostForecaster()
        xgb_model.fit(train_series)
        xgb_pred = xgb_model.predict(val_steps)
        val_forecasts["XGBoost"] = xgb_pred.values
        selector.add_result("XGBoost", val_series.values, xgb_pred.values)
        model_objects["XGBoost"] = xgb_model
    except Exception as e:
        logger.warning(f"  [XGBoost] FAILED: {e}")

    # --- 4. LSTM ----------------------------------------------------------
    try:
        lookback = min(12, len(train_series) // 2)
        lstm = LSTMForecaster(lookback=lookback, epochs=60, patience=10)
        lstm.fit(train_series)
        lstm_pred = lstm.predict_with_index(val_steps,
                                             last_date=train_series.index[-1],
                                             freq=pd.infer_freq(train_series.index) or "MS")
        val_forecasts["LSTM"] = lstm_pred.values
        selector.add_result("LSTM", val_series.values, lstm_pred.values)
        model_objects["LSTM"] = lstm
    except Exception as e:
        logger.warning(f"  [LSTM] FAILED: {e}")

    # --- Pick best model ---
    if not selector._results:
        raise RuntimeError(f"All models failed for state: {state}")

    best_name = selector.best_model()

    # --- Retrain best model on full data (train + val) --------------------
    full_series = pd.concat([train_series, val_series]).sort_index()
    freq = pd.infer_freq(full_series.index) or "MS"
    last_date = full_series.index[-1]

    logger.info(f"  Retraining {best_name} on full series ({len(full_series)} pts) …")

    if best_name == "SARIMA":
        final_model = ARIMAForecaster()
        final_model.fit(full_series)
        final_forecast = final_model.predict(forecast_steps)
        forecast_index = final_forecast.index

    elif best_name == "Prophet":
        final_model = ProphetForecaster()
        final_model.fit(full_series)
        final_forecast = final_model.predict(forecast_steps)
        forecast_index = final_forecast.index

    elif best_name == "XGBoost":
        final_model = XGBoostForecaster()
        final_model.fit(full_series)
        final_forecast = final_model.predict(forecast_steps)
        forecast_index = final_forecast.index

    elif best_name == "LSTM":
        lookback = min(12, len(full_series) // 2)
        final_model = LSTMForecaster(lookback=lookback, epochs=60, patience=10)
        final_model.fit(full_series)
        final_forecast = final_model.predict_with_index(forecast_steps,
                                                          last_date=last_date,
                                                          freq=freq)
        forecast_index = final_forecast.index

    forecast_values = np.maximum(np.array(final_forecast.values), 0)

    # Also compute all-model forecasts on the full series (for API comparison)
    all_forecasts: Dict = {}
    for mname, mobj in model_objects.items():
        try:
            if mname == "SARIMA":
                f_new = ARIMAForecaster()
                f_new.fit(full_series)
                f_out = f_new.predict(forecast_steps)
                all_forecasts[mname] = np.maximum(f_out.values, 0).tolist()
            elif mname == "Prophet":
                f_new = ProphetForecaster()
                f_new.fit(full_series)
                f_out = f_new.predict(forecast_steps)
                all_forecasts[mname] = np.maximum(f_out.values, 0).tolist()
            elif mname == "XGBoost":
                f_new = XGBoostForecaster()
                f_new.fit(full_series)
                f_out = f_new.predict(forecast_steps)
                all_forecasts[mname] = np.maximum(f_out.values, 0).tolist()
            elif mname == "LSTM":
                f_new = LSTMForecaster(lookback=min(12, len(full_series) // 2),
                                       epochs=60, patience=10)
                f_new.fit(full_series)
                f_out = f_new.predict_with_index(forecast_steps,
                                                  last_date=last_date, freq=freq)
                all_forecasts[mname] = np.maximum(f_out.values, 0).tolist()
        except Exception as e:
            logger.warning(f"  [{mname}] retrain for all-model comparison failed: {e}")

    return {
        "state":               state,
        "best_model":          best_name,
        "metrics":             selector.to_dict(),
        "forecast_dates":      [str(d.date()) for d in forecast_index],
        "forecast_values":     forecast_values.tolist(),
        "all_model_forecasts": all_forecasts,
        "model":               final_model,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(states: Optional[List[str]] = None,
                 forecast_steps: int = 2) -> Dict:
    """
    Run the complete forecasting pipeline.

    Parameters
    ----------
    states          : list of states to process (None → all)
    forecast_steps  : number of months to forecast (8 weeks ≈ 2 months)

    Returns
    -------
    dict  { state_name → result_dict }
    """
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  State Sales Forecasting Pipeline – START    ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # 1. Load & preprocess
    raw   = load_raw_data()
    clean = clean_data(raw)
    panel = build_monthly_panel(clean)

    # 2. Train/val split
    train_df, val_df = train_val_split(panel)

    # 3. Determine states to process
    all_states = sorted(panel["state"].unique())
    target_states = states if states else all_states
    logger.info(f"Processing {len(target_states)} states …")

    results: Dict = {}
    failed_states: List[str] = []

    for state in target_states:
        try:
            train_s = get_state_series(train_df, state)
            val_s   = get_state_series(val_df,   state)

            # Need at least lookback + a few training examples
            if len(train_s) < 15:
                logger.warning(f"  {state}: skipping (only {len(train_s)} train points)")
                continue

            result = _train_single_state(state, train_s, val_s, forecast_steps)
            results[state] = result

        except Exception as e:
            logger.error(f"  {state}: PIPELINE ERROR: {e}", exc_info=True)
            failed_states.append(state)

    # 4. Save artifacts
    _save_results(results, forecast_steps)

    logger.info(f"\n✅ Pipeline complete: {len(results)} states processed, "
                f"{len(failed_states)} failed: {failed_states}")
    return results


def _save_results(results: Dict, forecast_steps: int):
    """Persist forecast JSON and model objects."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)
    (OUTPUT_DIR / "forecasts").mkdir(exist_ok=True)

    # JSON summary (serialisable)
    summary = {}
    for state, r in results.items():
        summary[state] = {
            "best_model":          r["best_model"],
            "forecast_dates":      r["forecast_dates"],
            "forecast_values":     r["forecast_values"],
            "all_model_forecasts": r["all_model_forecasts"],
            "metrics":             r["metrics"],
        }

    json_path = OUTPUT_DIR / "forecasts" / "all_forecasts.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Forecasts saved → {json_path}")

    # Model objects (joblib)
    for state, r in results.items():
        model_path = OUTPUT_DIR / "models" / f"{state.replace(' ', '_')}_best_model.pkl"
        try:
            joblib.dump(r["model"], model_path)
        except Exception as e:
            logger.warning(f"  Could not save model for {state}: {e}")

    logger.info(f"  Models saved → {OUTPUT_DIR / 'models'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Allow specifying a subset of states via CLI for quick testing
    # e.g. python forecaster.py "California" "Texas"
    requested_states = sys.argv[1:] or None
    results = run_pipeline(states=requested_states, forecast_steps=2)

    # Print summary table
    rows = []
    for state, r in results.items():
        rows.append({
            "State": state,
            "Best Model": r["best_model"],
            "Forecast Month 1": f"{r['forecast_values'][0]:,.0f}",
            "Forecast Month 2": f"{r['forecast_values'][1]:,.0f}",
        })
    print("\n" + pd.DataFrame(rows).to_string(index=False))
