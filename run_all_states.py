"""
run_all_states.py — trains all 4 models on every state, auto-selects best,
saves forecasts + model PKLs, prints summary table.
Usage:
    python run_all_states.py                   # all 43 states
    python run_all_states.py California Texas  # specific states
"""
import json, logging, os, sys, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np, pandas as pd, joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

from preprocessing import (load_raw_data, clean_data, build_monthly_panel,
                            train_val_split, get_state_series)
from models.arima_model   import ARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model    import LSTMForecaster
from evaluation import ModelSelector

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "outputs"
PLOTS  = OUTPUT / "plots"
for d in [PLOTS, OUTPUT/"models", OUTPUT/"forecasts"]:
    d.mkdir(parents=True, exist_ok=True)

COLORS = {"SARIMA":"#e74c3c","Prophet":"#9b59b6","XGBoost":"#2ecc71","LSTM":"#f39c12"}


def _fit_predict(name, cls, train_s, val_n, freq, last_dt, full_s=None, steps=2):
    """Helper: fit model, return (val_preds, forecast_values, model_obj)."""
    if name == "LSTM":
        lb = min(12, len(train_s) // 2)
        m  = LSTMForecaster(lookback=lb, epochs=60, patience=12)
    else:
        m = cls()
    m.fit(train_s if full_s is None else full_s)
    if full_s is None:          # validation pass
        p = m.predict(val_n)
        return np.array(p.values if hasattr(p,"values") else p), m
    else:                       # final forecast pass
        if name == "LSTM":
            fc = m.predict_with_index(steps, last_date=last_dt, freq=freq)
        else:
            fc = m.predict(steps)
        return np.maximum(np.array(fc.values if hasattr(fc,"values") else fc), 0), \
               fc.index if hasattr(fc,"index") else pd.date_range(last_dt,periods=steps+1,freq=freq)[1:], \
               m


def train_state(state, train_s, val_s, full_s, freq, last_dt):
    selector, val_preds = ModelSelector(), {}
    val_n = len(val_s)

    for name, cls in [("SARIMA", ARIMAForecaster),("Prophet", ProphetForecaster),
                       ("XGBoost", XGBoostForecaster),("LSTM", None)]:
        try:
            pv, _ = _fit_predict(name, cls, train_s, val_n, freq, last_dt)
            val_preds[name] = pv
            selector.add_result(name, val_s.values, pv)
        except Exception as e:
            logger.warning(f"  [{name}] val failed: {e}")

    if not selector._results:
        raise RuntimeError(f"All models failed for {state}")

    best = selector.best_model()

    # ── Final forecast on full data ──────────────────────────────────
    fc_vals, fc_idx, fm = _fit_predict(best,
        {"SARIMA":ARIMAForecaster,"Prophet":ProphetForecaster,
         "XGBoost":XGBoostForecaster,"LSTM":None}.get(best),
        full_s, val_n, freq, last_dt, full_s=full_s)

    # All-model forecasts (for /compare endpoint)
    all_fc = {}
    for name, cls in [("SARIMA",ARIMAForecaster),("Prophet",ProphetForecaster),
                       ("XGBoost",XGBoostForecaster),("LSTM",None)]:
        try:
            v, _, _ = _fit_predict(name, cls, full_s, val_n, freq, last_dt, full_s=full_s)
            all_fc[name] = v.tolist()
        except Exception:
            pass

    return dict(state=state, best_model=best, metrics=selector.to_dict(),
                forecast_dates=[str(d.date() if hasattr(d,"date") else d) for d in fc_idx],
                forecast_values=fc_vals.tolist(), all_model_forecasts=all_fc,
                val_preds={k:v.tolist() for k,v in val_preds.items()},
                val_actuals=val_s.values.tolist(),
                val_dates=[str(d.date()) for d in val_s.index],
                model_obj=fm)


def plot_state(state, train_s, val_s, val_preds, best, fc_vals, fc_idx):
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(train_s.index, train_s.values, color="steelblue", lw=2, label="Train")
    ax.plot(val_s.index,   val_s.values,   color="darkorange",lw=2, label="Val (actual)")
    for mn, pv in val_preds.items():
        ax.plot(val_s.index, pv, color=COLORS.get(mn,"grey"),
                lw=2.5 if mn==best else 1.2, ls="-" if mn==best else "--",
                label=f"{mn}{'  ★' if mn==best else ''}")
    ax.plot(fc_idx, fc_vals, "k--o", lw=2, ms=7, label="8-wk Forecast")
    ax.axvline(val_s.index[-1], color="gray", ls=":", lw=1)
    ax.set_title(f"{state} – Model Comparison & 8-Week Forecast", fontsize=13)
    ax.set_ylabel("Sales (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e6:.0f}M"))
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / f"forecast_{state.replace(' ','_')}.png", dpi=110)
    plt.close()


def run(states=None):
    logger.info("═"*60 + "\n  FULL 43-STATE FORECASTING PIPELINE\n" + "═"*60)
    raw   = load_raw_data(); clean = clean_data(raw); panel = build_monthly_panel(clean)
    train_df, val_df = train_val_split(panel)
    targets  = states or sorted(panel["state"].unique())
    logger.info(f"Processing {len(targets)} states …\n")
    results, failed = {}, []

    for state in targets:
        logger.info(f"{'─'*55}\n  ► {state}")
        try:
            train_s = get_state_series(train_df, state)
            val_s   = get_state_series(val_df,   state)
            if len(train_s) < 15:
                logger.warning(f"  Skipping – only {len(train_s)} pts"); continue
            full_s  = pd.concat([train_s, val_s]).sort_index()
            freq    = pd.infer_freq(full_s.index) or "ME"
            last_dt = full_s.index[-1]

            r = train_state(state, train_s, val_s, full_s, freq, last_dt)

            try:
                plot_state(state, train_s, val_s, r["val_preds"], r["best_model"],
                           np.array(r["forecast_values"]),
                           pd.DatetimeIndex(r["forecast_dates"]))
            except Exception as pe:
                logger.warning(f"  Plot failed: {pe}")

            try:
                joblib.dump(r["model_obj"],
                    OUTPUT/"models"/f"{state.replace(' ','_')}.pkl")
            except Exception:
                pass

            results[state] = r
            v = r["forecast_values"]
            logger.info(f"  ✔ {r['best_model']} | ${v[0]:,.0f} / ${v[1]:,.0f}")
        except Exception as e:
            logger.error(f"  ✘ {state}: {e}")
            failed.append(state)

    # ── Persist JSON ────────────────────────────────────────────────
    summary = {s:{k:v for k,v in r.items() if k not in ("model_obj","val_preds")}
               for s,r in results.items()}
    out_path = OUTPUT/"forecasts"/"all_forecasts.json"
    with open(out_path,"w") as f: json.dump(summary, f, indent=2)
    logger.info(f"\nForecasts → {out_path}")

    print("\n" + "="*74)
    print(f"{'State':<22} {'Best Model':<12} {'Month 1':>18} {'Month 2':>18}")
    print("="*74)
    for state, r in sorted(results.items()):
        v = r["forecast_values"]
        print(f"{state:<22} {r['best_model']:<12} ${v[0]:>17,.0f} ${v[1]:>17,.0f}")
    print("="*74)
    print(f"\n✅  {len(results)} succeeded | {len(failed)} failed: {failed}\n")
    return results


if __name__ == "__main__":
    run(states=sys.argv[1:] or None)
