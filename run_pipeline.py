#!/usr/bin/env python3
"""
run_pipeline.py
===============
Single entry-point for the entire forecasting pipeline.

Usage
-----
    python run_pipeline.py                        # all 43 states
    python run_pipeline.py California Texas       # specific states
    python run_pipeline.py --demo                 # 5-state quick demo
    python run_pipeline.py --eda-only             # EDA plots only

What it does
------------
1. Loads & cleans data (handles both date formats + comma-numbers)
2. Builds monthly panel (interpolates gaps, no NaN)
3. Engineers 16 features per state (lag, rolling, calendar, holiday)
4. Trains SARIMA · Prophet · XGBoost · LSTM per state
5. Evaluates on 6-month holdout (MAE, RMSE, MAPE, sMAPE)
6. Auto-selects best model (lowest MAPE)
7. Retrains winner on full data → 8-week forecast
8. Saves: outputs/forecasts/all_forecasts.json + plots + model PKLs
"""

import json, logging, os, sys, warnings, argparse
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
from features import build_features
from models.arima_model   import ARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model    import LSTMForecaster
from evaluation import ModelSelector

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "outputs"
PLOTS  = OUTPUT / "plots"
for d in [PLOTS, OUTPUT / "models", OUTPUT / "forecasts"]:
    d.mkdir(parents=True, exist_ok=True)

DEMO_STATES    = ["California", "Texas", "Florida", "New York", "Illinois"]
MODEL_COLORS   = {"SARIMA": "#4cc9f0", "Prophet": "#f72585",
                  "XGBoost": "#06d6a0", "LSTM": "#ffd60a"}

# ── EDA ──────────────────────────────────────────────────────────────────────

def run_eda(panel: pd.DataFrame):
    log.info("Generating EDA plots …")

    # 1. Top-10 states
    top10 = panel.groupby("state")["total"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    top10.plot(kind="bar", ax=ax, color="#6c63ff", edgecolor="white")
    ax.set_title("Top 10 States by Total Beverage Sales (2019–2023)", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e9:.1f}B"))
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); fig.savefig(PLOTS / "eda_top10_states.png", dpi=120); plt.close()

    # 2. Aggregate trend
    agg = panel.groupby("date")["total"].sum()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(agg.index, agg.values, color="#f97316", lw=2)
    ax.fill_between(agg.index, agg.values, alpha=0.12, color="#f97316")
    ax.set_title("Aggregate US Beverage Sales — Monthly Trend", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e9:.1f}B"))
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig.savefig(PLOTS / "eda_aggregate_trend.png", dpi=120); plt.close()

    # 3. Seasonality heatmap
    p2 = panel.copy()
    p2["year"] = p2["date"].dt.year; p2["month"] = p2["date"].dt.month
    pivot = p2.groupby(["year", "month"])["total"].sum().unstack("month").fillna(0)
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values / 1e9, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_title("Seasonality Heat-map (Total Sales $B — Year × Month)", fontsize=13)
    plt.colorbar(im, ax=ax, label="USD Billion")
    plt.tight_layout(); fig.savefig(PLOTS / "eda_seasonality_heatmap.png", dpi=120); plt.close()

    # 4. Decomposition (California)
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        ca = get_state_series(panel, "California").dropna()
        if len(ca) >= 24:
            result = seasonal_decompose(ca, model="multiplicative", period=12)
            fig = result.plot(); fig.set_size_inches(12, 8)
            fig.suptitle("California — Seasonal Decomposition", fontsize=13)
            plt.tight_layout(); fig.savefig(PLOTS / "eda_california_decomposition.png", dpi=120)
            plt.close()
    except Exception as e:
        log.warning(f"Decomposition failed: {e}")

    log.info(f"EDA plots → {PLOTS}")


# ── Per-state training ────────────────────────────────────────────────────────

def train_state(state, train_s, val_s, full_s, freq, last_dt):
    sel, val_preds = ModelSelector(), {}
    val_n = len(val_s)

    for name, cls in [("SARIMA", ARIMAForecaster), ("Prophet", ProphetForecaster),
                       ("XGBoost", XGBoostForecaster)]:
        try:
            m = cls(); m.fit(train_s)
            p = m.predict(val_n)
            pv = np.array(p.values if hasattr(p, "values") else p)
            val_preds[name] = pv
            sel.add_result(name, val_s.values, pv)
        except Exception as e:
            log.warning(f"  [{name}] failed: {e}")

    try:
        lb = min(12, len(train_s) // 2)
        m  = LSTMForecaster(lookback=lb, epochs=60, patience=12)
        m.fit(train_s)
        p = m.predict_with_index(val_n, last_date=train_s.index[-1], freq=freq)
        val_preds["LSTM"] = p.values
        sel.add_result("LSTM", val_s.values, p.values)
    except Exception as e:
        log.warning(f"  [LSTM] failed: {e}")

    if not sel._results:
        raise RuntimeError("All models failed")

    best = sel.best_model()

    # Retrain best on full data
    cls_map = {"SARIMA": ARIMAForecaster, "Prophet": ProphetForecaster,
               "XGBoost": XGBoostForecaster}
    if best == "LSTM":
        lb = min(12, len(full_s) // 2)
        fm = LSTMForecaster(lookback=lb, epochs=60, patience=12); fm.fit(full_s)
        fc = fm.predict_with_index(2, last_date=last_dt, freq=freq)
    else:
        fm = cls_map[best](); fm.fit(full_s); fc = fm.predict(2)

    fc_vals = np.maximum(np.array(fc.values if hasattr(fc, "values") else fc), 0)
    fc_idx  = fc.index if hasattr(fc, "index") else pd.date_range(last_dt, periods=3, freq=freq)[1:]

    # All-model forecasts for /compare endpoint
    all_fc = {}
    for name, cls in [("SARIMA", ARIMAForecaster), ("Prophet", ProphetForecaster),
                       ("XGBoost", XGBoostForecaster)]:
        try:
            m2 = cls(); m2.fit(full_s); v2 = m2.predict(2)
            all_fc[name] = np.maximum(np.array(v2.values if hasattr(v2, "values") else v2), 0).tolist()
        except Exception:
            pass
    try:
        lb2 = min(12, len(full_s) // 2)
        m2  = LSTMForecaster(lookback=lb2, epochs=60, patience=12); m2.fit(full_s)
        v2  = m2.predict_with_index(2, last_date=last_dt, freq=freq)
        all_fc["LSTM"] = np.maximum(v2.values, 0).tolist()
    except Exception:
        pass

    return dict(state=state, best_model=best, metrics=sel.to_dict(),
                forecast_dates=[str(d.date() if hasattr(d, "date") else d) for d in fc_idx],
                forecast_values=fc_vals.tolist(), all_model_forecasts=all_fc,
                val_preds={k: v.tolist() for k, v in val_preds.items()},
                val_actuals=val_s.values.tolist(),
                val_dates=[str(d.date()) for d in val_s.index],
                model_obj=fm)


def plot_state(state, train_s, val_s, val_preds, best, fc_vals, fc_idx):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_s.index, train_s.values, color="#6c63ff", lw=2, label="Train")
    ax.plot(val_s.index,   val_s.values,   color="#f97316", lw=2, label="Val (actual)")
    for mn, pv in val_preds.items():
        ax.plot(val_s.index, pv, color=MODEL_COLORS.get(mn, "#888"),
                lw=2.5 if mn == best else 1.2, ls="-" if mn == best else "--",
                label=f"{mn}{'  ★' if mn == best else ''}")
    ax.plot(fc_idx, fc_vals, "k--o", lw=2, ms=6, label="8-wk Forecast")
    ax.axvline(val_s.index[-1], color="gray", ls=":", lw=1)
    ax.set_title(f"{state} — Model Comparison & 8-Week Forecast", fontsize=12)
    ax.set_ylabel("Sales (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / f"forecast_{state.replace(' ', '_')}.png", dpi=110)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def run(states=None, eda_only=False):
    log.info("═" * 58)
    log.info("  TIME-SERIES FORECASTING PIPELINE")
    log.info("═" * 58)

    raw   = load_raw_data(); clean = clean_data(raw); panel = build_monthly_panel(clean)
    train_df, val_df = train_val_split(panel)

    run_eda(panel)
    if eda_only:
        log.info("EDA-only mode complete."); return {}

    targets = states or sorted(panel["state"].unique())
    log.info(f"Processing {len(targets)} states …\n")

    # Load existing results so we can resume
    cache = OUTPUT / "forecasts" / "all_forecasts.json"
    results = {}
    if cache.exists():
        with open(cache) as f:
            results = json.load(f)
        log.info(f"Loaded {len(results)} cached results — skipping those states")

    failed = []
    for state in targets:
        if state in results:
            log.info(f"  ↷ {state} (cached)")
            continue
        log.info(f"  ► {state}")
        try:
            train_s = get_state_series(train_df, state)
            val_s   = get_state_series(val_df,   state)
            if len(train_s) < 15:
                log.warning(f"  Skipping — only {len(train_s)} pts"); continue
            full_s  = pd.concat([train_s, val_s]).sort_index()
            freq    = pd.infer_freq(full_s.index) or "ME"
            last_dt = full_s.index[-1]

            r = train_state(state, train_s, val_s, full_s, freq, last_dt)

            try:
                plot_state(state, train_s, val_s, r["val_preds"], r["best_model"],
                           np.array(r["forecast_values"]), pd.DatetimeIndex(r["forecast_dates"]))
            except Exception as pe:
                log.warning(f"  Plot failed: {pe}")

            try:
                joblib.dump(r["model_obj"],
                    OUTPUT / "models" / f"{state.replace(' ', '_')}.pkl")
            except Exception:
                pass

            results[state] = {k: v for k, v in r.items() if k not in ("model_obj", "val_preds")}
            v = r["forecast_values"]
            log.info(f"  ✔ {r['best_model']:<10} | ${v[0]:,.0f} / ${v[1]:,.0f}")

            # Save after each state (resumable)
            with open(cache, "w") as f: json.dump(results, f, indent=2)

        except Exception as e:
            log.error(f"  ✘ {state}: {e}")
            failed.append(state)

    # Final summary
    print("\n" + "=" * 74)
    print(f"{'State':<22} {'Best Model':<12} {'Jan Forecast':>18} {'Feb Forecast':>18}")
    print("=" * 74)
    for state in sorted(results):
        r = results[state]; v = r["forecast_values"]
        print(f"{state:<22} {r['best_model']:<12} ${v[0]:>17,.0f} ${v[1]:>17,.0f}")
    print("=" * 74)

    from collections import Counter
    wins = Counter(r["best_model"] for r in results.values())
    print(f"\nModel wins: {dict(wins)}")
    print(f"States: {len(results)} OK · {len(failed)} failed: {failed}")
    log.info(f"\nForecasts → {cache}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-Series Forecasting Pipeline")
    parser.add_argument("states", nargs="*", help="Specific states (default: all 43)")
    parser.add_argument("--demo",     action="store_true", help="Quick 5-state demo")
    parser.add_argument("--eda-only", action="store_true", help="EDA plots only, no training")
    args = parser.parse_args()

    if args.demo:
        run(states=DEMO_STATES)
    elif args.eda_only:
        run(eda_only=True)
    else:
        run(states=args.states or None)
