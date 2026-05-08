"""
train_and_visualize.py
======================
End-to-end training script that:
1. Loads and preprocesses data
2. Generates EDA charts
3. Runs the full forecasting pipeline (subset of states for speed)
4. Saves results, plots, and a summary report
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

from preprocessing import (
    load_raw_data, clean_data, build_monthly_panel,
    train_val_split, get_state_series
)
from features import build_features, prepare_supervised
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
PLOTS.mkdir(parents=True, exist_ok=True)


# =========================================================================
# SECTION 1 – EDA
# =========================================================================

def eda(panel: pd.DataFrame):
    logger.info("Generating EDA plots …")

    # ---- 1a. Top 10 states total sales ----
    total_by_state = (panel.groupby("state")["total"].sum()
                           .sort_values(ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(12, 5))
    total_by_state.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Top 10 States by Total Beverages Sales (All Time)", fontsize=14)
    ax.set_xlabel("State")
    ax.set_ylabel("Total Sales (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e9:.1f}B"))
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(PLOTS / "eda_top10_states.png", dpi=120)
    plt.close()

    # ---- 1b. Aggregate monthly trend ----
    monthly_agg = panel.groupby("date")["total"].sum()
    fig, ax = plt.subplots(figsize=(14, 4))
    monthly_agg.plot(ax=ax, color="darkorange", linewidth=2)
    ax.fill_between(monthly_agg.index, monthly_agg.values, alpha=0.15, color="darkorange")
    ax.set_title("Aggregate US Beverages Sales – Monthly Trend", fontsize=14)
    ax.set_ylabel("Total Sales (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e9:.1f}B"))
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(PLOTS / "eda_aggregate_trend.png", dpi=120)
    plt.close()

    # ---- 1c. Seasonality heat-map (avg by month × year) ----
    panel2 = panel.copy()
    panel2["year"]  = panel2["date"].dt.year
    panel2["month"] = panel2["date"].dt.month
    pivot = (panel2.groupby(["year", "month"])["total"]
                   .sum()
                   .unstack("month")
                   .fillna(0))
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values / 1e9, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Seasonality Heat-map (Total Sales $B by Year × Month)", fontsize=13)
    plt.colorbar(im, ax=ax, label="USD Billion")
    plt.tight_layout()
    fig.savefig(PLOTS / "eda_seasonality_heatmap.png", dpi=120)
    plt.close()

    # ---- 1d. Sample state decomposition (California) ----
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        ca = get_state_series(panel, "California").dropna()
        if len(ca) >= 24:
            result = seasonal_decompose(ca, model="multiplicative", period=12)
            fig = result.plot()
            fig.set_size_inches(12, 8)
            fig.suptitle("California Sales – Seasonal Decomposition", fontsize=14)
            plt.tight_layout()
            fig.savefig(PLOTS / "eda_california_decomposition.png", dpi=120)
            plt.close()
    except Exception as e:
        logger.warning(f"Decomposition plot failed: {e}")

    logger.info(f"EDA plots saved to {PLOTS}")


# =========================================================================
# SECTION 2 – Feature engineering visualisation
# =========================================================================

def feature_vis(panel: pd.DataFrame):
    state = "California"
    series = get_state_series(panel, state)
    feats  = build_features(series)
    X, y   = prepare_supervised(feats)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Feature Engineering – {state}", fontsize=14)

    # Lag scatter
    for i, lag in enumerate([1, 2, 3, 6]):
        ax = axes[i // 2][i % 2]
        col = f"lag_{lag}"
        if col in X.columns:
            ax.scatter(X[col], y, alpha=0.5, s=20, color="steelblue")
            ax.set_xlabel(f"Lag {lag} (t-{lag})")
            ax.set_ylabel("Sales (t)")
            ax.set_title(f"Sales vs Lag-{lag}")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))

    plt.tight_layout()
    fig.savefig(PLOTS / "feature_lag_scatter.png", dpi=120)
    plt.close()

    logger.info("Feature visualisation saved.")


# =========================================================================
# SECTION 3 – Train models on a representative subset
# =========================================================================

DEMO_STATES = ["California", "Texas", "Florida", "New York", "Illinois"]


def train_demo_states(panel: pd.DataFrame):
    train_df, val_df = train_val_split(panel)
    results = {}

    for state in DEMO_STATES:
        logger.info(f"\n{'─'*50}")
        logger.info(f" ► {state}")
        try:
            train_s = get_state_series(train_df, state)
            val_s   = get_state_series(val_df,   state)
            full_s  = pd.concat([train_s, val_s]).sort_index()
            freq    = pd.infer_freq(full_s.index) or "MS"
            last_dt = full_s.index[-1]
            val_n   = len(val_s)

            selector = ModelSelector()
            preds    = {}

            # SARIMA
            try:
                m = ARIMAForecaster(); m.fit(train_s)
                p = m.predict(val_n)
                preds["SARIMA"] = p.values
                selector.add_result("SARIMA", val_s.values, p.values)
            except Exception as e:
                logger.warning(f"  SARIMA failed: {e}")

            # Prophet
            try:
                m = ProphetForecaster(); m.fit(train_s)
                p = m.predict(val_n)
                preds["Prophet"] = p.values
                selector.add_result("Prophet", val_s.values, p.values)
            except Exception as e:
                logger.warning(f"  Prophet failed: {e}")

            # XGBoost
            try:
                m = XGBoostForecaster(); m.fit(train_s)
                p = m.predict(val_n)
                preds["XGBoost"] = p.values
                selector.add_result("XGBoost", val_s.values, p.values)
            except Exception as e:
                logger.warning(f"  XGBoost failed: {e}")

            # LSTM
            try:
                lookback = min(12, len(train_s) // 2)
                m = LSTMForecaster(lookback=lookback, epochs=50, patience=10)
                m.fit(train_s)
                p = m.predict_with_index(val_n, last_date=train_s.index[-1], freq=freq)
                preds["LSTM"] = p.values
                selector.add_result("LSTM", val_s.values, p.values)
            except Exception as e:
                logger.warning(f"  LSTM failed: {e}")

            best = selector.best_model()

            # Final forecast (2 months = ~8 weeks)
            final_model = {
                "SARIMA":  ARIMAForecaster,
                "Prophet": ProphetForecaster,
                "XGBoost": XGBoostForecaster,
            }.get(best, XGBoostForecaster)()

            if best == "LSTM":
                lb = min(12, len(full_s) // 2)
                final_model = LSTMForecaster(lookback=lb, epochs=50, patience=10)

            final_model.fit(full_s)
            if best == "LSTM":
                final_fc = final_model.predict_with_index(2, last_date=last_dt, freq=freq)
            elif best in ("SARIMA",):
                final_fc = final_model.predict(2)
            else:
                final_fc = final_model.predict(2)

            results[state] = {
                "best_model":      best,
                "metrics":         selector.to_dict(),
                "forecast_dates":  [str(d.date() if hasattr(d, "date") else d)
                                    for d in final_fc.index],
                "forecast_values": np.maximum(final_fc.values, 0).tolist(),
                "val_preds":       {k: v.tolist() for k, v in preds.items()},
                "val_dates":       [str(d.date()) for d in val_s.index],
                "val_actuals":     val_s.values.tolist(),
            }

            _plot_state_forecast(state, train_s, val_s, preds,
                                  best, final_fc, full_s)

        except Exception as e:
            logger.error(f"  {state} FAILED: {e}", exc_info=True)

    # Save JSON
    json_path = OUTPUT / "forecasts" / "demo_forecasts.json"
    json_path.parent.mkdir(exist_ok=True)
    safe_results = {
        s: {k: v for k, v in r.items() if k != "model"}
        for s, r in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(safe_results, f, indent=2)

    logger.info(f"\nDemo results saved → {json_path}")
    return results


def _plot_state_forecast(state, train_s, val_s, val_preds,
                          best_name, final_fc, full_s):
    """Produce a clean forecast chart for one state."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical
    ax.plot(train_s.index, train_s.values, color="steelblue",
            linewidth=2, label="Train")
    ax.plot(val_s.index, val_s.values, color="darkorange",
            linewidth=2, label="Validation (actual)")

    # Model predictions on validation
    colors = {"SARIMA": "#e74c3c", "Prophet": "#9b59b6",
              "XGBoost": "#2ecc71", "LSTM": "#f39c12"}
    for mname, pvals in val_preds.items():
        lw = 2.5 if mname == best_name else 1.2
        ls = "-"  if mname == best_name else "--"
        ax.plot(val_s.index, pvals, color=colors.get(mname, "grey"),
                linewidth=lw, linestyle=ls,
                label=f"{mname}{'  ★ BEST' if mname == best_name else ''}")

    # Future forecast
    fc_dates = pd.DatetimeIndex(final_fc.index)
    ax.plot(fc_dates, final_fc.values, "k--o", linewidth=2,
            markersize=7, label="8-Week Forecast (best model)")
    ax.axvline(full_s.index[-1], color="gray", linestyle=":", linewidth=1.5,
               label="Forecast start")

    ax.set_title(f"{state} – Forecast Comparison", fontsize=14)
    ax.set_ylabel("Sales (USD)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M")
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / f"forecast_{state.replace(' ', '_')}.png", dpi=130)
    plt.close()


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  Sales Forecasting – Training & Visualization   ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    raw   = load_raw_data()
    clean = clean_data(raw)
    panel = build_monthly_panel(clean)

    eda(panel)
    feature_vis(panel)
    results = train_demo_states(panel)

    # Print final summary table
    print("\n" + "=" * 70)
    print(f"{'State':<20} {'Best Model':<12} {'Month 1 Forecast':>20} {'Month 2 Forecast':>20}")
    print("=" * 70)
    for state, r in results.items():
        fv = r["forecast_values"]
        m1 = f"${fv[0]:,.0f}" if len(fv) > 0 else "N/A"
        m2 = f"${fv[1]:,.0f}" if len(fv) > 1 else "N/A"
        print(f"{state:<20} {r['best_model']:<12} {m1:>20} {m2:>20}")
    print("=" * 70)

    logger.info(f"\nAll outputs saved to: {OUTPUT}")
    logger.info("To start the API:  cd api && uvicorn main:app --reload --port 8000")
