# System Architecture

## High-Level Overview

The system is structured in five clean layers — ingestion, features, models, evaluation, and serving.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  INGESTION                                                               │
│  CSV → Date Parser → Sales Cleaner → Monthly Panel → Train/Val Split    │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│  FEATURES                                                                │
│  Lag(1,2,3,6,12) · Rolling Mean/Std(3,6,12m) · Calendar · Holiday · YoY │
│  ⚠ All rolling windows shift(1) before rolling() — ZERO DATA LEAKAGE    │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
         ┌────────────┬───────────┴───────────┬────────────┐
         ▼            ▼                       ▼            ▼
      SARIMA       PROPHET               XGBOOST         LSTM
   AIC grid     Multiplicative        Recursive       LSTM(64)
   ADF test     US holidays           walk-fwd        Dropout
   Seasonal     Changepoints          Lag feats       Huber loss
         └────────────┴───────────┬───────────┴────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│  EVALUATION                                                              │
│  MAE · RMSE · MAPE · sMAPE   →   ModelSelector.best_model()             │
│  Retrain winner on full data  →   Forecast 2 months (≈ 8 weeks)         │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│  SERVING                                                                 │
│  FastAPI · 7 REST endpoints · Swagger UI · CORS · Pydantic v2           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
sales_data.csv  (8,084 rows × 4 columns)
       │
       ▼  clean_data()
       •  Parse dates: M/D/YYYY  and  DD-MM-YYYY  both handled
       •  Clean sales: "109,574,036"  →  109574036.0
       •  Drop duplicate (state, date) rows
       │
       ▼  build_monthly_panel()
       •  Snap all dates to month-end
       •  Reindex to full 43-state × 60-month grid
       •  Interpolate gaps linearly within each state
       •  Result: 2,580 rows, zero NaNs
       │
       ▼  train_val_split(val_periods=6)
       •  Last 6 months per state  →  validation  (258 rows)
       •  Everything before        →  training    (2,322 rows)
       •  ZERO leakage: strictly time-ordered, no shuffle
       │
       ▼  [per state, independently]
       •  Train all 4 models on training series
       •  Predict 6 steps ahead (validation)
       •  Compute MAE, RMSE, MAPE, sMAPE
       •  Auto-select best (lowest MAPE)
       •  Retrain best on full series (train + val)
       •  Forecast 2 months ahead  →  saved to JSON + .pkl
```

## Multi-Step Forecasting Strategy

XGBoost and LSTM both use **recursive (one-step-ahead) walk-forward**:

```
Step 1:  history = [t-12 … t]           →  predict t+1
Step 2:  history = [t-12 … t, t+1_hat]  →  predict t+2
```

This is preferred over the direct strategy for a 2-step horizon because error
compounding is negligible, and it requires only a single trained model per state.

## Module Dependency Graph

```
preprocessing.py
    │
    └── features.py
            │
            └── models/
                    ├── arima_model.py    (statsmodels SARIMAX)
                    ├── prophet_model.py  (facebook/prophet)
                    ├── xgboost_model.py  (xgboost 2.0)
                    └── lstm_model.py     (TensorFlow 2.13)
                            │
                            └── evaluation.py  (metrics + ModelSelector)
                                    │
                                    └── forecaster.py  (43-state orchestrator)
                                            │
                                            └── api/main.py  (FastAPI)
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Monthly granularity | Raw data is irregular; snapping to month-end + interpolation gives clean, regular series |
| Per-state models | Each state has unique seasonal pattern and economic cycle |
| MAPE as primary metric | Scale-invariant; fair comparison across states of vastly different sizes |
| 6-month validation | Captures one seasonal half-cycle without over-shrinking training data |
| Huber loss for LSTM | Robust to COVID-era outliers (2020 Q2 shock) |
| Recursive forecasting | Single model, no horizon-specific complexity, negligible error compounding at 2 steps |
