# %% [markdown]
# # End-to-End Time Series Forecasting — Step-by-Step Demo
# **Dataset:** US Beverages Sales by State (2019–2023)  
# **Goal:** Forecast next 8 weeks of sales per state

# %% [markdown]
# ## Step 1 — Load & Inspect Raw Data

# %%
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from preprocessing import load_raw_data, clean_data, build_monthly_panel, train_val_split, get_state_series

raw = load_raw_data()
print("Raw shape:", raw.shape)
print(raw.head())

# %% [markdown]
# ## Step 2 — Clean & Build Monthly Panel

# %%
clean = clean_data(raw)
panel = build_monthly_panel(clean)

print(f"\nPanel shape: {panel.shape}")
print(f"States: {panel['state'].nunique()}")
print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
print(panel.groupby('state')['total'].sum().sort_values(ascending=False).head(10).apply(lambda x: f"${x/1e9:.2f}B"))

# %% [markdown]
# ## Step 3 — Feature Engineering (California example)

# %%
from features import build_features, prepare_supervised

ca_series = get_state_series(panel, 'California')
print("CA series tail:\n", ca_series.tail())

features_df = build_features(ca_series)
X, y = prepare_supervised(features_df)
print(f"\nFeature matrix: {X.shape}")
print("Feature names:", X.columns.tolist())

# %% [markdown]
# ## Step 4 — Train/Validation Split (no leakage)

# %%
train_df, val_df = train_val_split(panel)

for state in ['California', 'Texas', 'New York']:
    ts = get_state_series(train_df, state)
    vs = get_state_series(val_df, state)
    print(f"{state:<20}  train={len(ts)} months  val={len(vs)} months  "
          f"last_train={ts.index[-1].date()}  first_val={vs.index[0].date()}")

# %% [markdown]
# ## Step 5 — Train All 4 Models (California)

# %%
from models.arima_model   import ARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model    import LSTMForecaster
from evaluation import ModelSelector

STATE = 'California'
train_s = get_state_series(train_df, STATE)
val_s   = get_state_series(val_df, STATE)
freq    = pd.infer_freq(train_s.index) or 'ME'
val_n   = len(val_s)

selector  = ModelSelector()
val_preds = {}

# SARIMA
print("\n[1/4] Fitting SARIMA...")
arima = ARIMAForecaster()
arima.fit(train_s)
p = arima.predict(val_n)
val_preds['SARIMA'] = p.values
selector.add_result('SARIMA', val_s.values, p.values)

# Prophet
print("\n[2/4] Fitting Prophet...")
prophet = ProphetForecaster()
prophet.fit(train_s)
p = prophet.predict(val_n)
val_preds['Prophet'] = p.values
selector.add_result('Prophet', val_s.values, p.values)

# XGBoost
print("\n[3/4] Fitting XGBoost...")
xgb = XGBoostForecaster()
xgb.fit(train_s)
p = xgb.predict(val_n)
val_preds['XGBoost'] = p.values
selector.add_result('XGBoost', val_s.values, p.values)

# LSTM
print("\n[4/4] Fitting LSTM...")
lstm = LSTMForecaster(lookback=12, epochs=50, patience=10)
lstm.fit(train_s)
p = lstm.predict_with_index(val_n, last_date=train_s.index[-1], freq=freq)
val_preds['LSTM'] = p.values
selector.add_result('LSTM', val_s.values, p.values)

# %% [markdown]
# ## Step 6 — Compare Models & Auto-Select Best

# %%
print("\n--- Validation Metrics ---")
print(selector.summary_table().to_string())
print(f"\n★ Best model: {selector.best_model()}")

# %% [markdown]
# ## Step 7 — Retrain Best on Full Data & Forecast 8 Weeks

# %%
full_s  = pd.concat([train_s, val_s]).sort_index()
last_dt = full_s.index[-1]
best    = selector.best_model()

if best == 'SARIMA':
    fm = ARIMAForecaster(); fm.fit(full_s); fc = fm.predict(2)
elif best == 'Prophet':
    fm = ProphetForecaster(); fm.fit(full_s); fc = fm.predict(2)
elif best == 'XGBoost':
    fm = XGBoostForecaster(); fm.fit(full_s); fc = fm.predict(2)
else:
    fm = LSTMForecaster(lookback=12, epochs=50, patience=10)
    fm.fit(full_s); fc = fm.predict_with_index(2, last_date=last_dt, freq=freq)

print(f"\n8-Week Forecast for {STATE} (best: {best}):")
for d, v in zip(fc.index, fc.values):
    print(f"  {str(d.date())}: ${v:,.0f}")

# %% [markdown]
# ## Step 8 — Visualize

# %%
COLORS = {'SARIMA':'#e74c3c','Prophet':'#9b59b6','XGBoost':'#2ecc71','LSTM':'#f39c12'}

fig, ax = plt.subplots(figsize=(14,5))
ax.plot(train_s.index, train_s.values, color='steelblue', lw=2, label='Train')
ax.plot(val_s.index, val_s.values, color='darkorange', lw=2, label='Validation (actual)')
for mn, pv in val_preds.items():
    ax.plot(val_s.index, pv, color=COLORS.get(mn,'grey'),
            lw=2.5 if mn==best else 1.2, ls='-' if mn==best else '--',
            label=f"{mn}{'  ★ BEST' if mn==best else ''}")
ax.plot(fc.index, np.maximum(fc.values, 0), 'k--o', lw=2, ms=7, label='8-Wk Forecast')
ax.axvline(full_s.index[-1], color='gray', ls=':', lw=1.2)
ax.set_title(f'{STATE} – All Models vs Actual + 8-Week Forecast', fontsize=13)
ax.set_ylabel('Sales (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x/1e9:.1f}B'))
ax.legend(fontsize=9, loc='upper left'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig('../outputs/plots/demo_california_final.png', dpi=130)
plt.close()
print("Plot saved → outputs/plots/demo_california_final.png")

# %% [markdown]
# ## Step 9 — XGBoost Feature Importances

# %%
imp = xgb.feature_importance()
print("\nTop 10 Feature Importances (XGBoost):")
print(imp.head(10).to_string(index=False))

# %% [markdown]
# ## Step 10 — Full 43-State Results (from saved JSON)

# %%
import json
from pathlib import Path

fc_path = Path('../outputs/forecasts/all_forecasts.json')
if fc_path.exists():
    with open(fc_path) as f:
        all_fc = json.load(f)
    from collections import Counter
    wins = Counter(v['best_model'] for v in all_fc.values())
    print(f"\nModel wins across all {len(all_fc)} states:")
    for m, w in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {m:<12}: {w:>2} states ({w/len(all_fc)*100:.0f}%)")

    print(f"\n{'State':<22} {'Best':<12} {'Jan-2024':>18} {'Feb-2024':>18}")
    print('='*72)
    for state in sorted(all_fc.keys()):
        r = all_fc[state]
        v = r['forecast_values']
        print(f"{state:<22} {r['best_model']:<12} ${v[0]:>17,.0f} ${v[1]:>17,.0f}")
else:
    print("Run run_all_states.py first to generate forecasts for all 43 states.")

print("\n✅ Demo complete!")
print("   → Open outputs/dashboard.html in a browser for the interactive dashboard")
print("   → Run: cd api && uvicorn main:app --reload  to start the REST API")
