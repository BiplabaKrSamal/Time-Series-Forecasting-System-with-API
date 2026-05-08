# Model Documentation

## Overview

Four models are trained and compared per state. The best performer (lowest MAPE on the 6-month validation set) is retrained on the full data and used for the final forecast.

---

## 1. SARIMA

**File:** `src/models/arima_model.py`  
**Library:** statsmodels 0.14 (`SARIMAX`)

### What it does

Seasonal AutoRegressive Integrated Moving Average. Captures:
- **AR**: autocorrelation — sales this month depends on past months
- **I**: differencing to remove trend/non-stationarity  
- **MA**: moving average error correction
- **Seasonal**: annual (12-month) seasonal cycle

### Order Selection

Instead of requiring manual tuning, the model runs an **AIC grid search**:

```python
# Augmented Dickey-Fuller test auto-selects d
p_value = adfuller(series)[1]
d = 0 if p_value < 0.05 else 1

# Grid search over:
# p, q ∈ {0, 1, 2}  (non-seasonal AR and MA orders)
# P, Q ∈ {0, 1}     (seasonal AR and MA orders)
# D ∈ {0, 1}        (seasonal differencing)
# m = 12            (monthly → annual seasonality)
```

The combination with the lowest AIC is selected automatically.

### When it wins

SARIMA wins on states with **stable, strongly seasonal** patterns where the
linear autoregressive structure fits well: California, New York, Pennsylvania,
Virginia, Washington, South Carolina, Wisconsin.

---

## 2. Facebook Prophet

**File:** `src/models/prophet_model.py`  
**Library:** prophet 1.1.5

### What it does

Prophet decomposes time series into:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)
```

Configuration used:
```python
Prophet(
    yearly_seasonality  = True,
    weekly_seasonality  = False,   # monthly data — irrelevant
    daily_seasonality   = False,
    changepoint_prior_scale = 0.05,  # conservative — avoids overfitting
    seasonality_mode    = "multiplicative",  # sales scale with trend
)
model.add_country_holidays(country_name="US")
```

### Strengths

- Handles **missing data** and outliers naturally
- No stationarity assumption
- Interpretable trend + seasonality components
- US holiday effects captured automatically

### Limitation in this dataset

With only 5 years (60 months) of monthly data, Prophet sometimes struggles to
confidently separate trend from seasonality changepoints, leading to higher MAPE
than simpler models on this dataset.

---

## 3. XGBoost (Recursive Walk-Forward)

**File:** `src/models/xgboost_model.py`  
**Library:** xgboost 2.0

### What it does

Gradient-boosted decision trees trained on a **supervised feature matrix**
derived from lag and calendar features. Multi-step forecasting uses a
**recursive strategy**:

```
Step 1:  append NaN placeholder at t+1
         build features from extended history
         predict → ŷ(t+1)

Step 2:  replace NaN with ŷ(t+1)
         build features from extended history
         predict → ŷ(t+2)
```

### Hyperparameters

```python
XGBRegressor(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,   # L1 regularisation
    reg_lambda       = 1.0,   # L2 regularisation
    random_state     = 42,
)
```

### Feature Importance (aggregate, top 5)

| Feature | Avg Importance |
|---|---|
| `rolling_mean_3` | 24.2% |
| `trend` | 13.8% |
| `lag_3` | 11.8% |
| `yoy_growth` | 9.7% |
| `rolling_mean_12` | 7.3% |

### Why it wins most states (84%)

XGBoost captures **non-linear interactions** between lag features, rolling
statistics, and calendar position — something SARIMA's linear structure
cannot express. For states where sales are driven by complex regional
economic patterns (e.g. Texas energy sector, Florida tourism), the
feature-rich approach outperforms.

---

## 4. LSTM (Long Short-Term Memory)

**File:** `src/models/lstm_model.py`  
**Library:** TensorFlow 2.13 / Keras

### Architecture

```
Input(shape=(12, 1))          ← 12-month lookback window
    │
    ▼
LSTM(64 units)                ← learns temporal dependencies
    │
    ▼
Dropout(0.2)                  ← regularisation
    │
    ▼
Dense(32, activation='relu')  ← non-linear projection
    │
    ▼
Dense(1)                      ← scalar output (next month sales)
```

### Training

```python
optimizer = Adam(learning_rate=0.001)
loss      = "huber"     # robust to COVID-era outliers
epochs    = 80
patience  = 15          # early stopping on internal val split
batch_size = 16
```

### Scaling

Min-max scaling is applied **using only training statistics** before fitting,
and inverted after prediction:

```python
scaled = (series - train_min) / (train_max - train_min)
# ... train and predict on scaled values ...
predictions = preds_scaled * (train_max - train_min) + train_min
```

### Multi-step prediction

The same recursive walk-forward strategy as XGBoost:
each predicted value is appended to the sliding window before predicting the next step.

### Limitation

LSTM benefits from larger datasets. With only 54 training months per state,
SARIMA and XGBoost often outperform it. On longer series (100+ months),
LSTM would likely be competitive.

---

## Model Selection Summary

| Model | States Won | Avg MAPE |
|---|---|---|
| **XGBoost** | **36 / 43 (84%)** | **52.5%** |
| SARIMA | 7 / 43 (16%) | 74.4% |

**Note on MAPE:** The high absolute MAPE values reflect the fact that beverage sales
have large month-to-month swings and the validation period (mid-2023) falls
during a post-COVID normalisation phase that diverges from the 2019–2022 trend.
The **relative ranking** of models is what matters for auto-selection.

---

## Adding a New Model

To add a 5th model (e.g. LightGBM):

1. Create `src/models/lgbm_model.py` following the same interface:
   - `.fit(series: pd.Series)`
   - `.predict(steps: int) -> pd.Series`
   - `.get_params() -> dict`

2. Add it to the training loop in `src/forecaster.py`:
   ```python
   from models.lgbm_model import LGBMForecaster
   lgbm = LGBMForecaster()
   lgbm.fit(train_series)
   lgbm_pred = lgbm.predict(val_steps)
   selector.add_result("LightGBM", val_series.values, lgbm_pred.values)
   ```

3. Add a test case in `tests/test_forecasting_system.py`.

4. The API and leaderboard update automatically.
