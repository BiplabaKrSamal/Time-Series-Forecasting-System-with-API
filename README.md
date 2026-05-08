<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=26&pause=1000&color=6C63FF&center=true&width=700&lines=Time-Series+Forecasting+System;SARIMA+%C2%B7+Prophet+%C2%B7+XGBoost+%C2%B7+LSTM;43+US+States+%C2%B7+FastAPI+%C2%B7+Auto+Model+Selection" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189BCC?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Tests](https://img.shields.io/badge/Tests-24%20passing-22c55e?style=for-the-badge&logo=pytest)](https://pytest.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **Production-ready end-to-end time-series forecasting system.**  
> Trains SARIMA, Prophet, XGBoost and LSTM on **43 US states**, auto-selects the best model per state, and serves predictions through a 7-endpoint FastAPI REST service.

[📊 Demo](#-interactive-demo) · [🚀 Quick Start](#-quick-start) · [🔌 API](docs/API_REFERENCE.md) · [📐 Architecture](docs/ARCHITECTURE.md) · [📈 Results](docs/RESULTS.md)

</div>

---

## ✨ Key Highlights

| | |
|---|---|
| 🗺️ **43 / 43 states** forecasted | 📊 **Interactive demo** — `outputs/VIDEO_DEMO.html` |
| 🤖 **4 models** compared per state | 📄 **Full docs** — 5 markdown files + PDF |
| 🥇 **XGBoost wins 84%** of states | ✅ **24 / 24 tests** passing |
| 💰 **$16B** total US Jan 2024 forecast | ⚡ **7 FastAPI** endpoints |

---

## 📐 Architecture

```
CSV  →  Clean  →  Monthly Panel  →  Feature Engineering  →  4 Models  →  Auto-Select  →  FastAPI
         │              │                    │                   │
      2 date       interpolate          lag_1…12             SARIMA
      formats      sparse gaps          rolling stats        Prophet
      commas       no leakage           sin/cos month        XGBoost   →  best MAPE  →  retrain  →  forecast
      stripped     train/val split      holiday flag         LSTM
```

Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 📁 Project Structure

```
├── data/
│   └── sales_data.csv                ← Dataset: 43 states × 5 years
│
├── src/
│   ├── preprocessing.py              ← Clean, panel build, train/val split
│   ├── features.py                   ← Lag, rolling, calendar, holiday features
│   ├── evaluation.py                 ← MAE/RMSE/MAPE/sMAPE + ModelSelector
│   └── models/
│       ├── arima_model.py            ← SARIMA (AIC auto-order)
│       ├── prophet_model.py          ← Prophet + US holidays
│       ├── xgboost_model.py          ← XGBoost recursive walk-forward
│       └── lstm_model.py             ← LSTM + Dropout (TF/Keras)
│
├── api/
│   └── main.py                       ← FastAPI — 7 REST endpoints
│
├── tests/
│   └── test_forecasting_system.py    ← 24 unit & integration tests
│
├── docs/
│   ├── ARCHITECTURE.md               ← System design & data flow
│   ├── API_REFERENCE.md              ← All endpoints with examples
│   ├── MODELS.md                     ← Each model in depth
│   ├── RESULTS.md                    ← Full 43-state results & analysis
│   └── SETUP.md                      ← Installation & troubleshooting
│
├── notebooks/
│   └── demo_walkthrough.py           ← Annotated 10-step demo
│
├── outputs/
│   ├── VIDEO_DEMO.html               ← 🎬 Self-contained interactive demo
│   ├── Documentation.pdf             ← 📄 8-page technical documentation
│   ├── forecasts/all_forecasts.json  ← All 43 state forecasts + metrics
│   └── plots/                        ← 43 forecast charts + EDA plots
│
├── run_pipeline.py                   ← ⭐ Main training entry-point
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1 — Install
git clone https://github.com/BiplabaKrSamal/Time-Series-Forecasting-System-with-API.git
cd Time-Series-Forecasting-System-with-API
pip install -r requirements.txt

# 2 — Quick 5-state demo
python run_pipeline.py --demo

# 3 — All 43 states
python run_pipeline.py

# 4 — Specific states
python run_pipeline.py California Texas "New York"

# 5 — Start REST API
cd api && uvicorn main:app --reload --port 8000
# → Swagger UI: http://localhost:8000/docs

# 6 — Run tests
python -m pytest tests/ -v

# 7 — Open interactive demo (no server needed)
open outputs/VIDEO_DEMO.html
```

---

## 🔬 Feature Engineering

> All rolling windows use `.shift(1)` before `.rolling()` — **zero data leakage**.

| Feature | Purpose |
|---|---|
| `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` | Autoregressive memory |
| `rolling_mean_3/6/12`, `rolling_std_3/6/12` | Local trend + volatility |
| `sin_month`, `cos_month` | Circular seasonality encoding |
| `holiday_month` | US federal holiday flag |
| `yoy_growth` | Year-over-year % change |
| `month`, `quarter`, `year`, `trend` | Calendar position |

---

## 🤖 Models

| Model | Library | Selection Criterion | States Won |
|---|---|---|---|
| **SARIMA** | statsmodels | AIC grid search, ADF test | **7 / 43** |
| **Prophet** | facebook/prophet | Multiplicative seasonality + US holidays | 0 / 43 |
| **XGBoost** | xgboost 2.0 | Recursive walk-forward, 16 lag features | **36 / 43 ★** |
| **LSTM** | TensorFlow 2.13 | LSTM(64)→Dropout→Dense, Huber loss | 0 / 43 |

Selection logic: lowest MAPE on 6-month holdout → retrain on full data → forecast.

Full model docs: [`docs/MODELS.md`](docs/MODELS.md)

---

## 🔌 API Endpoints

```bash
GET  /                         # health check
GET  /states                   # list all 43 states
GET  /forecast/{state}         # best-model 8-week forecast
GET  /forecast/{state}/compare # all-model metrics + forecasts
GET  /forecast/{state}/history # historical + forecast (chart-ready)
POST /forecast/batch           # batch — multiple states
GET  /models/leaderboard       # global win counts
```

```bash
# Examples
curl http://localhost:8000/forecast/California
curl http://localhost:8000/models/leaderboard
curl -X POST http://localhost:8000/forecast/batch \
     -H "Content-Type: application/json" \
     -d '{"states": ["California", "Texas", "Florida"]}'
```

Full reference: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

---

## 📈 Results

```
XGBoost  ████████████████████████████████████  36 / 43 states (84%)
SARIMA   ███████                                7 / 43 states (16%)
```

Top forecasts (January 2024):

| State | Best Model | Forecast |
|---|---|---|
| Texas | XGBoost | $1.57 B |
| Florida | XGBoost | $1.46 B |
| California | SARIMA | $1.08 B |
| Georgia | XGBoost | $742 M |

Full 43-state table: [`docs/RESULTS.md`](docs/RESULTS.md)

---

## 📄 Documentation

| File | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System layers, data flow, dependency graph |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | All 7 endpoints with request/response schemas |
| [`docs/MODELS.md`](docs/MODELS.md) | Theory, config and win conditions for each model |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Full 43-state table, EDA findings, limitations |
| [`docs/SETUP.md`](docs/SETUP.md) | Installation, environment, troubleshooting |
| [`outputs/Documentation.pdf`](outputs/Documentation.pdf) | 8-page printable technical document |

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
# 24 passed in ~50s
```

| Class | Tests |
|---|---|
| `TestPreprocessing` | Date parsing, NaN handling, no-leakage split |
| `TestFeatures` | Shape, zero NaN output, sequence construction |
| `TestModelInterfaces` | fit/predict contract for all 4 models |
| `TestEvaluation` | Metrics, selector logic |
| `TestAPI` | All 7 endpoints, 404, batch |

---

## 🛠️ Stack

`Python 3.10+` · `statsmodels` · `prophet` · `xgboost` · `tensorflow` · `fastapi` · `uvicorn` · `pydantic v2` · `pandas` · `numpy` · `matplotlib` · `joblib` · `pytest`

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>SARIMA · Prophet · XGBoost · LSTM · FastAPI · 43 US States · 24 Tests Passing</sub>
</div>
