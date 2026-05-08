<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&pause=1000&color=6C63FF&center=true&vMultiline=false&width=700&lines=State+Sales+Forecasting+System;SARIMA+%C2%B7+Prophet+%C2%B7+XGBoost+%C2%B7+LSTM;43+US+States+%C2%B7+FastAPI+%C2%B7+Auto+Model+Selection" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189BCC?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Tests](https://img.shields.io/badge/Tests-24%20passing-22c55e?style=for-the-badge&logo=pytest)](https://pytest.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **Production-ready, end-to-end time series forecasting system.**
> Trains SARIMA, Prophet, XGBoost, and LSTM on **43 US states**, auto-selects
> the best model per state based on validation MAPE, and serves live predictions
> through a 7-endpoint FastAPI service — all backed by a self-contained
> interactive dashboard.

<br/>

[📊 Dashboard](#-interactive-dashboard) · [🚀 Quick Start](#-quick-start) · [🔌 API Docs](docs/API_REFERENCE.md) · [📐 Architecture](docs/ARCHITECTURE.md) · [📈 Results](docs/RESULTS.md)

</div>

---

## ✨ Highlights

<table>
<tr>
<td width="50%">

**🏗️ Production Engineering**
- Clean layered architecture (5 layers)
- 24 automated unit & integration tests
- GitHub Actions CI pipeline
- Pydantic v2 request validation
- Joblib model serialisation
- Strict no-leakage time-series split

</td>
<td width="50%">

**🤖 Advanced ML**
- 4 models × 43 states = 172 trained models
- Auto order selection (SARIMA AIC grid)
- Recursive walk-forward multi-step forecasting
- Cyclical feature encoding (sin/cos month)
- US holiday flag engineering
- Early stopping + Huber loss for LSTM

</td>
</tr>
<tr>
<td width="50%">

**🔌 Full-Stack API**
- FastAPI + Uvicorn + CORS
- 7 REST endpoints
- Swagger UI at `/docs`
- Batch forecasting endpoint
- Global model leaderboard
- History + forecast for charting

</td>
<td width="50%">

**📊 Visualisation**
- Self-contained interactive HTML dashboard
- 43 state forecast comparison charts
- EDA: trend, seasonality heatmap, decomposition
- Feature importance analysis
- Model comparison bar charts (Chart.js)

</td>
</tr>
</table>

---

## 🏆 Results at a Glance

<div align="center">

| Metric | Value |
|--------|-------|
| 🗺️ States Forecasted | **43 / 43** |
| 🤖 Models Compared Per State | **4** |
| 🥇 Winning Model | **XGBoost** — 36 states (84%) |
| 🥈 Runner-up | **SARIMA** — 7 states (16%) |
| 💰 Total US January 2024 Forecast | **$16.01 Billion** |
| ✅ Tests | **24 / 24 passing** |
| ⚡ API Endpoints | **7** |

</div>

**Model win distribution:**
```
XGBoost  ████████████████████████████████████  36/43 states (84%)
SARIMA   ███████                                7/43 states (16%)
```

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — INGESTION                                                         │
│                                                                              │
│  CSV  →  Date Parser  →  Sales Cleaner  →  Monthly Panel  →  Train/Val      │
│          (2 formats)     (commas/spaces)   (interpolate)     (no leakage)    │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────────┐
│  LAYER 2 — FEATURE ENGINEERING                                               │
│                                                                              │
│  Lag(1,2,3,6,12) · Rolling Mean/Std(3,6,12m) · Calendar · Holiday · YoY    │
│  All rolling windows use .shift(1) before .rolling() — ZERO DATA LEAKAGE   │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
         ┌───────────┬─────────────┼─────────────┬───────────┐
         ▼           ▼             ▼             ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │  SARIMA  │ │ Prophet  │ │ XGBoost  │ │   LSTM   │
   │ Auto AIC │ │ US hols  │ │ Recurs.  │ │ LSTM(64) │
   │ ADF test │ │ Mult.    │ │ walk-fwd │ │ Dropout  │
   │ m=12     │ │ season.  │ │ lag feat │ │ Huber    │
   └──────────┘ └──────────┘ └──────────┘ └──────────┘
         └───────────┴─────────────┼─────────────┴───────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────────┐
│  LAYER 4 — EVALUATION & MODEL SELECTION                                      │
│                                                                              │
│  MAE · RMSE · MAPE · sMAPE  →  argmin(MAPE)  →  Retrain on full data       │
│  →  Forecast 2 months ahead (= ~8 weeks)                                    │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────────────┐
│  LAYER 5 — SERVING                                                           │
│                                                                              │
│  FastAPI  ·  7 REST Endpoints  ·  Swagger UI  ·  Pydantic v2 validation     │
└──────────────────────────────────────────────────────────────────────────────┘
```

📖 Full architecture details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 🚀 Quick Start

### 1 — Install

```bash
git clone https://github.com/BiplabaKrSamal/Time-Series-Forecasting-System-with-API.git
cd Time-Series-Forecasting-System-with-API
pip install -r requirements.txt
```

### 2 — Run the step-by-step demo

```bash
python notebooks/demo_walkthrough.py
```

### 3 — Train all 43 states

```bash
python run_all_states.py
# Or specific states:
python run_all_states.py California Texas "New York"
```

### 4 — Start the REST API

```bash
cd api
uvicorn main:app --reload --port 8000
# Open: http://localhost:8000/docs
```

### 5 — Open the interactive dashboard

```bash
open outputs/dashboard.html      # macOS
xdg-open outputs/dashboard.html  # Linux
```

---

## 📁 Project Structure

```
Time-Series-Forecasting-System-with-API/
│
├── 📂 data/
│   └── sales_data.csv                ← Raw dataset: 43 states × 5 years
│
├── 📂 src/
│   ├── preprocessing.py             ← Load, clean, monthly panel, train/val split
│   ├── features.py                  ← All feature engineering (lag, rolling, calendar)
│   ├── evaluation.py                ← Metrics + ModelSelector auto-selection
│   ├── forecaster.py                ← Full 43-state pipeline orchestrator
│   └── models/
│       ├── arima_model.py           ← SARIMA with AIC grid-search auto-order
│       ├── prophet_model.py         ← Facebook Prophet + US holidays
│       ├── xgboost_model.py         ← XGBoost recursive walk-forward
│       └── lstm_model.py            ← LSTM + Dropout (TensorFlow/Keras)
│
├── 📂 api/
│   └── main.py                      ← FastAPI — 7 REST endpoints
│
├── 📂 docs/
│   ├── ARCHITECTURE.md              ← System design & data flow
│   ├── API_REFERENCE.md             ← All endpoint docs + examples
│   ├── MODELS.md                    ← Each model explained in depth
│   ├── RESULTS.md                   ← Full 43-state results + analysis
│   └── SETUP.md                     ← Installation & troubleshooting
│
├── 📂 notebooks/
│   └── demo_walkthrough.py          ← 10-step annotated demo
│
├── 📂 tests/
│   └── test_forecasting_system.py   ← 24 unit & integration tests
│
├── 📂 outputs/
│   ├── dashboard.html               ← 🌐 Self-contained interactive dashboard
│   ├── forecasts/all_forecasts.json ← Forecasts for all 43 states
│   ├── models/                      ← Serialised model objects (.pkl)
│   └── plots/                       ← 43 state charts + 4 EDA plots
│
├── run_all_states.py                ← Production training script
├── train_and_visualize.py           ← EDA + visualisation script
├── requirements.txt
├── .github/workflows/ci.yml         ← GitHub Actions CI
└── README.md
```

---

## 🔬 Feature Engineering

> All rolling windows use `.shift(1)` before `.rolling()` — **zero data leakage**.

| Feature | Code | Purpose |
|---|---|---|
| Lag 1–12 months | `lag_1` … `lag_12` | Autoregressive memory |
| Rolling mean | `rolling_mean_3/6/12` | Local trend smoothing |
| Rolling std | `rolling_std_3/6/12` | Volatility encoding |
| Month, quarter, year | `month`, `quarter`, `year` | Seasonal position |
| Cyclical encoding | `sin_month`, `cos_month` | Circular month (no boundary) |
| Holiday flag | `holiday_month` | US federal holiday month |
| YoY growth | `yoy_growth` | Year-over-year % change |
| Linear trend | `trend` | Long-term index |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/states` | List all 43 states |
| `GET` | `/forecast/{state}` | Best-model forecast |
| `GET` | `/forecast/{state}/compare` | All-model comparison |
| `GET` | `/forecast/{state}/history` | Historical + forecast |
| `POST` | `/forecast/batch` | Multi-state batch |
| `GET` | `/models/leaderboard` | Win counts |

```bash
# Examples
curl http://localhost:8000/forecast/California
curl http://localhost:8000/forecast/Texas/compare
curl -X POST http://localhost:8000/forecast/batch \
     -H "Content-Type: application/json" \
     -d '{"states": ["California", "Texas", "Florida"]}'
```

📖 Full API docs with request/response schemas: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

```
tests/test_forecasting_system.py::TestPreprocessing::test_clean_total_strips_commas  PASSED
tests/test_forecasting_system.py::TestPreprocessing::test_parse_date_slash           PASSED
tests/test_forecasting_system.py::TestPreprocessing::test_parse_date_hyphen          PASSED
tests/test_forecasting_system.py::TestPreprocessing::test_panel_no_nans             PASSED
tests/test_forecasting_system.py::TestPreprocessing::test_panel_sorted              PASSED
tests/test_forecasting_system.py::TestPreprocessing::test_train_val_split_no_leakage PASSED
tests/test_forecasting_system.py::TestFeatures::test_build_features_shape           PASSED
tests/test_forecasting_system.py::TestFeatures::test_prepare_supervised_no_nan      PASSED
tests/test_forecasting_system.py::TestFeatures::test_sequences_shape                PASSED
tests/test_forecasting_system.py::TestModelInterfaces::test_sarima_fit_predict      PASSED
tests/test_forecasting_system.py::TestModelInterfaces::test_prophet_fit_predict     PASSED
tests/test_forecasting_system.py::TestModelInterfaces::test_xgboost_fit_predict     PASSED
tests/test_forecasting_system.py::TestModelInterfaces::test_lstm_fit_predict        PASSED
tests/test_forecasting_system.py::TestModelInterfaces::test_model_raises_without_fit PASSED
tests/test_forecasting_system.py::TestEvaluation::test_mape_zero_actuals            PASSED
tests/test_forecasting_system.py::TestEvaluation::test_metrics_dict_keys            PASSED
tests/test_forecasting_system.py::TestEvaluation::test_selector_best_model          PASSED
tests/test_forecasting_system.py::TestAPI::test_root_returns_ok                     PASSED
tests/test_forecasting_system.py::TestAPI::test_health                              PASSED
tests/test_forecasting_system.py::TestAPI::test_states_list                         PASSED
tests/test_forecasting_system.py::TestAPI::test_forecast_california                 PASSED
tests/test_forecasting_system.py::TestAPI::test_forecast_unknown_state              PASSED
tests/test_forecasting_system.py::TestAPI::test_batch_forecast                      PASSED
tests/test_forecasting_system.py::TestAPI::test_leaderboard                         PASSED

24 passed in 47.23s
```

---

## 📊 Interactive Dashboard

Open `outputs/dashboard.html` in any browser — **no server, no install required**.

Features:
- 🗺️ All 43 states with model-tag labels and search filter
- 📈 Historical sales + forecast chart (last 24 months)
- 🏁 Validation metrics table with best-model highlight
- 📊 All-model forecast comparison bar chart
- 🏆 Global model leaderboard with win-rate bars
- 📱 Responsive design

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Time Series | statsmodels 0.14 (SARIMA), prophet 1.1.5 |
| Machine Learning | XGBoost 2.0 |
| Deep Learning | TensorFlow 2.13 / Keras |
| Feature Engineering | pandas, numpy, holidays |
| API | FastAPI 0.104 + Uvicorn + Pydantic v2 |
| Visualisation | matplotlib, Chart.js |
| Testing | pytest, httpx |
| CI/CD | GitHub Actions |
| Serialisation | joblib |

---

## 📖 Documentation

| Document | Description |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, data flow, dependency graph |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | All 7 endpoints with request/response examples |
| [`docs/MODELS.md`](docs/MODELS.md) | Each model explained: theory, config, when it wins |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Full 43-state results, EDA findings, limitations |
| [`docs/SETUP.md`](docs/SETUP.md) | Installation, running, troubleshooting |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built as a production-grade**

*SARIMA · Prophet · XGBoost · LSTM · FastAPI · 43 US States · 24 Tests*

</div>
