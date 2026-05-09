<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=26&pause=1000&color=6C63FF&center=true&width=700&lines=Time-Series+Forecasting+System;SARIMA+%C2%B7+Prophet+%C2%B7+XGBoost+%C2%B7+LSTM;43+US+States+%C2%B7+FastAPI+%C2%B7+Live+API" alt="Typing SVG" />

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

[📊 Demo](#-interactive-demo) · [🚀 Quick Start](#-quick-start) · [🔌 API](docs/API_REFERENCE.md) · [📐 Architecture](docs/ARCHITECTURE.md) · [📈 Results](docs/RESULTS.md) · [🌐 Deploy](#-deploy)

</div>

---

## ✨ Key Highlights

| | |
|---|---|
| 🗺️ **43 / 43 states** forecasted | 🌐 **Deploy-ready** — Railway · Render · Fly · Docker |
| 🤖 **4 models** compared per state | 📊 **Interactive demo** — `outputs/VIDEO_DEMO.html` |
| 🥇 **XGBoost wins 84%** of states | 📄 **Full docs** — 5 markdown files + PDF |
| 💰 **$16 B** total US Jan 2024 forecast | ✅ **24 / 24 tests** passing |

---

## 📐 Architecture

```
CSV  →  Clean  →  Monthly Panel  →  Feature Engineering  →  4 Models  →  Auto-Select  →  FastAPI
         │              │                    │                   │
      2 date       interpolate          lag_1…12             SARIMA
      formats      sparse gaps          rolling stats        Prophet
      commas       no leakage           sin/cos month        XGBoost   →  best MAPE  →  forecast
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
├── docs/                             ← Architecture · API · Models · Results · Setup
├── notebooks/
│   └── demo_walkthrough.py           ← Annotated 10-step demo
│
├── outputs/
│   ├── VIDEO_DEMO.html               ← 🎬 Self-contained interactive demo
│   ├── Documentation.pdf             ← 📄 8-page technical documentation
│   ├── forecasts/all_forecasts.json  ← All 43 state forecasts + metrics
│   └── plots/                        ← 43 forecast charts + EDA plots
│
├── run_pipeline.py                   ← ⭐ Main entry-point
├── Dockerfile                        ← Production container
├── docker-compose.yml                ← Local production run
├── railway.toml                      ← Railway deploy config
├── render.yaml                       ← Render deploy config
├── fly.toml                          ← Fly.io deploy config
└── Procfile                          ← Heroku/Railway process file
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

# 4 — Start REST API
cd api && uvicorn main:app --reload --port 8000
# → Swagger UI: http://localhost:8000/docs

# 5 — Run tests
python -m pytest tests/ -v

# 6 — Open interactive demo (no server needed)
open outputs/VIDEO_DEMO.html
```

---

## 🌐 Deploy

### Option 1 — Render (free tier)

1. Go to **[render.com/new](https://render.com/new)** → Web Service
2. Connect GitHub → select this repo
3. `render.yaml` is auto-detected
4. Live at: `https://forecasting-api.onrender.com`

### Option 2 — Fly.io

```bash
fly auth login
fly launch --config fly.toml
fly deploy
```

### Option 3 — Docker (anywhere)

```bash
docker compose up --build
# → http://localhost:8000/docs
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

| Model | Library | States Won |
|---|---|---|
| **SARIMA** | statsmodels | **7 / 43** |
| **XGBoost** | xgboost 2.0 | **36 / 43 ★** |
| Prophet | facebook/prophet | 0 / 43 |
| LSTM | TensorFlow 2.13 | 0 / 43 |

Selection: lowest MAPE on 6-month holdout → retrain on full data → forecast.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + metadata |
| `GET` | `/states` | List all 43 states |
| `GET` | `/forecast/{state}` | Best-model 8-week forecast |
| `GET` | `/forecast/{state}/compare` | All-model metrics + forecasts |
| `GET` | `/forecast/{state}/history` | Historical + forecast (chart-ready) |
| `POST` | `/forecast/batch` | Multiple states in one request |
| `GET` | `/models/leaderboard` | Global win counts |

```bash
curl https://YOUR-DEPLOYED-URL/forecast/California
curl https://YOUR-DEPLOYED-URL/models/leaderboard
curl -X POST https://YOUR-DEPLOYED-URL/forecast/batch \
     -H "Content-Type: application/json" \
     -d '{"states": ["California", "Texas", "Florida"]}'
```

Full reference: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

---

## 📈 Results

```
XGBoost  ████████████████████████████████████  36/43 states (84%)
SARIMA   ███████                                7/43 states (16%)
```

| State | Best Model | Jan 2024 | Feb 2024 |
|---|---|---|---|
| Texas | XGBoost | $1.57 B | $1.71 B |
| Florida | XGBoost | $1.46 B | $1.50 B |
| California | SARIMA | $1.08 B | $1.11 B |
| Georgia | XGBoost | $742 M | $842 M |

Full 43-state table: [`docs/RESULTS.md`](docs/RESULTS.md)

---

## 🛠️ Stack

`Python 3.11` · `statsmodels` · `prophet` · `xgboost` · `tensorflow-cpu` · `fastapi` · `gunicorn` · `uvicorn` · `pydantic v2` · `pandas` · `numpy` · `matplotlib` · `joblib` · `pytest` · `Docker`

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>SARIMA · Prophet · XGBoost · LSTM · FastAPI · 43 US States · 24 Tests · Docker · Railway · Render · Fly.io</sub>
</div>
