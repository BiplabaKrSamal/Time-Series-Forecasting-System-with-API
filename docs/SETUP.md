# Setup & Installation Guide

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| pip | 23+ | `pip install --upgrade pip` |
| RAM | 8 GB+ | LSTM training uses ~2 GB |
| Disk | 2 GB | Models + plots |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BiplabaKrSamal/Time-Series-Forecasting-System-with-API.git
cd Time-Series-Forecasting-System-with-API
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Core packages installed:**

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical operations |
| statsmodels | 0.14+ | SARIMA model |
| prophet | 1.1.5+ | Facebook Prophet |
| xgboost | 2.0+ | Gradient boosting |
| tensorflow | 2.13+ | LSTM deep learning |
| fastapi | 0.104+ | REST API framework |
| uvicorn | 0.24+ | ASGI server |
| holidays | 0.40+ | US holiday calendar |
| matplotlib | 3.7+ | Visualisation |
| joblib | 1.3+ | Model serialisation |
| pytest | 7.4+ | Testing |

---

## Running the System

### Step 1 — Run the interactive demo (recommended first step)

This walks through every pipeline stage with California as the example:

```bash
python notebooks/demo_walkthrough.py
```

Expected output:
```
Raw shape: (8084, 4)
Panel shape: (2580, 3)   ← 43 states × 60 months
States: 43
Date range: 2019-01-31 → 2023-12-31

[1/4] Fitting SARIMA...
[2/4] Fitting Prophet...
[3/4] Fitting XGBoost...
[4/4] Fitting LSTM...

★ Best model: XGBoost

8-Week Forecast for California (best: XGBoost):
  2024-01-31: $1,081,889,233
  2024-02-29: $1,113,320,813
```

### Step 2 — Train all 43 states

```bash
python run_all_states.py
```

For specific states:
```bash
python run_all_states.py California Texas "New York" Florida
```

Training time (approximate):
- Per state (3 models, no LSTM): ~30–60 seconds
- All 43 states: ~30–45 minutes
- With LSTM: ~2× longer

### Step 3 — Start the API

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open the Swagger UI: **http://localhost:8000/docs**

### Step 4 — Open the dashboard

```bash
# macOS
open outputs/dashboard.html

# Linux
xdg-open outputs/dashboard.html

# Windows
start outputs/dashboard.html
```

No server required — the dashboard is fully self-contained HTML.

---

## Running Tests

```bash
# All 24 tests
python -m pytest tests/ -v

# Specific test classes
python -m pytest tests/ -v -k "TestPreprocessing"
python -m pytest tests/ -v -k "TestAPI"
python -m pytest tests/ -v -k "TestModels"

# With coverage
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Project Outputs

After running the full pipeline, `outputs/` contains:

```
outputs/
├── dashboard.html              ← Self-contained interactive dashboard
├── forecasts/
│   └── all_forecasts.json     ← Forecasts + metrics for all 43 states
├── models/
│   ├── California.pkl
│   ├── Texas.pkl
│   └── ...                    ← Serialised best models
└── plots/
    ├── eda_top10_states.png
    ├── eda_aggregate_trend.png
    ├── eda_seasonality_heatmap.png
    ├── forecast_California.png
    ├── forecast_Texas.png
    └── ...                    ← 43 state forecast charts
```

---

## Troubleshooting

### Prophet installation fails

```bash
# On some systems, Prophet needs pystan first:
pip install pystan==2.19.1.1
pip install prophet
```

### TensorFlow not found

```bash
# CPU-only (smaller, no GPU needed):
pip install tensorflow-cpu>=2.13
```

### "Forecasts not yet loaded" (API 503 error)

You need to run the training pipeline before starting the API:
```bash
python run_all_states.py
cd api && uvicorn main:app --reload
```

### `asfreq("ME")` error on older pandas

The `"ME"` (month-end) frequency alias requires pandas >= 2.2.
Upgrade with: `pip install --upgrade pandas`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TF_CPP_MIN_LOG_LEVEL` | `3` | Suppress TensorFlow C++ logs |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Disable oneDNN (removes warning noise) |

Set these in your shell or `.env` file if needed.
