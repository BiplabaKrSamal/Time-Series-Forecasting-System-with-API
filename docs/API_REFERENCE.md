# REST API Reference

## Overview

The forecasting system exposes a production-ready REST API built with **FastAPI**.
All endpoints return JSON. Full interactive documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

**Base URL (local):** `http://localhost:8000`

---

## Quick Start

```bash
# Start the server
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Test it
curl http://localhost:8000/
curl http://localhost:8000/states
curl http://localhost:8000/forecast/California
```

---

## Endpoints

### `GET /`

Health check. Returns service metadata.

**Response:**
```json
{
  "status": "ok",
  "service": "State Sales Forecasting API",
  "version": "1.0.0",
  "states_loaded": 43,
  "docs": "/docs"
}
```

---

### `GET /health`

Liveness probe for container orchestration (Kubernetes, ECS, etc.).

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:23:45.123456"
}
```

---

### `GET /states`

Returns the sorted list of all states for which forecasts are available.

**Response:** `["Alabama", "Arizona", "Arkansas", ..., "Wyoming"]`

---

### `GET /forecast/{state}`

Returns the **best-model** 8-week sales forecast for a single state.

**Path parameter:** `state` — US state name (case-insensitive)

**Response:**
```json
{
  "state": "California",
  "best_model": "SARIMA",
  "forecast_steps": 2,
  "unit": "USD (monthly sales)",
  "forecast": [
    { "date": "2024-01-31", "value": 1081889233.46 },
    { "date": "2024-02-29", "value": 1113320813.50 }
  ],
  "generated_at": "2024-12-01T10:23:45.123456"
}
```

**Error (404):**
```json
{
  "detail": "State 'XYZ' not found. Use GET /states to see available states."
}
```

---

### `GET /forecast/{state}/compare`

Returns validation metrics and forecasts for **all four models**, plus the auto-selected best.

**Response:**
```json
{
  "state": "Texas",
  "best_model": "XGBoost",
  "metrics": [
    {
      "model": "SARIMA",
      "MAE": 485203012.5,
      "RMSE": 612047831.2,
      "MAPE": 52.3,
      "sMAPE": 40.1
    },
    {
      "model": "Prophet",
      "MAE": 1823041923.1,
      "RMSE": 2104837291.4,
      "MAPE": 201.4,
      "sMAPE": 95.3
    },
    {
      "model": "XGBoost",
      "MAE": 421847293.7,
      "RMSE": 534029183.5,
      "MAPE": 48.4,
      "sMAPE": 37.9
    }
  ],
  "all_forecasts": {
    "SARIMA":  [{ "date": "2024-01-31", "value": 1490231000 }, ...],
    "Prophet": [{ "date": "2024-01-31", "value": 1350000000 }, ...],
    "XGBoost": [{ "date": "2024-01-31", "value": 1572037760 }, ...]
  }
}
```

---

### `GET /forecast/{state}/history`

Returns the full historical sales series plus the forecast — ideal for charting.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `last_n_months` | int | 24 | How many months of history to include (3–120) |

**Response:**
```json
{
  "state": "Florida",
  "best_model": "XGBoost",
  "history": [
    { "date": "2022-01-31", "value": 1120394000 },
    { "date": "2022-02-28", "value": 1089234000 },
    ...
  ],
  "forecast": [
    { "date": "2024-01-31", "value": 1459442048 },
    { "date": "2024-02-29", "value": 1500678400 }
  ]
}
```

---

### `POST /forecast/batch`

Returns forecasts for multiple states in a single request.

**Request body:**
```json
{
  "states": ["California", "Texas", "Florida", "New York"]
}
```

**Response:**
```json
{
  "forecasts": {
    "California": {
      "best_model": "SARIMA",
      "forecast_dates": ["2024-01-31", "2024-02-29"],
      "forecast": [1081889233.46, 1113320813.50]
    },
    "Texas": {
      "best_model": "XGBoost",
      "forecast_dates": ["2024-01-31", "2024-02-29"],
      "forecast": [1572037760.0, 1712047616.0]
    }
  },
  "errors": {}
}
```

---

### `GET /models/leaderboard`

Global model performance — how many states each model won.

**Response:**
```json
{
  "total_states": 43,
  "leaderboard": [
    { "model": "XGBoost", "wins": 36, "win_pct": 83.7, "avg_MAPE": 52.5 },
    { "model": "SARIMA",  "wins": 7,  "win_pct": 16.3, "avg_MAPE": 74.4 }
  ]
}
```

---

## Error Codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `404` | State not found |
| `422` | Validation error (bad request body) |
| `503` | Forecasts not yet loaded (run the training pipeline first) |

---

## Python Client Example

```python
import requests

BASE = "http://localhost:8000"

# Get forecast for a single state
r = requests.get(f"{BASE}/forecast/California")
data = r.json()
print(f"Best model: {data['best_model']}")
for point in data['forecast']:
    print(f"  {point['date']}: ${point['value']:,.0f}")

# Batch forecast
r = requests.post(f"{BASE}/forecast/batch", json={
    "states": ["California", "Texas", "Florida"]
})
for state, info in r.json()["forecasts"].items():
    print(f"{state}: ${info['forecast'][0]:,.0f}")

# Model leaderboard
r = requests.get(f"{BASE}/models/leaderboard")
for entry in r.json()["leaderboard"]:
    print(f"{entry['model']}: {entry['wins']} wins ({entry['win_pct']}%)")
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t forecasting-api .
docker run -p 8000:8000 forecasting-api
```

### Production (Gunicorn + Uvicorn workers)

```bash
pip install gunicorn
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```
