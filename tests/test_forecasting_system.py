"""
tests/test_forecasting_system.py
==================================
Automated tests for preprocessing, features, models, evaluation, and API.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).parents[1] / "src"
API = Path(__file__).parents[1] / "api"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(API))

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_series():
    rng   = pd.date_range("2020-01-31", periods=36, freq="ME")
    trend = np.linspace(1_000_000, 1_500_000, 36)
    seas  = 200_000 * np.sin(2 * np.pi * np.arange(36) / 12)
    noise = np.random.default_rng(42).normal(0, 20_000, 36)
    return pd.Series(trend + seas + noise, index=rng, name="sales")


@pytest.fixture(scope="module")
def panel():
    from preprocessing import load_raw_data, clean_data, build_monthly_panel
    return build_monthly_panel(clean_data(load_raw_data()))


# -------------------------------------------------------------------------
# 1. Preprocessing
# -------------------------------------------------------------------------

class TestPreprocessing:

    def test_clean_total_strips_commas(self):
        from preprocessing import _clean_total
        assert abs(_clean_total("  109,574,036 ") - 109574036) < 1

    def test_parse_date_slash(self):
        from preprocessing import _parse_date_flexible
        ts = _parse_date_flexible("1/12/2019")
        assert ts.year == 2019

    def test_parse_date_hyphen(self):
        from preprocessing import _parse_date_flexible
        ts = _parse_date_flexible("31-01-2021")
        assert ts.month == 1 and ts.day == 31 and ts.year == 2021

    def test_panel_no_nans(self, panel):
        assert panel["total"].isna().sum() == 0

    def test_panel_sorted(self, panel):
        for state, grp in panel.groupby("state"):
            assert grp["date"].is_monotonic_increasing

    def test_train_val_split_no_leakage(self, panel):
        from preprocessing import train_val_split
        train, val = train_val_split(panel, val_periods=4)
        for state in panel["state"].unique():
            t_dates = set(train[train["state"] == state]["date"])
            v_dates = set(val[val["state"] == state]["date"])
            assert t_dates.isdisjoint(v_dates)


# -------------------------------------------------------------------------
# 2. Features
# -------------------------------------------------------------------------

class TestFeatures:

    def test_build_features_shape(self, sample_series):
        from features import build_features
        df = build_features(sample_series)
        assert "lag_1" in df.columns
        assert "rolling_mean_3" in df.columns
        assert "holiday_month" in df.columns

    def test_prepare_supervised_no_nan(self, sample_series):
        from features import build_features, prepare_supervised
        X, y = prepare_supervised(build_features(sample_series))
        assert X.isna().sum().sum() == 0

    def test_sequences_shape(self, sample_series):
        from features import make_sequences, scale_series
        scaled, _ = scale_series(sample_series)
        X, y = make_sequences(scaled.values, lookback=6)
        assert X.shape[1] == 6 and X.shape[2] == 1


# -------------------------------------------------------------------------
# 3. Models
# -------------------------------------------------------------------------

class TestModelInterfaces:

    def test_sarima_fit_predict(self, sample_series):
        from models.arima_model import ARIMAForecaster
        m = ARIMAForecaster(); m.fit(sample_series)
        fc = m.predict(2)
        assert len(fc) == 2 and (fc >= 0).all()

    def test_prophet_fit_predict(self, sample_series):
        from models.prophet_model import ProphetForecaster
        m = ProphetForecaster(); m.fit(sample_series)
        assert len(m.predict(2)) == 2

    def test_xgboost_fit_predict(self, sample_series):
        from models.xgboost_model import XGBoostForecaster
        m = XGBoostForecaster(); m.fit(sample_series)
        fc = m.predict(2)
        assert len(fc) == 2 and (fc >= 0).all()

    def test_lstm_fit_predict(self, sample_series):
        from models.lstm_model import LSTMForecaster
        m = LSTMForecaster(lookback=6, epochs=3, patience=2)
        m.fit(sample_series)
        fc = m.predict_with_index(2, last_date=sample_series.index[-1], freq="ME")
        assert len(fc) == 2

    def test_model_raises_without_fit(self):
        from models.arima_model import ARIMAForecaster
        with pytest.raises(RuntimeError):
            ARIMAForecaster().predict(2)


# -------------------------------------------------------------------------
# 4. Evaluation
# -------------------------------------------------------------------------

class TestEvaluation:

    def test_mape_zero_actuals(self):
        from evaluation import mean_absolute_percentage_error
        assert np.isnan(mean_absolute_percentage_error(
            np.array([0, 0, 0]), np.array([1, 2, 3])
        ))

    def test_metrics_dict_keys(self):
        from evaluation import compute_metrics
        m = compute_metrics(np.array([100, 200]), np.array([110, 190]), "test")
        assert {"model", "MAE", "RMSE", "MAPE", "sMAPE"} == set(m.keys())

    def test_selector_best_model(self):
        from evaluation import ModelSelector
        sel = ModelSelector()
        y = np.array([100, 200, 300])
        sel.add_result("ModelA", y, np.array([100, 200, 300]))
        sel.add_result("ModelB", y, np.array([200, 100, 150]))
        assert sel.best_model() == "ModelA"


# -------------------------------------------------------------------------
# 5. API
# -------------------------------------------------------------------------

class TestAPI:

    @pytest.fixture(scope="class")
    def client(self):
        import json
        import main as api_main
        fc_path = Path(__file__).parents[1] / "outputs" / "forecasts" / "all_forecasts.json"
        fc_path.parent.mkdir(parents=True, exist_ok=True)
        if not fc_path.exists():
            fake = {"California": {
                "best_model": "XGBoost",
                "forecast_dates": ["2024-01-31", "2024-02-29"],
                "forecast_values": [500000000.0, 510000000.0],
                "all_model_forecasts": {"XGBoost": [500000000.0, 510000000.0]},
                "metrics": [{"model": "XGBoost", "MAE": 1e6, "RMSE": 2e6,
                              "MAPE": 2.1, "sMAPE": 2.0}],
            }}
            with open(fc_path, "w") as f:
                json.dump(fake, f)

        with open(fc_path) as f:
            api_main._forecasts = json.load(f)

        from fastapi.testclient import TestClient
        return TestClient(api_main.app)

    def test_root_returns_ok(self, client):
        assert client.get("/").json()["status"] == "ok"

    def test_health(self, client):
        assert client.get("/health").status_code == 200

    def test_states_list(self, client):
        r = client.get("/states")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) > 0

    def test_forecast_california(self, client):
        r = client.get("/forecast/California")
        assert r.status_code == 200
        data = r.json()
        assert data["state"] == "California"
        assert len(data["forecast"]) >= 2

    def test_forecast_unknown_state(self, client):
        assert client.get("/forecast/Narnia").status_code == 404

    def test_batch_forecast(self, client):
        r = client.post("/forecast/batch", json={"states": ["California"]})
        assert r.status_code == 200
        assert "California" in r.json()["forecasts"]

    def test_leaderboard(self, client):
        r = client.get("/models/leaderboard")
        assert r.status_code == 200
        assert "leaderboard" in r.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
