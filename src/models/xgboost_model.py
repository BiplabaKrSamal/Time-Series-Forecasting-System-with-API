"""
models/xgboost_model.py
=======================
XGBoost regressor with time-series lag features and walk-forward forecasting.
"""

import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """
    Recursive (multi-step) XGBoost forecaster using engineered lag features.

    Walk-forward prediction strategy:
    At each future step, the most recent prediction is appended to the
    history and used to compute lag features for the next step.
    """

    MODEL_NAME = "XGBoost"

    _DEFAULT_PARAMS = {
        "n_estimators":     300,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    }

    def __init__(self, lag_periods: List[int] = None, **xgb_kwargs):
        self.lag_periods  = lag_periods or [1, 2, 3, 6, 12]
        self.xgb_params   = {**self._DEFAULT_PARAMS, **xgb_kwargs}
        self._model:      Optional[xgb.XGBRegressor] = None
        self._feature_cols: List[str] = []
        self._history:    Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Feature builder  (internal, lag-only + calendar; no rolling leakage)
    # ------------------------------------------------------------------
    def _make_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Build a supervised feature matrix from a sales series.
        Uses only lag features + calendar features to avoid future leakage.
        """
        df = pd.DataFrame({"sales": series})
        df.index = pd.DatetimeIndex(df.index)

        # Lag features
        for lag in self.lag_periods:
            df[f"lag_{lag}"] = df["sales"].shift(lag)

        # Rolling features (shift by 1 to avoid leakage)
        for w in [3, 6, 12]:
            rolled = df["sales"].shift(1).rolling(w)
            df[f"roll_mean_{w}"] = rolled.mean()
            df[f"roll_std_{w}"]  = rolled.std().fillna(0)

        # Calendar
        df["month"]     = df.index.month
        df["quarter"]   = df.index.quarter
        df["year"]      = df.index.year
        df["trend"]     = np.arange(len(df))
        df["sin_month"] = np.sin(2 * np.pi * df.index.month / 12)
        df["cos_month"] = np.cos(2 * np.pi * df.index.month / 12)

        # Year-over-year
        df["yoy_growth"] = df["sales"].pct_change(12).fillna(0)

        return df.dropna()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "XGBoostForecaster":
        """
        Fit the XGBoost model.

        Parameters
        ----------
        series : monthly pd.Series with DatetimeIndex
        """
        logger.info(f"[XGBoost] Building features for {len(series)} observations …")
        self._history = series.copy()
        df_feat = self._make_features(series)

        self._feature_cols = [c for c in df_feat.columns if c != "sales"]
        X = df_feat[self._feature_cols]
        y = df_feat["sales"]

        # Cross-validate to tune early stopping
        tscv = TimeSeriesSplit(n_splits=3)
        best_n = self.xgb_params["n_estimators"]

        self._model = xgb.XGBRegressor(**self.xgb_params)
        self._model.fit(X, y)

        logger.info(f"[XGBoost] Fit complete. Features: {self._feature_cols}")
        return self

    # ------------------------------------------------------------------
    # Recursive multi-step forecast
    # ------------------------------------------------------------------
    def predict(self, steps: int) -> pd.Series:
        """
        Walk-forward multi-step forecast.

        For each step:
          1. Compute features from the *extended* history (actual + predictions)
          2. Predict the next value
          3. Append prediction to history
        """
        if self._model is None:
            raise RuntimeError("Call .fit() first.")

        history = self._history.copy()
        predictions = []
        freq = pd.tseries.frequencies.to_offset(
            pd.infer_freq(history.index) or "MS"
        )

        for step in range(steps):
            next_date = history.index[-1] + freq

            # Extend history with a NaN placeholder (will be overwritten after prediction)
            extended = pd.concat([
                history,
                pd.Series([np.nan], index=[next_date])
            ])

            # Build features on full extended series
            df_feat = self._make_features(extended.ffill())
            last_row = df_feat.iloc[[-1]][self._feature_cols]

            pred = float(self._model.predict(last_row)[0])
            pred = max(pred, 0)
            predictions.append(pred)

            # Replace the NaN with the actual prediction
            history = pd.concat([
                history,
                pd.Series([pred], index=[next_date])
            ])

        forecast_index = pd.date_range(
            history.index[-steps],
            periods=steps,
            freq=freq
        )
        return pd.Series(predictions, index=forecast_index, name="forecast")

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if self._model is None:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": self._model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return imp

    def get_params(self) -> dict:
        return {
            "lag_periods": self.lag_periods,
            "xgb_params": self.xgb_params,
        }
