"""
models/prophet_model.py
=======================
Facebook Prophet wrapper for state-level sales forecasting.
Handles regressors, holiday effects, and weekly/yearly seasonality.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Wraps Facebook Prophet for monthly sales forecasting.

    Prophet automatically handles:
    - Trend change-points
    - Yearly seasonality
    - US holiday effects (optional)
    """

    MODEL_NAME = "Prophet"

    def __init__(self,
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = False,
                 daily_seasonality:  bool = False,
                 add_holidays:       bool = True,
                 changepoint_prior:  float = 0.05):

        self.yearly_seasonality  = yearly_seasonality
        self.weekly_seasonality  = weekly_seasonality
        self.daily_seasonality   = daily_seasonality
        self.add_holidays        = add_holidays
        self.changepoint_prior   = changepoint_prior

        self._model:  Optional[object] = None
        self._last_ds: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
        """Convert a pd.Series to Prophet's required ds/y format."""
        df = pd.DataFrame({
            "ds": series.index,
            "y":  series.values
        })
        df["ds"] = pd.to_datetime(df["ds"])
        return df.reset_index(drop=True)

    def _build_model(self):
        """Lazy import to avoid import-time side effects."""
        from prophet import Prophet
        model = Prophet(
            yearly_seasonality  = self.yearly_seasonality,
            weekly_seasonality  = self.weekly_seasonality,
            daily_seasonality   = self.daily_seasonality,
            changepoint_prior_scale = self.changepoint_prior,
            seasonality_mode    = "multiplicative",
        )
        if self.add_holidays:
            model.add_country_holidays(country_name="US")
        return model

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "ProphetForecaster":
        """
        Fit Prophet model.

        Parameters
        ----------
        series : monthly pd.Series with DatetimeIndex
        """
        logger.info(f"[Prophet] Fitting on {len(series)} observations …")
        df = self._to_prophet_df(series)
        self._model   = self._build_model()
        self._model.fit(df)
        self._last_ds = series.index[-1]
        logger.info("[Prophet] Fit complete.")
        return self

    def predict(self, steps: int) -> pd.Series:
        """
        Forecast `steps` periods ahead (monthly).

        Returns pd.Series with DatetimeIndex and forecasted values.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        future = self._model.make_future_dataframe(periods=steps, freq="MS")
        forecast_df = self._model.predict(future)

        # Keep only the future rows
        fut_mask = forecast_df["ds"] > pd.Timestamp(self._last_ds)
        future_forecast = forecast_df[fut_mask][["ds", "yhat"]].tail(steps)
        future_forecast["yhat"] = np.maximum(future_forecast["yhat"], 0)

        result = pd.Series(
            future_forecast["yhat"].values,
            index=pd.DatetimeIndex(future_forecast["ds"].values),
            name="forecast"
        )
        return result

    def get_components(self) -> Optional[pd.DataFrame]:
        """Return the full forecast DataFrame including trend / seasonality components."""
        if self._model is None:
            return None
        from prophet import Prophet
        future = self._model.make_future_dataframe(periods=0, freq="MS")
        return self._model.predict(future)

    def get_params(self) -> dict:
        return {
            "changepoint_prior": self.changepoint_prior,
            "yearly_seasonality": self.yearly_seasonality,
            "add_holidays": self.add_holidays,
        }
