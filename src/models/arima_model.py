"""
models/arima_model.py
=====================
ARIMA / SARIMA wrapper with automatic order selection via AIC grid search.
"""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """
    Fits a SARIMA(p,d,q)(P,D,Q,m) model per state.
    Automatically selects order via AIC grid search over a compact candidate set.
    """

    MODEL_NAME = "SARIMA"
    SEASONAL_PERIOD = 12   # monthly → annual seasonality

    # Candidate orders for grid search
    _P_RANGE = [0, 1, 2]
    _D_RANGE = [0, 1]
    _Q_RANGE = [0, 1, 2]
    _SP_RANGE = [0, 1]
    _SD_RANGE = [0, 1]
    _SQ_RANGE = [0, 1]

    def __init__(self):
        self.best_params_: Optional[Tuple] = None
        self.model_fit_    = None
        self.fitted_values_: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Stationarity check
    # ------------------------------------------------------------------
    @staticmethod
    def _is_stationary(series: pd.Series, alpha: float = 0.05) -> bool:
        try:
            p_value = adfuller(series.dropna())[1]
            return p_value < alpha
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Auto-order selection
    # ------------------------------------------------------------------
    def _select_order(self, series: pd.Series) -> Tuple:
        """
        Perform a lightweight AIC grid search.
        To keep runtime manageable we restrict the seasonal part to (0/1, 0/1, 0/1, 12).
        """
        d = 0 if self._is_stationary(series) else 1

        best_aic    = np.inf
        best_params = (1, d, 1, 0, 1, 0, self.SEASONAL_PERIOD)

        candidates = []
        for p in self._P_RANGE:
            for q in self._Q_RANGE:
                for P in self._SP_RANGE:
                    for D in self._SD_RANGE:
                        for Q in self._SQ_RANGE:
                            candidates.append((p, d, q, P, D, Q, self.SEASONAL_PERIOD))

        for params in candidates:
            try:
                p, d_, q, P, D, Q, m = params
                model = SARIMAX(series, order=(p, d_, q),
                                seasonal_order=(P, D, Q, m),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                res = model.fit(disp=False, maxiter=100)
                if res.aic < best_aic:
                    best_aic    = res.aic
                    best_params = params
            except Exception:
                continue

        logger.debug(f"  Best SARIMA params: {best_params} (AIC={best_aic:.2f})")
        return best_params

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """
        Fit SARIMA to the training series.

        Parameters
        ----------
        series : monthly pd.Series with DatetimeIndex
        """
        logger.info(f"[SARIMA] Selecting best order for series of length {len(series)} …")
        self.best_params_ = self._select_order(series)
        p, d, q, P, D, Q, m = self.best_params_

        model = SARIMAX(series, order=(p, d, q),
                        seasonal_order=(P, D, Q, m),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        self.model_fit_    = model.fit(disp=False, maxiter=200)
        self.fitted_values_ = self.model_fit_.fittedvalues
        logger.info(f"[SARIMA] Fitted SARIMA{(p,d,q)}×{(P,D,Q,m)} | AIC={self.model_fit_.aic:.2f}")
        return self

    def predict(self, steps: int) -> pd.Series:
        """
        Return `steps` out-of-sample forecasts as a pd.Series.
        Clips predictions to be non-negative.
        """
        if self.model_fit_ is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
        forecast = self.model_fit_.forecast(steps=steps)
        forecast = np.maximum(forecast, 0)
        return forecast

    def get_params(self) -> dict:
        return {
            "order":          self.best_params_[:3] if self.best_params_ else None,
            "seasonal_order": self.best_params_[3:] if self.best_params_ else None,
            "aic":            self.model_fit_.aic if self.model_fit_ else None,
        }
