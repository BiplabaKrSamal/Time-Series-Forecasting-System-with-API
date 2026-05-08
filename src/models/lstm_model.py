"""
models/lstm_model.py
====================
LSTM deep-learning forecaster with look-back window sequences.
Uses TensorFlow / Keras under the hood.
"""

import logging
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Suppress TF noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    Univariate LSTM forecaster.

    Architecture:  Input(lookback,1) → LSTM(64) → Dropout → Dense(32) → Dense(1)

    Walk-forward recursive prediction for multi-step horizon.
    """

    MODEL_NAME = "LSTM"

    def __init__(self,
                 lookback:   int   = 12,
                 epochs:     int   = 80,
                 batch_size: int   = 16,
                 patience:   int   = 15,
                 units:      int   = 64,
                 dropout:    float = 0.2):

        self.lookback   = lookback
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.units      = units
        self.dropout    = dropout

        self._model:  Optional[object] = None
        self._scaler: Optional[Tuple]  = None
        self._history_scaled: Optional[np.ndarray] = None
        self._freq: Optional[str]      = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _scale(self, series: pd.Series) -> np.ndarray:
        """Min-max scale using training stats; store params for inversion."""
        s_min   = series.min()
        s_range = series.max() - s_min + 1e-8
        self._scaler = (float(s_min), float(s_range))
        return ((series - s_min) / s_range).values.astype(np.float32)

    def _inverse_scale(self, arr: np.ndarray) -> np.ndarray:
        s_min, s_range = self._scaler
        return arr * s_range + s_min

    def _make_sequences(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback: i])
            y.append(scaled[i])
        X = np.array(X).reshape(-1, self.lookback, 1)
        y = np.array(y)
        return X, y

    def _build_model(self):
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        model = Sequential([
            Input(shape=(self.lookback, 1)),
            LSTM(self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="huber")
        return model

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "LSTMForecaster":
        """
        Fit the LSTM on monthly sales data.

        Parameters
        ----------
        series : pd.Series with DatetimeIndex
        """
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.callbacks import EarlyStopping

        logger.info(f"[LSTM] Preparing sequences (lookback={self.lookback}) …")

        # Detect frequency
        self._freq = pd.infer_freq(series.index) or "MS"

        scaled = self._scale(series)
        self._history_scaled = scaled.copy()

        X, y = self._make_sequences(scaled)
        if len(X) == 0:
            raise ValueError(
                f"Not enough data ({len(series)} points) for lookback={self.lookback}. "
                f"Need at least {self.lookback + 1}."
            )

        # Train / validation split within training data (last 20%)
        n_val = max(1, int(len(X) * 0.2))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        self._model = self._build_model()
        early_stop  = EarlyStopping(monitor="val_loss", patience=self.patience,
                                    restore_best_weights=True, verbose=0)

        logger.info(f"[LSTM] Training on {len(X_tr)} sequences …")
        self._model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        logger.info("[LSTM] Training complete.")
        return self

    # ------------------------------------------------------------------
    # Recursive multi-step forecast
    # ------------------------------------------------------------------
    def predict(self, steps: int) -> pd.Series:
        """
        Recursively forecast `steps` months ahead.

        Each predicted value is fed back as an input to the next step.
        """
        if self._model is None:
            raise RuntimeError("Call .fit() first.")

        # Start with the last `lookback` scaled values
        window = list(self._history_scaled[-self.lookback:])
        preds_scaled = []

        for _ in range(steps):
            x = np.array(window[-self.lookback:]).reshape(1, self.lookback, 1)
            p = float(self._model.predict(x, verbose=0)[0][0])
            preds_scaled.append(p)
            window.append(p)

        # Inverse-scale
        preds = self._inverse_scale(np.array(preds_scaled))
        preds = np.maximum(preds, 0)

        # Build DatetimeIndex for forecast
        last_date = pd.Timestamp(
            pd.date_range(end=None, periods=1, freq=self._freq)[0]
            if False else None
        )
        # Find the last actual date
        # We stored history_scaled but not dates; use steps back from "now" trick
        # In practice the orchestrator passes the original series date index
        # so we use a relative index here and the orchestrator reconciles it
        forecast_index = pd.RangeIndex(steps)

        return pd.Series(preds, name="forecast")

    def predict_with_index(self, steps: int, last_date: pd.Timestamp,
                           freq: str = "MS") -> pd.Series:
        """
        Same as predict() but returns a proper DatetimeIndex.

        Parameters
        ----------
        steps     : forecast horizon
        last_date : last date in the training series
        freq      : pandas frequency string (e.g. 'MS')
        """
        forecast = self.predict(steps)
        idx = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
        return pd.Series(forecast.values, index=idx, name="forecast")

    def get_params(self) -> dict:
        return {
            "lookback":   self.lookback,
            "epochs":     self.epochs,
            "batch_size": self.batch_size,
            "units":      self.units,
            "dropout":    self.dropout,
        }
