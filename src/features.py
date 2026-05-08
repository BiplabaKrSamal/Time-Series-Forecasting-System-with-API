"""
features.py
===========
All feature engineering for the ML-based models (XGBoost, LSTM).
Creates lag features, rolling statistics, calendar features, and holiday flags.
"""

import logging
from typing import List, Tuple

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calendar / Holiday helpers
# ---------------------------------------------------------------------------

US_HOLIDAYS = holidays.UnitedStates(years=range(2015, 2030))


def _is_holiday_month(date: pd.Timestamp) -> int:
    """Return 1 if any major US holiday falls in the same month."""
    month_start = date.replace(day=1)
    month_end   = (month_start + pd.offsets.MonthEnd(1))
    for d in pd.date_range(month_start, month_end, freq="D"):
        if d in US_HOLIDAYS:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def build_features(series: pd.Series,
                   lag_periods: List[int] = None,
                   rolling_windows: List[int] = None) -> pd.DataFrame:
    """
    Given a monthly sales Series (DatetimeIndex), produce a feature DataFrame
    with:
      • Lag features         : sales at t-1, t-2, t-3, t-6, t-12
      • Rolling statistics   : rolling mean & std over 3, 6, 12 months
      • Calendar features    : month-of-year, quarter, year, trend index
      • Holiday flag         : 1 if a major US holiday is in that month

    Parameters
    ----------
    series        : pd.Series with DatetimeIndex (monthly frequency)
    lag_periods   : list of int lags; defaults are [1, 2, 3, 6, 12]
    rolling_windows: list of int windows; defaults are [3, 6, 12]

    Returns
    -------
    pd.DataFrame  : feature matrix (rows with NaN from lags are *kept*;
                    caller decides whether to drop them)
    """
    if lag_periods is None:
        # t-1, t-7, t-30 interpreted as 1-period, 2-period, 3-period
        # plus longer memories
        lag_periods = [1, 2, 3, 6, 12]
    if rolling_windows is None:
        rolling_windows = [3, 6, 12]

    df = pd.DataFrame({"sales": series})
    df.index = pd.DatetimeIndex(df.index)

    # --- Lag features -------------------------------------------------------
    for lag in lag_periods:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # --- Rolling statistics -------------------------------------------------
    for w in rolling_windows:
        rolled = df["sales"].shift(1).rolling(window=w)   # shift(1) → no leakage
        df[f"rolling_mean_{w}"] = rolled.mean()
        df[f"rolling_std_{w}"]  = rolled.std()

    # --- Calendar features --------------------------------------------------
    df["month"]     = df.index.month
    df["quarter"]   = df.index.quarter
    df["year"]      = df.index.year
    df["trend"]     = np.arange(len(df))   # linear trend proxy

    # sine / cosine seasonality encoding (better than raw month for tree models)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Holiday flag -------------------------------------------------------
    df["holiday_month"] = df.index.map(_is_holiday_month)

    # --- Year-over-year growth (when available) ----------------------------
    df["yoy_growth"] = df["sales"].pct_change(12)

    return df


def prepare_supervised(df_features: pd.DataFrame,
                       target_col: str = "sales",
                       min_lag: int = 12) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop the leading NaN rows caused by the maximum lag / rolling window,
    then split into X (features) and y (target).

    Parameters
    ----------
    df_features : output of build_features()
    target_col  : name of the target column
    min_lag     : minimum number of rows to discard at the start

    Returns
    -------
    X, y as (pd.DataFrame, pd.Series)
    """
    # Drop rows where ANY feature is NaN (caused by lags / rolling)
    df_clean = df_features.dropna()

    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    return X, y


def scale_series(train_series: pd.Series,
                 val_series:   pd.Series | None = None):
    """
    Min-max scale a series using only train statistics (no leakage).
    Returns scaled arrays and the (min, scale) pair for inverse transform.
    """
    s_min   = train_series.min()
    s_scale = train_series.max() - s_min + 1e-8   # avoid div-by-zero

    train_scaled = (train_series - s_min) / s_scale

    if val_series is not None:
        val_scaled = (val_series - s_min) / s_scale
        return train_scaled, val_scaled, (s_min, s_scale)

    return train_scaled, (s_min, s_scale)


def inverse_scale(scaled_values, scaler_params: tuple) -> np.ndarray:
    """Undo min-max scaling."""
    s_min, s_scale = scaler_params
    return np.array(scaled_values) * s_scale + s_min


def make_sequences(series: np.ndarray,
                   lookback: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) sliding-window sequences for LSTM.

    Parameters
    ----------
    series   : 1-D numpy array of scaled sales values
    lookback : number of past months used as input

    Returns
    -------
    X : shape (n_samples, lookback, 1)
    y : shape (n_samples,)
    """
    X_list, y_list = [], []
    for i in range(lookback, len(series)):
        X_list.append(series[i - lookback:i])
        y_list.append(series[i])
    X = np.array(X_list).reshape(-1, lookback, 1)
    y = np.array(y_list)
    return X, y


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocessing import load_raw_data, clean_data, build_monthly_panel, get_state_series
    panel  = build_monthly_panel(clean_data(load_raw_data()))
    series = get_state_series(panel, "California")
    feats  = build_features(series)
    X, y   = prepare_supervised(feats)
    print(f"California features: {X.shape}, target: {y.shape}")
    print(X.head())
