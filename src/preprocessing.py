"""
preprocessing.py
================
Handles all data loading, cleaning, and date normalization for the
State-level Sales Forecasting System.
"""

import re
import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from dateutil import parser as date_parser

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_PATH = Path(__file__).parents[1] / "data" / "sales_data.csv"
FORECAST_HORIZON = 8          # weeks → translated to 2 months ahead (data is ~monthly)
VALIDATION_PERIODS = 6        # hold-out periods per state for evaluation
MIN_PERIODS_PER_STATE = 12    # drop states with too few data points


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _clean_total(value: str) -> float:
    """Strip whitespace and commas from currency strings, return float."""
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"[,\s]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _parse_date_flexible(date_str: str) -> pd.Timestamp:
    """
    Handle two date formats present in the dataset:
      • M/D/YYYY  → e.g. '1/12/2019'  (American slash)
      • DD-MM-YYYY → e.g. '31-01-2021' (European hyphen)
    Falls back to dateutil for anything else.
    """
    if pd.isna(date_str):
        return pd.NaT
    s = str(date_str).strip()

    # Pattern: DD-MM-YYYY  (day always 28-31 means first part is day)
    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            pass

    # Pattern: M/D/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            pass

    # Fallback
    try:
        return pd.Timestamp(date_parser.parse(s, dayfirst=False))
    except Exception:
        return pd.NaT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load CSV; return raw DataFrame."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, dtype=str)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Standardise column names
    2. Parse dates with mixed formats
    3. Clean 'Total' column (remove commas / spaces)
    4. Drop fully-duplicate rows
    5. Log missing-value summary
    """
    df = df.copy()

    # --- column names ---
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {c: c for c in df.columns}   # identity; adjust if needed
    if "total" not in df.columns:
        raise KeyError(f"Expected 'total' column, found: {list(df.columns)}")

    # --- date ---
    logger.info("Parsing dates …")
    df["date"] = df["date"].apply(_parse_date_flexible)
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        logger.warning(f"  {n_bad_dates} rows have unparseable dates → dropped")
    df = df.dropna(subset=["date"])

    # --- sales ---
    df["total"] = df["total"].apply(_clean_total)
    n_bad_sales = df["total"].isna().sum()
    if n_bad_sales:
        logger.warning(f"  {n_bad_sales} rows have missing/invalid 'total' → will be imputed")

    # --- deduplicate ---
    before = len(df)
    df = df.drop_duplicates(subset=["state", "date"])
    after = len(df)
    if before != after:
        logger.info(f"  Removed {before - after} duplicate (state, date) rows")

    # --- sort ---
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    logger.info(f"After cleaning: {len(df):,} rows, {df['state'].nunique()} states, "
                f"date range {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def build_monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw data contains irregular observation dates.
    1. Snap every observation to its month-end (period-end)
    2. Re-aggregate (sum) in case multiple observations fall in the same month
    3. For each state, forward-fill sparse months so we have a regular index
    4. Interpolate any remaining NaN sales values
    """
    df = df.copy()

    # Snap to month-end
    df["month"] = df["date"].dt.to_period("M")

    # Aggregate duplicates within the same (state, month)
    df = (df.groupby(["state", "month"], as_index=False)["total"]
            .sum())

    # Build a complete monthly date range
    all_periods = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    all_states  = df["state"].unique()

    # Multi-index reindex
    idx = pd.MultiIndex.from_product([all_states, all_periods], names=["state", "month"])
    df = (df.set_index(["state", "month"])
            .reindex(idx)
            .reset_index())

    # Fill gaps: interpolate within each state, then forward/back fill edges
    df = df.sort_values(["state", "month"])
    df["total"] = (
        df.groupby("state")["total"]
          .transform(lambda s: s.interpolate(method="linear")
                                .ffill()
                                .bfill())
    )

    # Convert period back to timestamp (month-end)
    df["date"] = df["month"].dt.to_timestamp("M")

    # Drop states with too few observations
    counts = df.groupby("state")["total"].count()
    valid_states = counts[counts >= MIN_PERIODS_PER_STATE].index
    dropped = set(all_states) - set(valid_states)
    if dropped:
        logger.warning(f"Dropping {len(dropped)} states with < {MIN_PERIODS_PER_STATE} periods: {dropped}")
        df = df[df["state"].isin(valid_states)]

    df = df[["state", "date", "total"]].sort_values(["state", "date"]).reset_index(drop=True)
    logger.info(f"Monthly panel built: {len(df):,} rows, "
                f"{df['state'].nunique()} states, "
                f"{df['date'].nunique()} months")
    return df


def train_val_split(df: pd.DataFrame,
                    val_periods: int = VALIDATION_PERIODS
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split: last `val_periods` months per state → validation;
    everything before → train.  No data leakage.
    """
    splits = []
    for state, grp in df.groupby("state"):
        grp = grp.copy().sort_values("date")
        cutoff = grp["date"].iloc[-val_periods]
        grp["split"] = np.where(grp["date"] >= cutoff, "val", "train")
        splits.append(grp)

    df_split = pd.concat(splits, ignore_index=True)
    train = df_split[df_split["split"] == "train"].drop(columns="split").reset_index(drop=True)
    val   = df_split[df_split["split"] == "val"].drop(columns="split").reset_index(drop=True)

    logger.info(f"Train: {len(train):,} rows | Val: {len(val):,} rows")
    return train, val


def get_state_series(df: pd.DataFrame, state: str) -> pd.Series:
    """Return a DatetimeIndex'd Series for one state."""
    s = (df[df["state"] == state]
           .set_index("date")["total"]
           .sort_index()
           .asfreq("ME"))   # month-end frequency to match panel
    return s


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw   = load_raw_data()
    clean = clean_data(raw)
    panel = build_monthly_panel(clean)
    train, val = train_val_split(panel)
    print(panel.head())
    print(f"\nStates: {sorted(panel['state'].unique())}")
