"""
evaluation.py
=============
Metric computation, model comparison, and best-model selection logic.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE, handles zero actuals by replacing with a small epsilon."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE — bounded between 0 % and 200 %."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom  = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask   = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str = "") -> Dict[str, float]:
    """Return a dict of all evaluation metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return {
        "model":  model_name,
        "MAE":    round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAPE":   round(mean_absolute_percentage_error(y_true, y_pred), 4),
        "sMAPE":  round(symmetric_mape(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# Model comparison and selection
# ---------------------------------------------------------------------------

class ModelSelector:
    """
    Collects validation metrics from multiple models and picks the best one
    based on a primary metric (default: MAPE).

    Usage
    -----
    selector = ModelSelector()
    selector.add_result("SARIMA",  y_val, sarima_preds)
    selector.add_result("Prophet", y_val, prophet_preds)
    selector.add_result("XGBoost", y_val, xgb_preds)
    selector.add_result("LSTM",    y_val, lstm_preds)
    best     = selector.best_model()       # name of best model
    summary  = selector.summary_table()    # pd.DataFrame
    """

    def __init__(self, primary_metric: str = "MAPE"):
        self.primary_metric = primary_metric
        self._results: List[Dict] = []

    def add_result(self, model_name: str,
                   y_true: np.ndarray,
                   y_pred: np.ndarray) -> Dict:
        metrics = compute_metrics(y_true, y_pred, model_name=model_name)
        self._results.append(metrics)
        logger.info(
            f"  [{model_name}] MAE={metrics['MAE']:,.0f} | "
            f"RMSE={metrics['RMSE']:,.0f} | "
            f"MAPE={metrics['MAPE']:.2f}% | "
            f"sMAPE={metrics['sMAPE']:.2f}%"
        )
        return metrics

    def best_model(self) -> str:
        if not self._results:
            raise ValueError("No results have been added yet.")
        df = self.summary_table()
        # Prefer the model with the lowest MAPE; break ties with RMSE
        sorted_df = df.reset_index().sort_values([self.primary_metric, "RMSE"])
        winner = sorted_df.iloc[0]["model"]
        logger.info(f"  ★ Best model: {winner} "
                    f"({self.primary_metric}={sorted_df.iloc[0][self.primary_metric]:.2f}%)")
        return str(winner)

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(self._results).set_index("model")

    def to_dict(self) -> List[Dict]:
        return self._results
