"""
Naive baseline forecaster: repeats the last observed value for all future steps.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from pneumonia.models.base import BaseForecaster
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class NaiveForecaster(BaseForecaster):
    """
    Naive forecaster that repeats the last observed training value.

    Useful as a lower-bound baseline: any decent model should beat this.
    """

    def __init__(self, department: str, age_group: str):
        super().__init__("Naive", department, age_group)
        self._last_value: float = None

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        if len(train_data) == 0:
            raise ValueError("Training data is empty")

        self._last_value = float(train_data.iloc[-1])
        self.is_fitted = True
        self.fitted_date = datetime.now().isoformat()
        self.metadata.update({
            "last_value": self._last_value,
            "train_size": len(train_data),
            "fit_method": "last_value",
        })
        logger.info(f"Naive fitted — last value: {self._last_value:.4f}")

    def predict(self, steps: int) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        return np.full(steps, self._last_value)
