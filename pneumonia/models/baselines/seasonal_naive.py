"""
Seasonal naive forecaster: repeats the block of values from the last season.

For weekly data with season_length=52 this means each forecast step gets
the value observed exactly one year prior in the training set.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from pneumonia.models.base import BaseForecaster
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naive forecaster.

    predict(h) returns the h values that cycle through the last full season
    stored during fit(), i.e. forecast[i] = train[-season_length + (i % season_length)].
    """

    def __init__(self, department: str, age_group: str, season_length: int = 52):
        super().__init__("SeasonalNaive", department, age_group, season_length=season_length)
        self.season_length = season_length
        self._season_values: np.ndarray = None

    def get_params(self) -> dict:
        return {"season_length": self.season_length}

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        if len(train_data) < self.season_length:
            raise ValueError(
                f"Training data has {len(train_data)} observations but "
                f"season_length is {self.season_length}. Need at least one full season."
            )

        self._season_values = train_data.values[-self.season_length:].astype(float)
        self.is_fitted = True
        self.fitted_date = datetime.now().isoformat()
        self.metadata.update({
            "season_length": self.season_length,
            "train_size": len(train_data),
            "fit_method": "last_season_repeat",
            "season_mean": float(self._season_values.mean()),
        })
        logger.info(
            f"SeasonalNaive fitted — season length: {self.season_length}, "
            f"season mean: {self._season_values.mean():.4f}"
        )

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        available = len(data)
        if available < self.season_length:
            logger.warning(
                f"SeasonalNaive: data has {available} observations, "
                f"less than season_length={self.season_length}. Using all available."
            )
            season_vals = data.values.astype(float)
        else:
            season_vals = data.values[-self.season_length:].astype(float)

        logger.info(
            f"SeasonalNaive predict — anchoring on last {len(season_vals)} observations, "
            f"steps: {steps}"
        )
        n_full = (steps + len(season_vals) - 1) // len(season_vals)
        return np.tile(season_vals, n_full)[:steps]
