"""
BaseRNNModel — shared logic for all stacked-RNN forecasters (LSTM, GRU).

Subclasses only need to implement _build_network(), which receives the input
shape and must return a compiled Keras Sequential model.

Architecture (fixed across all RNN variants):
    RNNCell(units_1, return_sequences=True)
    Dropout(dropout_rate)
    RNNCell(units_2)
    Dropout(dropout_rate)
    Dense(16, relu)
    Dense(forecast_horizon)          ← multi-step output block
    Compiled with optimizer=adam, loss=mae

Feature set (built internally from the pd.Series DatetimeIndex):
    - smoothed target   (rolling mean, MinMax-scaled per training window)
    - sin/cos week-of-year encoding  (circular seasonal signal)

Walk-forward notes:
    - fit() / predict() follow the BaseForecaster interface so this model
      plugs directly into WalkForwardValidator without any modifications.
    - refit_every=1 is strongly recommended: the MinMaxScaler is fitted per
      training window, so reusing it across windows with shifted ranges will
      degrade predictions.

Persistence:
    - Keras models are not picklable; save()/load() use a directory format:
        <dir>/keras_model.keras   — Keras native format
        <dir>/state.pkl           — scaler + hyperparams + class info
        <dir>/<Name>_metadata.json
"""

import importlib
import json
import pickle
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pneumonia.config import MODEL_STORAGE_PATH, RANDOM_SEED
from pneumonia.models.base import BaseForecaster
from pneumonia.models.rnn.config import RNN_DEFAULT_PARAMS
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class BaseRNNModel(BaseForecaster):
    """
    Abstract base for stacked-RNN forecasters.

    Parameters
    ----------
    name            : str   Model identifier ('LSTM' or 'GRU').
    department      : str
    age_group       : str   'under5' or '60plus'
    lookback        : int   Weeks of history in each input sequence.
    forecast_horizon: int   Dense output size (internal block size).
    epochs          : int   Maximum training epochs.
    batch_size      : int
    val_weeks       : int   Last N sequences used for EarlyStopping validation.
    units_1         : int   Units in the first recurrent layer.
    units_2         : int   Units in the second recurrent layer.
    dropout_rate    : float Dropout applied after each recurrent layer.
    smooth_window   : int   Rolling-mean window; set 1 to disable.
    """

    def __init__(
        self,
        name: str,
        department: str,
        age_group: str,
        lookback: int         = RNN_DEFAULT_PARAMS["lookback"],
        forecast_horizon: int = RNN_DEFAULT_PARAMS["forecast_horizon"],
        epochs: int           = RNN_DEFAULT_PARAMS["epochs"],
        batch_size: int       = RNN_DEFAULT_PARAMS["batch_size"],
        val_weeks: int        = RNN_DEFAULT_PARAMS["val_weeks"],
        units_1: int          = RNN_DEFAULT_PARAMS["units_1"],
        units_2: int          = RNN_DEFAULT_PARAMS["units_2"],
        dropout_rate: float   = RNN_DEFAULT_PARAMS["dropout_rate"],
        smooth_window: int    = RNN_DEFAULT_PARAMS["smooth_window"],
    ):
        super().__init__(name=name, department=department, age_group=age_group)

        self.lookback         = lookback
        self.forecast_horizon = forecast_horizon
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.val_weeks        = val_weeks
        self.units_1          = units_1
        self.units_2          = units_2
        self.dropout_rate     = dropout_rate
        self.smooth_window    = smooth_window

        self._keras_model = None
        self._scaler: Optional[MinMaxScaler] = None

        self.metadata.update({
            "lookback":         lookback,
            "forecast_horizon": forecast_horizon,
            "epochs":           epochs,
            "batch_size":       batch_size,
            "val_weeks":        val_weeks,
            "units_1":          units_1,
            "units_2":          units_2,
            "dropout_rate":     dropout_rate,
            "smooth_window":    smooth_window,
        })

    def get_params(self) -> dict:
        return {
            "lookback":         self.lookback,
            "forecast_horizon": self.forecast_horizon,
            "epochs":           self.epochs,
            "batch_size":       self.batch_size,
            "val_weeks":        self.val_weeks,
            "units_1":          self.units_1,
            "units_2":          self.units_2,
            "dropout_rate":     self.dropout_rate,
            "smooth_window":    self.smooth_window,
        }

    # ------------------------------------------------------------------
    # Abstract — subclasses provide the recurrent cell
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_network(self, input_shape: tuple):
        """Return a compiled Keras Sequential model for the given input shape."""

    # ------------------------------------------------------------------
    # Helper — build the shared 2-layer stacked architecture
    # ------------------------------------------------------------------

    def _stack_rnn(self, cell_class, input_shape: tuple):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Input

        model = Sequential([
            Input(shape=input_shape),
            cell_class(self.units_1, return_sequences=True),
            Dropout(self.dropout_rate),
            cell_class(self.units_2),
            Dropout(self.dropout_rate),
            Dense(16, activation="relu"),
            Dense(self.forecast_horizon),
        ])
        model.compile(optimizer="adam", loss="mae")
        return model

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        """
        Smooth → scale → build sequences → train Keras RNN.

        Args:
            train_data: Weekly series with DatetimeIndex (raw case counts).
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        if not isinstance(train_data, pd.Series):
            raise TypeError(f"Expected pd.Series, got {type(train_data)}")

        min_len = self.lookback + self.forecast_horizon + self.val_weeks
        if len(train_data) < min_len:
            raise ValueError(
                f"train_data has {len(train_data)} obs; need at least {min_len} "
                f"(lookback={self.lookback} + horizon={self.forecast_horizon} "
                f"+ val_weeks={self.val_weeks})"
            )

        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        logger.info(
            f"{self.name} fit — {self.department}/{self.age_group}, "
            f"{len(train_data)} obs, lookback={self.lookback}, "
            f"horizon={self.forecast_horizon}"
        )

        smooth = self._smooth(train_data.values.astype(float))
        exog   = self._seasonal_features(train_data.index)

        self._scaler  = MinMaxScaler()
        target_scaled = self._scaler.fit_transform(smooth.reshape(-1, 1))

        X, y = self._create_sequences(target_scaled, exog)

        if len(X) <= self.val_weeks:
            raise ValueError(
                f"Not enough sequences ({len(X)}) for val_weeks={self.val_weeks}. "
                f"Reduce val_weeks or increase train_size."
            )

        X_train, y_train = X[:-self.val_weeks], y[:-self.val_weeks]
        X_val,   y_val   = X[-self.val_weeks:],  y[-self.val_weeks:]

        self._keras_model = self._build_network(
            input_shape=(self.lookback, X_train.shape[2])
        )

        self._keras_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
            ],
        )

        self.is_fitted   = True
        self.fitted_date = datetime.now().isoformat()
        self.metadata.update({"train_size": len(train_data)})
        logger.info(f"{self.name} fit complete.")

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        """
        Autoregressive block rollout anchored to the end of data.

        Predicts `forecast_horizon` steps at a time, appends them to the
        history buffer, and repeats until `steps` total values are collected.

        Args:
            data:  Real observations available at prediction time.
            steps: Number of weeks to forecast ahead.

        Returns:
            np.ndarray of shape (steps,), clipped to >= 0.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        n_needed = self.lookback + self.smooth_window - 1
        if len(data) < n_needed:
            raise ValueError(
                f"data has {len(data)} obs; need at least {n_needed} "
                f"(lookback={self.lookback} + smooth_window-1={self.smooth_window - 1})"
            )

        logger.info(
            f"{self.name} predict — {steps} steps from {len(data)} obs "
            f"({self.department}/{self.age_group})"
        )

        recent_vals   = data.values[-n_needed:].astype(float)
        smooth_vals   = pd.Series(recent_vals).rolling(
            window=self.smooth_window, min_periods=1
        ).mean().values
        window_scaled = self._scaler.transform(
            smooth_vals[-self.lookback:].reshape(-1, 1)
        )
        window_exog = self._seasonal_features(data.index[-self.lookback:])

        predictions: list  = []
        current_target     = window_scaled.copy()
        current_exog       = window_exog.copy()
        last_date          = data.index[-1]

        while len(predictions) < steps:
            seq = np.concatenate(
                [current_target[-self.lookback:], current_exog[-self.lookback:]],
                axis=1,
            ).reshape(1, self.lookback, -1)

            pred_scaled = self._keras_model.predict(seq, verbose=0).reshape(-1, 1)
            block       = np.clip(
                self._scaler.inverse_transform(pred_scaled).flatten(), 0.0, None
            )
            predictions.extend(block.tolist())

            next_dates  = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=self.forecast_horizon,
                freq="7D",
            )
            next_exog   = self._seasonal_features(next_dates)
            next_scaled = self._scaler.transform(block.reshape(-1, 1))

            current_target = np.vstack([current_target, next_scaled])
            current_exog   = np.vstack([current_exog, next_exog])
            last_date      = next_dates[-1]

        return np.array(predictions[:steps])

    # ------------------------------------------------------------------
    # Save / Load  (override: Keras models are not picklable)
    # ------------------------------------------------------------------

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save to a directory:
          keras_model.keras, state.pkl, <Name>_metadata.json
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        save_dir = (
            Path(filepath)
            if filepath is not None
            else MODEL_STORAGE_PATH / self.department / self.age_group / self.name
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        self._keras_model.save(save_dir / "keras_model.keras")

        state = {
            "class_module":   type(self).__module__,
            "class_name":     type(self).__name__,
            "scaler":         self._scaler,
            "department":     self.department,
            "age_group":      self.age_group,
            "lookback":       self.lookback,
            "forecast_horizon": self.forecast_horizon,
            "epochs":         self.epochs,
            "batch_size":     self.batch_size,
            "val_weeks":      self.val_weeks,
            "units_1":        self.units_1,
            "units_2":        self.units_2,
            "dropout_rate":   self.dropout_rate,
            "smooth_window":  self.smooth_window,
            "is_fitted":      self.is_fitted,
            "fitted_date":    self.fitted_date,
            "metadata":       self.metadata,
        }
        with open(save_dir / "state.pkl", "wb") as f:
            pickle.dump(state, f)

        with open(save_dir / f"{self.name}_metadata.json", "w") as f:
            json.dump(
                {**self.metadata, "saved_date": datetime.now().isoformat()},
                f, indent=2, default=str,
            )

        logger.info(f"{self.name} model saved to {save_dir}")
        return save_dir

    @staticmethod
    def load(filepath: Path) -> "BaseRNNModel":
        """
        Load any RNN subclass from its save directory.

        The correct subclass (LSTMModel / GRUModel) is resolved automatically
        from the class info stored in state.pkl.
        """
        import tensorflow as tf

        save_dir = Path(filepath)
        if save_dir.is_file():
            save_dir = save_dir.parent

        state_path = save_dir / "state.pkl"
        keras_path = save_dir / "keras_model.keras"

        if not state_path.exists():
            raise FileNotFoundError(f"state.pkl not found in {save_dir}")
        if not keras_path.exists():
            raise FileNotFoundError(f"keras_model.keras not found in {save_dir}")

        with open(state_path, "rb") as f:
            state = pickle.load(f)

        cls = getattr(
            importlib.import_module(state["class_module"]),
            state["class_name"],
        )

        instance = cls(
            department=state["department"],
            age_group=state["age_group"],
            lookback=state["lookback"],
            forecast_horizon=state["forecast_horizon"],
            epochs=state["epochs"],
            batch_size=state["batch_size"],
            val_weeks=state["val_weeks"],
            units_1=state["units_1"],
            units_2=state["units_2"],
            dropout_rate=state["dropout_rate"],
            smooth_window=state["smooth_window"],
        )

        instance._scaler      = state["scaler"]
        instance._keras_model = tf.keras.models.load_model(keras_path)
        instance.is_fitted    = state["is_fitted"]
        instance.fitted_date  = state["fitted_date"]
        instance.metadata     = state["metadata"]

        logger.info(f"{state['class_name']} loaded from {save_dir}")
        return instance

    # ------------------------------------------------------------------
    # Internal feature engineering
    # ------------------------------------------------------------------

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        return (
            pd.Series(values)
            .rolling(window=self.smooth_window, min_periods=1)
            .mean()
            .values
        )

    @staticmethod
    def _seasonal_features(idx: pd.DatetimeIndex) -> np.ndarray:
        week = idx.isocalendar().week.astype(int).values
        return np.column_stack([
            np.sin(2 * np.pi * week / 52.0),
            np.cos(2 * np.pi * week / 52.0),
        ])

    def _create_sequences(self, target_scaled: np.ndarray, exog: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(target_scaled) - self.forecast_horizon + 1):
            seq = np.concatenate(
                [target_scaled[i - self.lookback:i], exog[i - self.lookback:i]],
                axis=1,
            )
            X.append(seq)
            y.append(target_scaled[i:i + self.forecast_horizon, 0])
        return np.array(X), np.array(y)
