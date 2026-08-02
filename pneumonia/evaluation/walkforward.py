"""
Walk-forward validation (rolling-origin backtesting) for time series models.

All models build features internally from pd.Series, so no external feature
matrix is needed — the validator only passes the training Series to fit()
and the anchoring Series to predict().
"""

from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
import pandas as pd

from pneumonia.evaluation.metrics import compute_all_metrics
from pneumonia.models.base import BaseForecaster
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class WalkForwardValidator:
    """
    Walk-forward (rolling-origin) backtesting for BaseForecaster subclasses.

    Parameters
    ----------
    model_class : Type[BaseForecaster]
        The model class to instantiate at each refit.
    model_params : dict
        Keyword arguments passed to model_class() (e.g. department, age_group).
    initial_train_size : int | str | pd.Timestamp
        Size of the first training window (int = weeks) or the end date of it.
    horizon : int
        Weeks to forecast at each origin (default 4 = 1 month).
    step : int
        Weeks to advance the origin between evaluations (default 4).
    window_type : 'sliding' | 'expanding'
        'sliding'   — fixed-size window moves forward (recommended).
        'expanding' — window grows from the start.
    refit_every : int
        Re-train the model every N steps. 1 = every step, 0 = fit once only.
        For SARIMA use a higher value (e.g. 52) to keep runtime tractable.
    min_train_size : int
        Minimum weeks required to fit the model (default 104 = 2 years).
    fit_kwargs : dict
        Extra keyword arguments forwarded to model.fit().

    This class has no notion of "excluded" or "invalid" dates — it trains,
    forecasts, and scores every step uniformly. Any date-based scoring policy
    (e.g. dropping a known reporting-shock period from the reported metrics)
    is intentionally kept out of this engine and applied afterward, on this
    method's output — see pneumonia.evaluation.metrics.recompute_metrics(),
    called from pneumonia.pipelines.walkforward_runner.run_walkforward_for().
    """

    def __init__(
        self,
        model_class: Type[BaseForecaster],
        model_params: Dict[str, Any],
        initial_train_size: Union[int, str, pd.Timestamp],
        horizon: int = 4,
        step: int = 4,
        window_type: Literal["sliding", "expanding"] = "sliding",
        refit_every: int = 1,
        min_train_size: int = 104,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        if window_type not in ("sliding", "expanding"):
            raise ValueError(f"window_type must be 'sliding' or 'expanding', got {window_type}")

        self.model_class          = model_class
        self.model_params         = model_params
        self.initial_train_size   = initial_train_size
        self.horizon              = horizon
        self.step                 = step
        self.window_type          = window_type
        self.refit_every          = refit_every if refit_every > 0 else 0
        self.min_train_size       = min_train_size
        self.fit_kwargs           = fit_kwargs or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        y: pd.Series,
        step_range: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Execute the walk-forward loop over series y.

        Args:
            y: Weekly time series with DatetimeIndex.
            step_range: (start_step_idx, end_step_idx) to compute only a
                sub-range of steps, using the same GLOBAL step numbering a
                full run (step_range=None) would use — so refit_every
                decisions land on the same steps regardless of whether the
                full range or a sub-range is requested. end_step_idx is
                exclusive; None means "through the end of the series".
                When start_step_idx > 0, the model is "bridge-fit" once on
                the training window of the last step at-or-before
                start_step_idx that a continuous run would have refit on
                (found by rounding down to the nearest multiple of
                refit_every), so the model handed to the first computed
                step is in the same state a continuous run would have had
                — without re-executing any of the earlier steps. This is
                what lets a "search" (pre-holdout) run and a "holdout" run
                be computed independently, later, in either order, while
                still reproducing one continuous run's refit schedule and
                predictions exactly.

        Returns:
            {
              'metrics_by_horizon': {1: {mae, rmse, ...}, 2: {...}, ...},
              'step_results':       list of per-step dicts,
              'predictions':        DataFrame(index=dates, cols=[actual, pred_h1..hN]),
              'aggregate_metrics':  metrics averaged across all steps (h=1),
            }
        """
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pd.Series, got {type(y)}")
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("y must have a DatetimeIndex")

        train_size = self._resolve_train_size(y)
        n = len(y)

        if train_size >= n:
            raise ValueError(
                f"initial_train_size ({train_size}) must be less than series length ({n} weeks, "
                f"{y.index[0].date()} → {y.index[-1].date()}). "
                f"Pass a smaller --train_size (max {n - 1}) or provide more history."
            )
        if train_size < self.min_train_size:
            raise ValueError(
                f"initial_train_size ({train_size}) < min_train_size ({self.min_train_size}). "
                f"Series has {n} weeks available ({y.index[0].date()} → {y.index[-1].date()}). "
                f"Pass a smaller --train_size (up to {n - 1}, keeping enough for evaluation steps), "
                f"lower --min_train_size if you accept a shorter seasonal history, "
                f"or provide more historical data (e.g. remove/adjust --start_year)."
            )

        start_step_idx, end_step_idx = step_range if step_range is not None else (0, None)
        if start_step_idx < 0:
            raise ValueError(f"step_range start must be >= 0, got {start_step_idx}")
        if end_step_idx is not None and end_step_idx <= start_step_idx:
            raise ValueError(
                f"step_range end ({end_step_idx}) must be > start ({start_step_idx})"
            )

        # predictions DataFrame: one row per evaluation date (full series
        # range regardless of step_range, so a partial run's DataFrame lines
        # up positionally with a full run's — callers merge on this index).
        eval_index = y.index[train_size:]
        pred_df = pd.DataFrame({"actual": y.iloc[train_size:]}, index=eval_index)
        for h in range(1, self.horizon + 1):
            pred_df[f"pred_h{h}"] = np.nan

        step_results: List[Dict] = []
        model: Optional[BaseForecaster] = None
        step_idx = start_step_idx

        # Bridge-fit: if starting mid-schedule, fit once on the window the
        # last refit at-or-before start_step_idx would have used, so the
        # first computed step sees a model in the same state a continuous
        # run would have had. Skipped when that refit point IS
        # start_step_idx itself — the loop's own should_refit (model is
        # None) already handles that case identically, with no extra fit.
        if start_step_idx > 0:
            last_refit_step = (
                (start_step_idx // self.refit_every) * self.refit_every
                if self.refit_every > 0 else 0
            )
            if last_refit_step < start_step_idx:
                bridge_train_end = train_size + last_refit_step * self.step
                bridge_train_start = (
                    0 if self.window_type == "expanding"
                    else (bridge_train_end - train_size)
                )
                y_bridge = y.iloc[bridge_train_start:bridge_train_end]
                logger.info(
                    f"Bridge-fit at step {last_refit_step} (for start_step_idx="
                    f"{start_step_idx}): {y_bridge.index[0].date()} → "
                    f"{y_bridge.index[-1].date()} ({len(y_bridge)} obs)"
                )
                model = self.model_class(**self.model_params)
                model.fit(y_bridge, **self.fit_kwargs)

        while True:
            train_end = train_size + step_idx * self.step
            if train_end >= n:
                break
            if end_step_idx is not None and step_idx >= end_step_idx:
                break

            train_start = 0 if self.window_type == "expanding" else (train_end - train_size)
            forecast_end = min(train_end + self.horizon, n)
            current_horizon = forecast_end - train_end

            if current_horizon <= 0:
                break

            y_train = y.iloc[train_start:train_end]
            y_val   = y.iloc[train_end:forecast_end]

            should_refit = (
                model is None
                or (self.refit_every > 0 and step_idx % self.refit_every == 0)
            )

            if should_refit:
                logger.info(
                    f"Step {step_idx}: refitting on "
                    f"{y_train.index[0].date()} → {y_train.index[-1].date()} "
                    f"({len(y_train)} obs)"
                )
                model = self.model_class(**self.model_params)
                model.fit(y_train, **self.fit_kwargs)

            forecast = np.asarray(
                model.predict(y_train, steps=current_horizon)
            ).flatten()

            # Write predictions into the DataFrame
            for j, pred_val in enumerate(forecast):
                h = j + 1
                date = y.index[train_end + j]
                pred_df.loc[date, f"pred_h{h}"] = pred_val

            step_metrics = {}
            try:
                step_metrics = compute_all_metrics(
                    y_val.values, forecast, warn_on_nan=False
                )
            except Exception as exc:
                logger.warning(f"Step {step_idx} metrics failed: {exc}")

            step_results.append({
                "step":               step_idx,
                "train_start":        str(y_train.index[0].date()),
                "train_end":          str(y_train.index[-1].date()),
                "forecast_start":     str(y_val.index[0].date()),
                "forecast_end":       str(y_val.index[-1].date()),
                "n_train":            len(y_train),
                "dates":              [str(d.date()) for d in y_val.index],
                "actuals":            y_val.values.tolist(),
                "predictions":        forecast.tolist(),
                "metrics":            step_metrics,
            })

            step_idx += 1

        # Aggregate metrics per horizon
        metrics_by_horizon: Dict[int, Dict] = {}
        for h in range(1, self.horizon + 1):
            col = f"pred_h{h}"
            valid = pred_df[["actual", col]].dropna()
            if len(valid) == 0:
                metrics_by_horizon[h] = {}
                continue
            try:
                m = compute_all_metrics(
                    valid["actual"].values, valid[col].values, warn_on_nan=False
                )
                metrics_by_horizon[h] = m
                logger.info(
                    f"h={h} overall ({len(valid)} evals): "
                    f"MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}"
                )
            except Exception as exc:
                logger.error(f"Aggregate metrics for h={h} failed: {exc}")
                metrics_by_horizon[h] = {}

        # Convenience: aggregate across h=1 predictions for summary
        h1 = metrics_by_horizon.get(1, {})

        return {
            "metrics_by_horizon": metrics_by_horizon,
            "aggregate_metrics":  h1,
            "step_results":       step_results,
            "predictions":        pred_df,
            "n_steps":            step_idx,
            "model_params":       model.get_params() if model is not None else {},
            "config": {
                "horizon":      self.horizon,
                "step":         self.step,
                "window_type":  self.window_type,
                "train_size":   train_size,
                "refit_every":  self.refit_every,
            },
        }

    def resolve_train_size(self, y: pd.Series) -> int:
        """Public wrapper around _resolve_train_size() — lets orchestration
        code (e.g. pneumonia.pipelines.walkforward_runner) compute the same
        step_idx <-> date mapping this validator uses internally, without
        duplicating the int/date-resolution logic, before calling run()
        with a specific step_range."""
        return self._resolve_train_size(y)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_train_size(self, y: pd.Series) -> int:
        if isinstance(self.initial_train_size, (str, pd.Timestamp)):
            dt = pd.to_datetime(self.initial_train_size)
            return int((y.index <= dt).sum())
        return int(self.initial_train_size)
