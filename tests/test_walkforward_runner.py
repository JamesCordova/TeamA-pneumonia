"""
Unit tests for pneumonia.pipelines.walkforward_runner's pure date-arithmetic
helpers (_n_steps_total, _holdout_boundary_step) — no DB, no model fitting.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.pipelines.walkforward_runner import _holdout_boundary_step, _n_steps_total


class TestNStepsTotal:
    def test_matches_validator_step_counts(self):
        """Must agree with WalkForwardValidator's own loop termination for
        the same (n, train_size, step) — cross-checked against the existing
        test_walkforward.py fixtures' known step counts."""
        assert _n_steps_total(n=20, train_size=10, step=1) == 10
        assert _n_steps_total(n=20, train_size=10, step=2) == 5
        assert _n_steps_total(n=20, train_size=12, step=2) == 4

    def test_train_size_at_or_above_series_length(self):
        assert _n_steps_total(n=20, train_size=20, step=1) == 0
        assert _n_steps_total(n=20, train_size=25, step=1) == 0


class TestHoldoutBoundaryStep:
    @pytest.fixture
    def series(self):
        return pd.Series(
            range(30), index=pd.date_range("2023-01-01", periods=30, freq="W")
        )

    def test_boundary_before_training_ends(self, series):
        """holdout_start inside (or before) the training window -> boundary
        is step 0: the entire evaluated range is 'holdout'."""
        k = _holdout_boundary_step(series, train_size=10, step=2, holdout_start=series.index[3])
        assert k == 0

    def test_boundary_exactly_on_a_step(self, series):
        # train_size=10, step=2 -> eval dates at positions 10, 12, 14, ...
        # holdout_start = position 14 -> that step's train_end (14) already
        # reaches it, so k should land exactly on step index 2.
        holdout_start = series.index[14]
        k = _holdout_boundary_step(series, train_size=10, step=2, holdout_start=holdout_start)
        assert k == 2

    def test_boundary_between_steps_rounds_up(self, series):
        # position 15 falls between step 2 (train_end=14) and step 3
        # (train_end=16) -> must round UP to 3, since step 2's forecast
        # would otherwise reach into the holdout at position 14..15.
        holdout_start = series.index[15]
        k = _holdout_boundary_step(series, train_size=10, step=2, holdout_start=holdout_start)
        assert k == 3

    def test_boundary_after_all_data(self, series):
        k = _holdout_boundary_step(
            series, train_size=10, step=2, holdout_start=series.index[-1] + pd.Timedelta(weeks=10)
        )
        n_total = _n_steps_total(n=len(series), train_size=10, step=2)
        assert k >= n_total  # nothing left for a 'holdout' range

    def test_consistent_with_full_run_evaluation_dates(self, series):
        """The date at step k's train_end must be >= holdout_start, and the
        date at step (k-1)'s train_end must be < holdout_start (when k>0) —
        i.e. k really is the first step whose window starts at/after the
        holdout, cross-checked against the raw date arithmetic directly."""
        train_size, step = 10, 3
        holdout_start = series.index[17]
        k = _holdout_boundary_step(series, train_size, step, holdout_start)

        train_end_k = train_size + k * step
        assert series.index[train_end_k] >= holdout_start if train_end_k < len(series) else True
        if k > 0:
            train_end_prev = train_size + (k - 1) * step
            assert series.index[train_end_prev] < holdout_start
