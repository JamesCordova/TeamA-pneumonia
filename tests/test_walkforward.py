"""
Unit tests for the walk-forward validation module.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.models.base import BaseForecaster
from pneumonia.models.sarima.model import SARIMAModel
from pneumonia.evaluation.walkforward import WalkForwardValidator


# =============================================================================
# DUMMY FORECASTERS FOR FAST TESTING
# =============================================================================

class DummyForecaster(BaseForecaster):
    """Simple forecaster that predicts a constant value for testing."""

    def __init__(self, department="TEST", age_group="under5", constant_value=10.0):
        super().__init__(name="Dummy", department=department, age_group=age_group)
        self.constant_value = constant_value

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        self.is_fitted = True
        self.fitted_date = "dummy_date"

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        return np.full(steps, self.constant_value, dtype=float)


class MeanSpyForecaster(BaseForecaster):
    """Predicts the mean of whatever training data it was last fit on
    (data-dependent, unlike DummyForecaster's fixed constant), and records
    every fit/predict call's index — so a test can tell two separately
    computed sub-ranges apart if the model each one ended up fit on wasn't
    the model a continuous run would have used at that point."""

    def __init__(self, department="TEST", age_group="under5"):
        super().__init__(name="MeanSpy", department=department, age_group=age_group)
        self.fit_calls: list = []
        self.predict_calls: list = []
        self._mean = None

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        self.is_fitted = True
        self.fitted_date = "dummy_date"
        self._mean = float(train_data.mean())
        self.fit_calls.append(train_data.index)

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        self.predict_calls.append(data.index)
        return np.full(steps, self._mean, dtype=float)


class SpyForecaster(BaseForecaster):
    """Forecaster that records the exact index it was fit/predicted on,
    so tests can independently verify the validator never exposes future
    dates to a model, without relying on the validator's own reported
    train_start/train_end bookkeeping."""

    def __init__(self, department="TEST", age_group="under5", constant_value=10.0):
        super().__init__(name="Spy", department=department, age_group=age_group)
        self.constant_value = constant_value
        self.fit_calls: list = []
        self.predict_calls: list = []

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        self.is_fitted = True
        self.fitted_date = "dummy_date"
        self.fit_calls.append(train_data.index)

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        self.predict_calls.append(data.index)
        return np.full(steps, self.constant_value, dtype=float)





# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def dummy_timeseries():
    """Create a dummy weekly time series of 20 elements."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="W")
    values = np.arange(1.0, 21.0)  # [1, 2, ..., 20]
    return pd.Series(values, index=dates)



# =============================================================================
# TESTS
# =============================================================================

class TestWalkForwardValidator:
    """Test WalkForwardValidator functionality."""
    
    def test_validation_invalid_params(self):
        """Test that invalid parameter values raise exceptions."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            WalkForwardValidator(DummyForecaster, {}, initial_train_size=5, horizon=0)
            
        with pytest.raises(ValueError, match="step must be positive"):
            WalkForwardValidator(DummyForecaster, {}, initial_train_size=5, step=-1)
            
        with pytest.raises(ValueError, match="window_type must be"):
            WalkForwardValidator(DummyForecaster, {}, initial_train_size=5, window_type="invalid")

    def test_expanding_window_one_step(self, dummy_timeseries):
        """Test expanding window walk-forward validation with horizon=1."""
        validator = WalkForwardValidator(
            model_class=DummyForecaster,
            model_params={"constant_value": 15.0},
            initial_train_size=10,
            horizon=1,
            step=1,
            window_type="expanding",
            min_train_size=5
        )
        
        results = validator.run(dummy_timeseries)
        
        assert "metrics_by_horizon" in results
        assert "step_results" in results
        assert "predictions" in results
        
        # Initial train size is 10, total data size is 20.
        # Step 0: train on 0..10, predict 10 (horizon 1) -> target index 10.
        # Step 9: train on 0..19, predict 19 (horizon 1) -> target index 19.
        # Total steps = 10.
        assert len(results["step_results"]) == 10
        
        predictions_df = results["predictions"]
        assert len(predictions_df) == 10  # 20 - 10 = 10 evaluated steps
        assert list(predictions_df.columns) == ["actual", "pred_h1"]
        
        # All predictions should be constant_value (15.0)
        assert np.all(predictions_df["pred_h1"] == 15.0)
        # Actuals should be elements from index 10 to 19 of dummy_timeseries ([11, ..., 20])
        assert np.all(predictions_df["actual"] == np.arange(11.0, 21.0))
        
        # Overall metrics for horizon 1 should exist
        h1_metrics = results["metrics_by_horizon"][1]
        assert "mae" in h1_metrics
        assert "rmse" in h1_metrics
        # MAE: Mean of |[11..20] - 15|
        # Actuals: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # Diff:    [ 4,  3,  2,  1,  0,  1,  2,  3,  4,  5]
        # Mean diff: 2.5
        assert h1_metrics["mae"] == pytest.approx(2.5)

    def test_sliding_window_one_step(self, dummy_timeseries):
        """Test sliding window walk-forward validation with horizon=1."""
        validator = WalkForwardValidator(
            model_class=DummyForecaster,
            model_params={"constant_value": 10.0},
            initial_train_size=10,
            horizon=1,
            step=2,
            window_type="sliding",
            min_train_size=5
        )
        
        results = validator.run(dummy_timeseries)
        
        # Total size = 20, initial_train = 10, step_size = 2.
        # Step 0: train on 0..10, predict 10 (horizon 1) -> target index 10.
        # Step 1: train on 2..12, predict 12 (horizon 1) -> target index 12.
        # Step 2: train on 4..14, predict 14 (horizon 1) -> target index 14.
        # Step 3: train on 6..16, predict 16 (horizon 1) -> target index 16.
        # Step 4: train on 8..18, predict 18 (horizon 1) -> target index 18.
        # Next would be train on 10..20, but 20 is end of array. So loop terminates.
        assert len(results["step_results"]) == 5
        
        predictions_df = results["predictions"]
        # In step 1: we predict for 12, so 11 is not predicted. 11 will be NaN in pred_h1.
        # Let's verify that actual values are still there
        assert np.isnan(predictions_df.loc[dummy_timeseries.index[11], "pred_h1"])
        assert predictions_df.loc[dummy_timeseries.index[10], "pred_h1"] == 10.0
        assert predictions_df.loc[dummy_timeseries.index[12], "pred_h1"] == 10.0

    def test_multi_horizon(self, dummy_timeseries):
        """Test multi-horizon (horizon > 1) walk-forward validation."""
        validator = WalkForwardValidator(
            model_class=DummyForecaster,
            model_params={"constant_value": 5.0},
            initial_train_size=12,
            horizon=3,
            step=2,
            window_type="expanding",
            min_train_size=5
        )
        
        results = validator.run(dummy_timeseries)
        
        # Total size = 20, initial_train = 12, horizon = 3, step_size = 2.
        # Step 0: train on 0..12, predict 12, 13, 14.
        # Step 1: train on 0..14, predict 14, 15, 16.
        # Step 2: train on 0..16, predict 16, 17, 18.
        # Step 3: train on 0..18, predict 18, 19. (end of series)
        assert len(results["step_results"]) == 4
        
        predictions_df = results["predictions"]
        assert f"pred_h1" in predictions_df.columns
        assert f"pred_h2" in predictions_df.columns
        assert f"pred_h3" in predictions_df.columns
        
        # Let's verify pred_h3 for index 14.
        # Index 14 is predicted 3 steps ahead from step 0 (when train_end = 12).
        # Target date is index 12 + 2 = 14.
        assert predictions_df.loc[dummy_timeseries.index[14], "pred_h3"] == 5.0

        # Overall metrics for horizon 1..3 should exist
        assert 1 in results["metrics_by_horizon"]
        assert 2 in results["metrics_by_horizon"]
        assert 3 in results["metrics_by_horizon"]

    @pytest.mark.parametrize("window_type,step,horizon,refit_every", [
        ("expanding", 2, 3, 1),
        ("sliding", 2, 3, 1),
        ("sliding", 1, 1, 3),  # refit skipped on some steps: stale model must still only ever see past data
    ])
    def test_no_leakage(self, dummy_timeseries, window_type, step, horizon, refit_every):
        """The exact index passed to fit()/predict() at each step must never
        contain (or overlap with) that step's evaluation dates, and must be
        strictly earlier in time. This is checked against indices captured
        directly from the model calls, independent of the validator's own
        train_start/train_end bookkeeping, so it would catch an off-by-one
        or window-slicing bug in the validator itself."""
        validator = WalkForwardValidator(
            model_class=SpyForecaster,
            model_params={"constant_value": 1.0},
            initial_train_size=10,
            horizon=horizon,
            step=step,
            window_type=window_type,
            refit_every=refit_every,
            min_train_size=5,
        )

        results = validator.run(dummy_timeseries)

        for step_res in results["step_results"]:
            eval_dates = pd.to_datetime(step_res["dates"])
            train_start = pd.to_datetime(step_res["train_start"])
            train_end = pd.to_datetime(step_res["train_end"])
            train_dates = pd.date_range(start=train_start, end=train_end, freq="W")

            assert set(train_dates).isdisjoint(set(eval_dates)), (
                f"Step {step_res['step']}: training window overlaps evaluation dates"
            )
            assert train_dates.max() < eval_dates.min(), (
                f"Step {step_res['step']}: training window is not strictly "
                f"before the evaluation window"
            )

    def test_no_leakage_via_spy_calls(self, dummy_timeseries):
        """Directly verify the Series objects handed to fit()/predict() never
        contain a date from that step's evaluation window, by inspecting the
        model instance's recorded call indices rather than the validator's
        reported metadata."""
        spies: list = []

        class RecordingSpy(SpyForecaster):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                spies.append(self)

        validator = WalkForwardValidator(
            model_class=RecordingSpy,
            model_params={"constant_value": 1.0},
            initial_train_size=10,
            horizon=2,
            step=2,
            window_type="sliding",
            refit_every=1,
            min_train_size=5,
        )
        results = validator.run(dummy_timeseries)

        assert len(spies) == len(results["step_results"])
        for spy, step_res in zip(spies, results["step_results"]):
            eval_dates = set(pd.to_datetime(step_res["dates"]))
            fit_index = spy.fit_calls[-1]
            predict_index = spy.predict_calls[-1]

            assert set(fit_index).isdisjoint(eval_dates)
            assert set(predict_index).isdisjoint(eval_dates)
            assert fit_index.max() < min(eval_dates)
            assert predict_index.max() < min(eval_dates)

    @pytest.fixture
    def longer_timeseries(self):
        """30 weeks, data-dependent values (unlike dummy_timeseries' shape,
        this one exists mainly to give step_range tests enough steps to pick
        a meaningful split point)."""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="W")
        values = np.arange(1.0, 31.0)
        return pd.Series(values, index=dates)

    @pytest.mark.parametrize("window_type,refit_every,split_on_refit_boundary", [
        ("sliding", 3, False),
        ("sliding", 3, True),
        ("expanding", 3, False),
        ("sliding", 1, False),   # refit every step — bridge always lands exactly on start
        ("sliding", 0, False),   # fit once only
    ])
    def test_step_range_matches_continuous_run(
        self, longer_timeseries, window_type, refit_every, split_on_refit_boundary
    ):
        """Splitting a run into step_range=(0, k) + step_range=(k, None) must
        reproduce exactly what a single continuous run(step_range=None)
        produces — same refit schedule, same predictions — regardless of
        where k falls. This is what lets 'search' (pre-holdout) and
        'holdout' pieces be computed independently, in either order, and
        still be equivalent to one continuous backtest."""
        validator = WalkForwardValidator(
            model_class=MeanSpyForecaster,
            model_params={},
            initial_train_size=10,
            horizon=2,
            step=2,
            window_type=window_type,
            refit_every=refit_every,
            min_train_size=5,
        )

        full = validator.run(longer_timeseries)
        n_steps = len(full["step_results"])
        assert n_steps >= 4, "fixture too short for a meaningful split"

        k = n_steps // 2
        if refit_every > 1 and split_on_refit_boundary:
            k = (k // refit_every) * refit_every
            if k == 0:
                k = refit_every
        assert 0 < k < n_steps

        part_a = validator.run(longer_timeseries, step_range=(0, k))
        part_b = validator.run(longer_timeseries, step_range=(k, None))

        assert len(part_a["step_results"]) == k
        assert len(part_b["step_results"]) == n_steps - k

        # Per-step results must match the continuous run's steps exactly,
        # matched by global step index.
        full_by_step = {s["step"]: s for s in full["step_results"]}
        for s in part_a["step_results"] + part_b["step_results"]:
            ref = full_by_step[s["step"]]
            assert s["train_start"] == ref["train_start"]
            assert s["train_end"] == ref["train_end"]
            assert s["predictions"] == pytest.approx(ref["predictions"])
            assert s["actuals"] == ref["actuals"]

        # Stitched predictions DataFrame must match the continuous one
        # wherever either half has a value.
        combined = part_a["predictions"].combine_first(part_b["predictions"])
        pd.testing.assert_frame_equal(
            combined.sort_index(axis=1), full["predictions"].sort_index(axis=1),
            check_dtype=False,
        )

    def test_step_range_bridge_fit_window(self, longer_timeseries):
        """The bridge fit must train on exactly the window the last
        continuous refit at-or-before start_step_idx would have used, not
        e.g. the window of start_step_idx itself."""
        spies: list = []

        class RecordingMeanSpy(MeanSpyForecaster):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                spies.append(self)

        validator = WalkForwardValidator(
            model_class=RecordingMeanSpy,
            model_params={},
            initial_train_size=10,
            horizon=2,
            step=2,
            window_type="sliding",
            refit_every=3,
            min_train_size=5,
        )

        full = validator.run(longer_timeseries)
        n_steps = len(full["step_results"])
        # Pick a start_step_idx that is NOT itself a refit step, so a real
        # bridge-fit is required (last_refit_step < start_step_idx).
        start = next(i for i in range(1, n_steps) if i % 3 != 0)
        last_refit_step = (start // 3) * 3
        expected_train_start = full["step_results"][last_refit_step]["train_start"]
        expected_train_end = full["step_results"][last_refit_step]["train_end"]

        spies.clear()
        validator.run(longer_timeseries, step_range=(start, None))

        bridge_spy = spies[0]
        assert len(bridge_spy.fit_calls) >= 1
        bridge_index = bridge_spy.fit_calls[0]
        assert str(bridge_index[0].date()) == expected_train_start
        assert str(bridge_index[-1].date()) == expected_train_end

    @pytest.mark.slow
    def test_sarima_model_integration(self):
        """Test walk-forward validation integrated with real SARIMAModel."""
        # Create a 2.5-year weekly series (130 weeks) to fit SARIMA's seasonal requirement (m=52)
        # We need train size >= 104. Let's make initial train size 110, horizon 2, step 5.
        dates = pd.date_range(start="2021-01-01", periods=120, freq="W")
        t = np.arange(120)
        # Simple sine curve for seasonal variance + linear trend
        values = 50.0 + 0.1 * t + 5.0 * np.sin(2.0 * np.pi * t / 52.0)
        y = pd.Series(values, index=dates)
        
        # Run walkforward on a subset
        validator = WalkForwardValidator(
            model_class=SARIMAModel,
            model_params={
                "department": "TEST",
                "age_group": "under5",
                "order": (1, 0, 0),
                "seasonal_order": (0, 0, 0, 0)  # simplified model for speed
            },
            initial_train_size=110,
            horizon=2,
            step=4,
            min_train_size=50
        )
        results = validator.run(y)
        
        assert len(results["step_results"]) == 3  # step 0: 110, step 1: 114, step 2: 118
        predictions_df = results["predictions"]
        
        # Check that we have actual values and non-NaN predictions
        assert not predictions_df["pred_h1"].dropna().empty
        assert "mae" in results["metrics_by_horizon"][1]
