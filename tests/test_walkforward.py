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
from pneumonia.evaluation.walkforward import WalkForwardValidator, walkforward_validation


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
        
    def predict(self, steps: int) -> np.ndarray:
        return np.full(steps, self.constant_value, dtype=float)


class DummyMLForecaster(BaseForecaster):
    """Dummy ML forecaster that expects features for testing."""
    
    def __init__(self, department="TEST", age_group="under5"):
        super().__init__(name="DummyML", department=department, age_group=age_group)
        
    def fit(self, train_data: pd.Series, train_features: np.ndarray, **kwargs) -> None:
        self.is_fitted = True
        self.fitted_date = "dummy_date"
        self.train_features_shape = train_features.shape
        
    def predict(self, steps: int, test_features: np.ndarray) -> np.ndarray:
        # Predict the mean of test features along columns as a dummy prediction
        return np.mean(test_features, axis=1)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def dummy_timeseries():
    """Create a dummy weekly time series of 20 elements."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="W")
    values = np.arange(1.0, 21.0)  # [1, 2, ..., 20]
    return pd.Series(values, index=dates)


@pytest.fixture
def dummy_features():
    """Create dummy features with same index length of 20."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="W")
    return pd.DataFrame({
        "feat1": np.arange(10.0, 30.0),
        "feat2": np.arange(20.0, 40.0)
    }, index=dates)


# =============================================================================
# TESTS
# =============================================================================

class TestWalkForwardValidator:
    """Test WalkForwardValidator functionality."""
    
    def test_validation_invalid_params(self):
        """Test that invalid parameter values raise exceptions."""
        with pytest.raises(ValueError, match="horizon must be positive"):
            WalkForwardValidator(DummyForecaster, {}, initial_train_size=5, horizon=0)
            
        with pytest.raises(ValueError, match="step_size must be positive"):
            WalkForwardValidator(DummyForecaster, {}, initial_train_size=5, step_size=-1)
            
        with pytest.raises(ValueError, match="window_type must be"):
            WalkForwardValidator(DummyForecaster, {}, initial_train_size=5, window_type="invalid")

    def test_expanding_window_one_step(self, dummy_timeseries):
        """Test expanding window walk-forward validation with horizon=1."""
        validator = WalkForwardValidator(
            model_class=DummyForecaster,
            model_params={"constant_value": 15.0},
            initial_train_size=10,
            horizon=1,
            step_size=1,
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
            step_size=2,
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
            step_size=2,
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

    def test_ml_features_integration(self, dummy_timeseries, dummy_features):
        """Test that feature matrix X is properly aligned and sliced for ML forecasters."""
        results = walkforward_validation(
            data=dummy_timeseries,
            X=dummy_features,
            model_class=DummyMLForecaster,
            model_params={},
            initial_train_size=10,
            horizon=2,
            step_size=2,
            min_train_size=5
        )
        
        assert "metrics_by_horizon" in results
        assert len(results["step_results"]) == 5
        
        predictions_df = results["predictions"]
        # Predictions are the mean of test features.
        # At step 0: train 0..10. Predict 10, 11.
        # Test features at step 10: feat1=20.0, feat2=30.0 -> mean is 25.0
        # Test features at step 11: feat1=21.0, feat2=31.0 -> mean is 26.0
        assert predictions_df.loc[dummy_timeseries.index[10], "pred_h1"] == pytest.approx(25.0)
        assert predictions_df.loc[dummy_timeseries.index[11], "pred_h2"] == pytest.approx(26.0)

    @pytest.mark.slow
    def test_sarima_model_integration(self):
        """Test walk-forward validation integrated with real SARIMAModel."""
        # Create a 2.5-year weekly series (130 weeks) to fit SARIMA's seasonal requirement (m=52)
        # We need train size >= 104. Let's make initial train size 110, horizon 2, step_size 5.
        dates = pd.date_range(start="2021-01-01", periods=120, freq="W")
        t = np.arange(120)
        # Simple sine curve for seasonal variance + linear trend
        values = 50.0 + 0.1 * t + 5.0 * np.sin(2.0 * np.pi * t / 52.0)
        y = pd.Series(values, index=dates)
        
        # Run walkforward on a subset
        results = walkforward_validation(
            data=y,
            model_class=SARIMAModel,
            model_params={
                "department": "TEST",
                "age_group": "under5",
                "order": (1, 0, 0),
                "seasonal_order": (0, 0, 0, 0)  # simplified model for speed
            },
            initial_train_size=110,
            horizon=2,
            step_size=4,
            min_train_size=50
        )
        
        assert len(results["step_results"]) == 3  # step 0: 110, step 1: 114, step 2: 118
        predictions_df = results["predictions"]
        
        # Check that we have actual values and non-NaN predictions
        assert not predictions_df["pred_h1"].dropna().empty
        assert "mae" in results["metrics_by_horizon"][1]
