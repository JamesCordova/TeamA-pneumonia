"""
Pytest fixtures and tests for SARIMA pipeline

Run with: pytest tests/test_sarima_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.models.utils import temporal_split, handle_missing_values
from pneumonia.models.sarima.model import SARIMAModel
from pneumonia.evaluation.metrics import (
    compute_all_metrics,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    mean_error,
    r2_score
)
from pneumonia.pipelines.sarima_pipeline import SARIMAPipeline


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_timeseries():
    """Create a sample time series with seasonal pattern."""
    # Create 2 years of weekly data (104 weeks)
    dates = pd.date_range(start="2022-01-01", periods=104, freq="W")
    # Add seasonal pattern: higher in winter
    t = np.arange(104)
    trend = np.linspace(50, 60, 104)
    seasonal = 10 * np.sin(2 * np.pi * t / 52)
    noise = np.random.normal(0, 2, 104)
    values = trend + seasonal + noise
    values = np.maximum(values, 0)  # Ensure non-negative
    
    ts = pd.Series(values, index=dates)
    return ts


@pytest.fixture
def sample_timeseries_long():
    """Create a longer time series with 5 years of data."""
    dates = pd.date_range(start="2019-01-01", periods=260, freq="W")
    t = np.arange(260)
    trend = np.linspace(50, 70, 260)
    seasonal = 10 * np.sin(2 * np.pi * t / 52)
    noise = np.random.normal(0, 2, 260)
    values = trend + seasonal + noise
    values = np.maximum(values, 0)
    
    ts = pd.Series(values, index=dates)
    return ts


@pytest.fixture
def sample_predictions():
    """Create sample actual vs predicted values."""
    actual = np.array([10, 15, 12, 18, 20, 17, 22, 25, 23, 20])
    predicted = np.array([11, 14, 13, 17, 21, 16, 23, 24, 24, 19])
    return actual, predicted


# =============================================================================
# TESTS: Data Processing
# =============================================================================

class TestTemporalSplit:
    """Test temporal split functionality."""
    
    def test_temporal_split_dynamic(self, sample_timeseries_long):
        """Test dynamic temporal split (ratio-based)."""
        train, val, test = temporal_split(
            sample_timeseries_long,
            strategy="dynamic",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        
        # Check sizes
        assert len(train) == 156  # 60% of 260
        assert len(val) == 52     # 20% of 260
        assert len(test) == 52    # 20% of 260
        
        # Check continuity
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]
    
    def test_temporal_split_years(self, sample_timeseries_long):
        """Test year-based temporal split."""
        train, val, test = temporal_split(
            sample_timeseries_long,
            strategy="years",
            train_years=(2019, 2021),
            val_years=(2022, 2022),
            test_years=(2022, 2023),
        )
        
        # Should have non-empty splits
        assert len(train) > 0
        assert len(val) >= 0  # May be empty
        assert len(test) > 0
    
    def test_temporal_split_invalid_ratios(self, sample_timeseries):
        """Test that invalid ratios raise error."""
        with pytest.raises(ValueError):
            temporal_split(
                sample_timeseries,
                strategy="dynamic",
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.2,  # Sum != 1.0
            )
    
    def test_temporal_split_insufficient_data(self, sample_timeseries):
        """Test that insufficient training data raises error."""
        short_ts = sample_timeseries[:50]  # Less than 104 weeks
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            temporal_split(short_ts, strategy="dynamic")


class TestMissingValues:
    """Test missing value handling."""
    
    def test_handle_missing_interpolate(self):
        """Test interpolation method for missing values."""
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        values = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10], index=dates)
        
        result = handle_missing_values(values, method="interpolate")
        
        # Check no NaNs remaining
        assert result.isna().sum() == 0
        # Check interpolated values are reasonable
        assert result.iloc[2] == 3  # Linear interpolation between 2 and 4
    
    def test_handle_missing_forward_fill(self):
        """Test forward fill method."""
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        values = pd.Series([1, 2, np.nan, np.nan, 5], index=dates)
        
        result = handle_missing_values(values, method="forward_fill")
        
        assert result.isna().sum() == 0
        assert result.iloc[2] == 2
        assert result.iloc[3] == 2


# =============================================================================
# TESTS: Metrics
# =============================================================================

class TestMetrics:
    """Test evaluation metrics."""
    
    def test_mae(self, sample_predictions):
        """Test Mean Absolute Error."""
        actual, predicted = sample_predictions
        mae = mean_absolute_error(actual, predicted)
        
        # Expected: mean of |[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]| = 1.0
        assert mae == pytest.approx(1.0)
    
    def test_rmse(self, sample_predictions):
        """Test Root Mean Squared Error."""
        actual, predicted = sample_predictions
        rmse = root_mean_squared_error(actual, predicted)
        
        # Expected: sqrt(mean of [1, 1, 1, ...]) = 1.0
        assert rmse == pytest.approx(1.0)
    
    def test_mape_normal(self, sample_predictions):
        """Test MAPE with normal data."""
        actual, predicted = sample_predictions
        mape = mean_absolute_percentage_error(actual, predicted)
        
        # Should return a positive percentage
        assert not np.isnan(mape)
        assert mape >= 0
    
    def test_mape_with_zeros(self):
        """Test MAPE when actual values are near zero."""
        actual = np.array([0, 0.000001, 0.000002, 0.000005])
        predicted = np.array([0.5, 0.5, 0.5, 0.5])
        
        mape = mean_absolute_percentage_error(actual, predicted, epsilon=1e-5)
        
        # Should return NaN with warning, not crash
        assert np.isnan(mape)
    
    def test_compute_all_metrics(self, sample_predictions):
        """Test computing all metrics at once."""
        actual, predicted = sample_predictions
        metrics = compute_all_metrics(actual, predicted)
        
        # Check all metrics present
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "smape" in metrics
        assert "mda" in metrics
        assert "me" in metrics
        assert "r2" in metrics
        
        # Check values are numeric
        for metric_name, value in metrics.items():
            assert isinstance(value, (int, float, type(np.nan)))
            
    def test_me(self, sample_predictions):
        """Test Mean Error."""
        actual, predicted = sample_predictions
        me = mean_error(actual, predicted)
        # Expected: mean of [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1] = 0.0
        assert me == pytest.approx(0.0)

    def test_r2(self, sample_predictions):
        """Test R2 Score."""
        actual, predicted = sample_predictions
        r2 = r2_score(actual, predicted)
        assert isinstance(r2, float)
        assert r2 <= 1.0
    
    def test_metrics_shape_mismatch(self, sample_predictions):
        """Test that mismatched shapes raise error."""
        actual, _ = sample_predictions
        predicted = np.array([1, 2, 3])  # Different length
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_all_metrics(actual, predicted)


# =============================================================================
# TESTS: SARIMA Model
# =============================================================================

class TestSARIMAModel:
    """Test SARIMA model functionality."""
    
    def test_model_initialization(self):
        """Test SARIMA model initialization."""
        model = SARIMAModel(
            department="TEST",
            age_group="under5",
        )
        
        assert model.department == "TEST"
        assert model.age_group == "under5"
        assert not model.is_fitted
        assert model.order == (1, 1, 1)
        assert model.seasonal_order == (1, 1, 1, 52)
    
    def test_model_fit_and_predict(self, sample_timeseries_long):
        """Test fitting and predicting with SARIMA."""
        model = SARIMAModel(
            department="TEST",
            age_group="under5",
        )
        
        # Fit
        model.fit(sample_timeseries_long)
        assert model.is_fitted
        assert model.results is not None

        # Predict
        forecast = model.predict(sample_timeseries_long, steps=10)
        assert len(forecast) == 10
    
    def test_model_get_forecast_interval(self, sample_timeseries_long):
        """Test getting forecast with confidence intervals."""
        model = SARIMAModel(
            department="TEST",
            age_group="under5",
        )
        
        model.fit(sample_timeseries_long)
        pred, lower, upper = model.get_forecast_interval(sample_timeseries_long, steps=10, alpha=0.05)
        
        assert len(pred) == 10
        assert len(lower) == 10
        assert len(upper) == 10
        assert np.all(lower <= upper)
    
    def test_model_diagnostics(self, sample_timeseries_long):
        """Test model diagnostics."""
        model = SARIMAModel(
            department="TEST",
            age_group="under5",
        )
        
        model.fit(sample_timeseries_long)
        diag = model.diagnostics()
        
        assert "aic" in diag
        assert "bic" in diag
        assert "residuals" in diag
        assert diag["residuals"]["mean"] is not None


# =============================================================================
# TESTS: Pipeline
# =============================================================================

class TestSARIMAPipeline:
    """Test SARIMA training pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = SARIMAPipeline(
            department="TEST",
            age_group="under5",
        )
        
        assert pipeline.department == "TEST"
        assert pipeline.age_group == "under5"
        assert pipeline.model is None
        assert "status" not in pipeline.results
    
    def test_pipeline_with_mock_data(self, sample_timeseries_long, monkeypatch):
        """Test pipeline with mocked data loading."""
        def mock_get_departmental_data(dept, age_group):
            return sample_timeseries_long
        
        # Patch the data loading function
        import pneumonia.pipelines.sarima_pipeline as pipeline_module
        monkeypatch.setattr(
            pipeline_module,
            "get_departmental_data",
            mock_get_departmental_data
        )
        
        pipeline = SARIMAPipeline(
            department="TEST",
            age_group="under5",
        )
        
        results = pipeline.run()
        
        # Check results structure
        assert "stages" in results
        assert "data_loading" in results["stages"]
        assert "training" in results["stages"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_pipeline(self, sample_timeseries_long):
        """Test complete pipeline from data to evaluation."""
        # Split data
        train, val, test = temporal_split(sample_timeseries_long, strategy="dynamic")
        
        # Train model
        model = SARIMAModel(department="TEST", age_group="under5")
        model.fit(train)
        
        # Validate
        val_forecast = model.predict(train, steps=len(val))
        val_metrics = compute_all_metrics(val.values, val_forecast)

        # Test
        test_forecast = model.predict(pd.concat([train, val]), steps=len(test))
        test_metrics = compute_all_metrics(test.values, test_forecast)
        
        # Check metrics
        for metrics in [val_metrics, test_metrics]:
            assert "mae" in metrics
            assert metrics["mae"] > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests."""
    
    def test_model_fit_speed(self, sample_timeseries_long):
        """Test that model fitting completes in reasonable time."""
        import time
        
        model = SARIMAModel(department="PERF_TEST", age_group="under5")
        
        start = time.time()
        model.fit(sample_timeseries_long)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (adjust as needed)
        assert elapsed < 60, f"Model fitting took {elapsed:.1f}s, expected < 60s"
    
    def test_large_forecast(self, sample_timeseries_long):
        """Test forecasting large number of steps."""
        model = SARIMAModel(department="PERF_TEST", age_group="under5")
        model.fit(sample_timeseries_long)

        forecast = model.predict(sample_timeseries_long, steps=104)

        assert len(forecast) == 104


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.parametrize("age_group", ["under5", "60plus"])
def test_sarima_both_age_groups(age_group, sample_timeseries_long):
    """Test SARIMA model works for both age groups."""
    model = SARIMAModel(
        department="TEST",
        age_group=age_group,
    )
    
    model.fit(sample_timeseries_long)
    forecast = model.predict(sample_timeseries_long, steps=10)

    assert len(forecast) == 10


@pytest.mark.parametrize("strategy", ["dynamic", "years"])
def test_split_strategies(strategy, sample_timeseries_long):
    """Test both split strategies."""
    if strategy == "dynamic":
        train, val, test = temporal_split(
            sample_timeseries_long,
            strategy="dynamic",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
    else:
        train, val, test = temporal_split(
            sample_timeseries_long,
            strategy="years",
            train_years=(2019, 2021),
            val_years=(2022, 2022),
            test_years=(2022, 2023),
        )
    
    assert len(train) > 0
    assert train.index[-1] < val.index[0] or len(val) == 0
