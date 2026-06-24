"""
Unit tests for the Prophet forecasting model.
"""

import unittest
import numpy as np
import pandas as pd

from pneumonia.models.prophet.model import ProphetModel


class TestProphetModel(unittest.TestCase):
    """Test cases for ProphetModel forecaster."""

    def setUp(self):
        # Generate dummy weekly time series (150 weeks)
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=150, freq="7D")
        t = np.arange(150)
        # Seasonal sine wave + slight trend + random noise
        cases = 50 + 15 * np.sin(2 * np.pi * t / 52) + 0.05 * t + np.random.normal(0, 3, 150)
        cases = np.maximum(0.0, cases)  # Ensure non-negative cases
        self.dummy_data = pd.Series(cases, index=dates)

    def test_initialization(self):
        """Test ProphetModel parameter merging and initialization."""
        model = ProphetModel(
            department="AMAZONAS",
            age_group="under5",
            growth="linear",
            changepoint_prior_scale=0.1,
        )
        self.assertEqual(model.name, "Prophet")
        self.assertEqual(model.department, "AMAZONAS")
        self.assertEqual(model.age_group, "under5")
        self.assertFalse(model.is_fitted)
        self.assertEqual(model._params["growth"], "linear")
        self.assertEqual(model._params["changepoint_prior_scale"], 0.1)

    def test_fit_and_predict(self):
        """Test ProphetModel fitting and step forecasting."""
        model = ProphetModel(department="LIMA", age_group="60plus")
        model.fit(self.dummy_data)
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.fitted_date)

        # Test predict method
        steps = 12
        preds = model.predict(self.dummy_data, steps=steps)
        self.assertEqual(len(preds), steps)
        self.assertTrue((preds >= 0.0).all())

    def test_forecast_interval(self):
        """Test confidence intervals are calculated correctly and bounds are consistent."""
        model = ProphetModel(department="AMAZONAS", age_group="under5")
        model.fit(self.dummy_data)

        steps = 8
        alpha = 0.05
        preds, lower, upper = model.get_forecast_interval(
            self.dummy_data, steps=steps, alpha=alpha
        )

        self.assertEqual(len(preds), steps)
        self.assertEqual(len(lower), steps)
        self.assertEqual(len(upper), steps)

        # All values should be non-negative
        self.assertTrue((preds >= 0.0).all())
        self.assertTrue((lower >= 0.0).all())
        self.assertTrue((upper >= 0.0).all())

        # Check mathematical consistency of intervals
        self.assertTrue((lower <= preds).all())
        self.assertTrue((preds <= upper).all())


if __name__ == "__main__":
    unittest.main()
