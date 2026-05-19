"""
Validation script for forecasting models

Tests that all modules import correctly and trains a sample SARIMA model.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pneumonia.models import BaseForecaster, SARIMAModel
from pneumonia.models import utils as model_utils
from pneumonia.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    compute_all_metrics,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def test_imports():
    """Test that all modules import correctly."""
    print("\n" + "=" * 80)
    print("TESTING IMPORTS")
    print("=" * 80)
    
    try:
        from pneumonia.models.base import BaseForecaster
        print("✓ BaseForecaster imported")
    except Exception as e:
        print(f"✗ BaseForecaster failed: {e}")
        return False
    
    try:
        from pneumonia.models.sarima.model import SARIMAModel
        print("✓ SARIMAModel imported")
    except Exception as e:
        print(f"✗ SARIMAModel failed: {e}")
        return False
    
    try:
        from pneumonia.evaluation import compute_all_metrics
        print("✓ Evaluation metrics imported")
    except Exception as e:
        print(f"✗ Evaluation metrics failed: {e}")
        return False
    
    try:
        from pneumonia.models import utils as model_utils
        print("✓ Model utils imported")
    except Exception as e:
        print(f"✗ Model utils failed: {e}")
        return False
    
    print("\n✓ All imports successful\n")
    return True


def test_data_loading():
    """Test data loading utilities."""
    print("=" * 80)
    print("TESTING DATA LOADING")
    print("=" * 80)
    
    try:
        # Get available departments
        depts = model_utils.get_available_departments()
        print(f"✓ Found {len(depts)} departments with sufficient data")
        print(f"  Departments: {depts[:5]}{'...' if len(depts) > 5 else ''}")
        
        if len(depts) == 0:
            print("✗ No departments with sufficient data found")
            return False
        
        # Load data for first department
        test_dept = depts[0]
        ts = model_utils.get_departmental_data(test_dept, age_group="under5")
        print(f"✓ Loaded time series for {test_dept} (under5): {len(ts)} observations")
        
        # Test temporal split
        train, val, test = model_utils.temporal_split(ts)
        print(f"✓ Temporal split successful:")
        print(f"  Train: {len(train)} weeks")
        print(f"  Val:   {len(val)} weeks")
        print(f"  Test:  {len(test)} weeks")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sarima_training():
    """Test SARIMA model training."""
    print("\n" + "=" * 80)
    print("TESTING SARIMA TRAINING")
    print("=" * 80)
    
    try:
        # Get data
        depts = model_utils.get_available_departments()
        if not depts:
            print("✗ No data available")
            return False
        
        test_dept = depts[0]
        ts = model_utils.get_departmental_data(test_dept, age_group="under5")
        
        # Split data (use smaller subset for faster testing)
        train, val, test_set = model_utils.temporal_split(ts)
        # Use only last 2 years of training for faster fitting
        train = train[-104:]
        print(f"  Using {len(train)} weeks for training (reduced for faster testing)")
        
        # Create model
        model = SARIMAModel(
            department=test_dept,
            age_group="under5",
        )
        print(f"✓ Created model: {model}")
        
        # Fit model (with smaller subset)
        print(f"\n  Fitting SARIMA on {len(train)} observations...")
        model.fit(train)
        print(f"✓ Model fitted successfully")
        print(f"  AIC: {model.metadata.get('aic', 'N/A')}")
        
        # Make prediction (just 4 weeks for speed)
        forecast = model.predict(steps=4)
        print(f"✓ Generated forecast: {len(forecast)} predictions")
        
        # Evaluate on small test set
        test_eval = test_set.iloc[:4] if len(test_set) >= 4 else test_set
        if len(test_eval) > 0 and len(forecast) > 0:
            metrics = model.evaluate(test_eval, forecast[:len(test_eval)])
            print(f"✓ Model evaluation:")
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    print(f"    {metric_name}: {value:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ SARIMA training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\n" + "=" * 80)
    print("TESTING METRICS")
    print("=" * 80)
    
    try:
        # Create sample data
        actual = np.array([10, 15, 12, 18, 20, 17, 22, 25, 23, 20])
        predicted = np.array([11, 14, 13, 17, 21, 16, 23, 24, 24, 19])
        
        # Test individual metrics
        mae = mean_absolute_error(actual, predicted)
        print(f"✓ MAE: {mae:.2f}")
        
        rmse = root_mean_squared_error(actual, predicted)
        print(f"✓ RMSE: {rmse:.2f}")
        
        mape = mean_absolute_percentage_error(actual, predicted)
        print(f"✓ MAPE: {mape:.2f}%")
        
        # Test all metrics
        all_metrics = compute_all_metrics(actual, predicted)
        print(f"✓ All metrics computed: {list(all_metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("FORECASTING MODELS VALIDATION")
    print("=" * 80)
    
    results = {
        "imports": test_imports(),
        "data_loading": test_data_loading(),
        "metrics": test_metrics(),
        "sarima_training": test_sarima_training(),
    }
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("=" * 80)
    if all_passed:
        print("✓ All validation tests passed!")
    else:
        print("✗ Some tests failed. Review errors above.")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
