"""
Feature engineering and building functions.

This module contains functions for constructing features from raw pneumonia data
for use in machine learning models.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def build_features(
    data: pd.DataFrame,
    include_lagged: bool = True,
    include_seasonal: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Build feature set from pneumonia time series data.
    
    This function constructs engineered features from raw pneumonia case data,
    including lagged features, seasonal components, and temporal aggregations.
    
    Args:
        data: DataFrame with pneumonia cases indexed by time
        include_lagged: Whether to include lagged features (t-1, t-2, etc.)
        include_seasonal: Whether to include seasonal decomposition features
        **kwargs: Additional feature engineering parameters
        
    Returns:
        DataFrame with engineered features ready for modeling
        
    Notes:
        - Integrates with pneumonia.data module for data loading
        - Handles missing values using interpolation strategy from pneumonia.data.load_data
        - Feature names should follow pattern: 'lag_<n>', 'seasonal_<component>'
        
    TODO: Implement feature engineering logic
        - Add lagged features: cases_t-1, cases_t-2, ..., cases_t-k
        - Add rolling statistics: 7-week and 13-week rolling means, std
        - Add seasonal decomposition: trend, seasonal, residual components
        - Add time-based features: week_of_year, month, quarter
        - Add domain features: population-adjusted rates, YoY growth
        - Handle edge cases (start of series, missing values)
    """
    pass


def prepare_features_for_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    scaler=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare and scale features for model training and prediction.
    
    Applies consistent preprocessing to both training and test features,
    including feature scaling and alignment.
    
    Args:
        train_data: Training feature set
        test_data: Test/validation feature set
        scaler: Sklearn-compatible scaler object (StandardScaler, MinMaxScaler, etc.)
                If None, StandardScaler is used by default
        
    Returns:
        Tuple of (scaled_train_array, scaled_test_array) as numpy arrays
        
    Raises:
        ValueError: If train_data and test_data have mismatched columns
        
    TODO: Implement feature scaling pipeline
        - Initialize scaler if not provided (default: StandardScaler)
        - Fit scaler on training data
        - Transform both train and test data consistently
        - Handle NaN values and edge cases
        - Return as numpy arrays for ML model compatibility
        - Store feature names for later interpretation
    """
    pass
