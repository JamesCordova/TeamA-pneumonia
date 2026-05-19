"""
Configuration parameters for ML models.

This module contains default hyperparameters, search ranges, and
department-specific configurations for XGBoost, RandomForest, and Ensemble models.
"""

# XGBoost default hyperparameters
XGBOOST_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}

# XGBoost hyperparameter search ranges (for GridSearchCV/RandomizedSearchCV)
XGBOOST_SEARCH_RANGES = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# RandomForest default hyperparameters
RANDOM_FOREST_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

# RandomForest hyperparameter search ranges
RANDOM_FOREST_SEARCH_RANGES = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

# Ensemble model weights for combining predictions
# Format: {model_name: weight}
# Weights should sum to 1.0 or will be normalized
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.4,
    "random_forest": 0.4,
    "sarima": 0.2,
}

# Department-specific model configurations
# Override defaults for specific departments
DEPARTMENTAL_CONFIGS = {
    "LIMA": {
        "xgboost_params": {},  # Empty dict means use XGBOOST_DEFAULT_PARAMS
        "random_forest_params": {},
    },
    "AMAZONAS": {
        "xgboost_params": {"learning_rate": 0.05},
        "random_forest_params": {"max_depth": 7},
    },
    # Add more departments as needed
}

# Age group-specific factors and parameters
AGE_GROUP_FACTORS = {
    "under5": {
        "seasonal_period": 52,  # Weekly seasonality
        "min_training_weeks": 104,  # 2 years minimum
        "trend_strength": "moderate",
    },
    "60plus": {
        "seasonal_period": 52,
        "min_training_weeks": 104,
        "trend_strength": "strong",
    },
}

# Feature engineering configuration
FEATURE_ENGINEERING_CONFIG = {
    "include_lagged_features": True,
    "lag_periods": [1, 2, 4, 8, 13],  # 1, 2, 4, 8 weeks and quarterly
    "include_rolling_stats": True,
    "rolling_windows": [4, 13, 26],  # 1 month, 3 months, 6 months
    "include_seasonal_features": True,
    "include_trend_features": True,
    "scaling_method": "standard",  # 'standard', 'minmax', 'robust'
}

# Model evaluation configuration
MODEL_EVALUATION_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "cross_validation_folds": 5,
    "metrics": ["mae", "rmse", "mape", "r2"],
}
