"""
Prophet model configuration

Default parameters and departmental overrides for Prophet forecasting models.
"""

# Default hyperparameters for Prophet
PROPHET_DEFAULT_PARAMS = {
    "growth": "linear",
    "yearly_seasonality": True,       # Critical for annual winter pneumonia cycles
    "weekly_seasonality": False,      # Weekly data, so no day-of-week patterns
    "daily_seasonality": False,       # Weekly data, so no hour-of-day patterns
    "seasonality_mode": "additive",   # Additive seasonality is robust
    "changepoint_prior_scale": 0.05,  # Control trend flexibility (default 0.05)
    "seasonality_prior_scale": 10.0,  # Control seasonality flexibility (default 10.0)
}

# Department-specific Prophet configurations. Overrides defaults if present.
# Set country_name='PE' by default to use Peru's official holidays.
DEPARTMENTAL_CONFIGS = {
    # Example:
    # "LIMA": {
    #     "changepoint_prior_scale": 0.08,
    # }
}

# Hyperparameter search ranges for Prophet tuning
PROPHET_SEARCH_RANGES = {
    "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.3, 0.5],
    "seasonality_prior_scale": [0.1, 1.0, 5.0, 10.0, 20.0],
    "seasonality_mode": ["additive", "multiplicative"],
}

