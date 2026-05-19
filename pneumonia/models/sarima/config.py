"""
SARIMA model configuration

Default parameters for SARIMA (Seasonal ARIMA) models by data characteristics.
"""

# Default SARIMA order: (p, d, q) × (P, D, Q, s)
# p, d, q: AR order, differencing, MA order
# P, D, Q: Seasonal AR, seasonal differencing, seasonal MA
# s: Seasonal period (52 for weekly data with annual seasonality)

DEFAULT_SARIMA_ORDER = (1, 1, 1)
DEFAULT_SARIMA_SEASONAL_ORDER = (1, 1, 1, 52)

# Hyperparameter search ranges (for auto_arima)
SARIMA_P_RANGE = range(0, 3)  # 0, 1, 2
SARIMA_D_RANGE = range(0, 2)  # 0, 1 (seasonal data usually needs d=1)
SARIMA_Q_RANGE = range(0, 3)  # 0, 1, 2

SARIMA_P_SEASONAL_RANGE = range(0, 2)  # 0, 1
SARIMA_D_SEASONAL_RANGE = range(0, 2)  # 0, 1
SARIMA_Q_SEASONAL_RANGE = range(0, 2)  # 0, 1
SARIMA_SEASONAL_PERIOD = 52  # weeks per year

# Fitting options
SARIMA_MAX_ITERATIONS = 400
SARIMA_STEPWISE = True  # Use stepwise algorithm (faster)
SARIMA_TRACE = False   # Print progress

# Department-specific overrides (if needed)
DEPARTMENTAL_CONFIGS = {
    # Example: "LIMA": {"order": (2, 1, 1), "seasonal_order": (1, 1, 0, 52)},
}

# Age group considerations
AGE_GROUP_FACTORS = {
    "under5": {
        "description": "Children under 5 years",
        "seasonality_strength": "strong",  # Strong seasonal pattern
        "volatility": "high",
    },
    "60plus": {
        "description": "Adults 60+ years",
        "seasonality_strength": "moderate",  # Moderate seasonal pattern
        "volatility": "medium",
    },
}
