"""
SARIMA model configuration

Default parameters for SARIMA (Seasonal ARIMA) models by data characteristics.

SARIMA order notation: (p, d, q) × (P, D, Q, s)
    p: AR (AutoRegressive) order
    d: Differencing order (0=no diff, 1=first-order diff, 2=second-order diff)
    q: MA (Moving Average) order
    P: Seasonal AR order
    D: Seasonal differencing order
    Q: Seasonal MA order
    s: Seasonal period (52 for weekly data with annual seasonality)

Example: (2, 1, 1) × (1, 1, 1, 52) means:
    - Non-seasonal: AR(2), differenced once, MA(1)
    - Seasonal: SAR(1), seasonal differencing, SMA(1), period=52 weeks

Note: If USE_AUTO_ARIMA=True in config.py, these defaults are overridden
by pmdarima.auto_arima() which searches the parameter space automatically.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# DEFAULT SARIMA PARAMETERS (fallback when auto_arima disabled)
# =============================================================================
DEFAULT_SARIMA_ORDER = (1, 1, 1)
"""Default (p, d, q) order. Typically: d=1 for differencing, p/q=1-2 for AR/MA"""

DEFAULT_SARIMA_SEASONAL_ORDER = (1, 1, 1, 52)
"""Default (P, D, Q, s) seasonal order. s=52 for weekly data (annual seasonality)"""

# =============================================================================
# AUTO_ARIMA SEARCH SPACE (if USE_AUTO_ARIMA=True in main config.py)
# =============================================================================
SARIMA_P_RANGE = range(0, 3)  # 0, 1, 2 (AR order)
SARIMA_D_RANGE = range(0, 2)  # 0, 1 (differencing)
SARIMA_Q_RANGE = range(0, 3)  # 0, 1, 2 (MA order)

SARIMA_P_SEASONAL_RANGE = range(0, 2)  # 0, 1 (seasonal AR)
SARIMA_D_SEASONAL_RANGE = range(0, 2)  # 0, 1 (seasonal differencing)
SARIMA_Q_SEASONAL_RANGE = range(0, 2)  # 0, 1 (seasonal MA)
SARIMA_SEASONAL_PERIOD = 52  # weeks per year

# =============================================================================
# AUTO_ARIMA FITTING OPTIONS
# =============================================================================
# Note: These are read from main config.py but referenced here for clarity
# See pneumonia/config.py: SARIMA_MAX_ITERATIONS, SARIMA_STEPWISE, SARIMA_TRACE

# Fitting options for SARIMAX model
SARIMA_ENFORCE_STATIONARITY = False
"""If True, enforces AR parameters to be stationary (may fail in some cases)"""

SARIMA_ENFORCE_INVERTIBILITY = False
"""If True, enforces MA parameters to be invertible (may fail in some cases)"""

# =============================================================================
# DEPARTMENT-SPECIFIC OVERRIDES
# =============================================================================
DEPARTMENTAL_CONFIGS = {
    # Example: "LIMA": {
    #     "order": (2, 1, 1),
    #     "seasonal_order": (1, 1, 1, 52),
    #     "use_auto_arima": False
    # },
}
"""Department-specific SARIMA configurations. Overrides defaults if present."""

# =============================================================================
# AGE GROUP CHARACTERISTICS (informational)
# =============================================================================
AGE_GROUP_FACTORS = {
    "under5": {
        "description": "Children under 5 years",
        "seasonality_strength": "strong",
        "volatility": "high",
        "notes": "Higher volatility, strong winter peaks",
    },
    "60plus": {
        "description": "Adults 60+ years",
        "seasonality_strength": "moderate",
        "volatility": "medium",
        "notes": "Moderate seasonality, lower overall volatility",
    },
}
