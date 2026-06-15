"""
Configuration module for pneumonia detection project

Centralizes all configuration for the pneumonia forecasting pipeline:
- Data paths and file locations
- Training/validation/test split strategy
- SARIMA model hyperparameters
- Logging and model storage
- Reproducibility settings
"""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
"""Global random seed for reproducibility. Set in all np.random.seed, tf.seed, etc."""

# =============================================================================
# DATA PATHS
# =============================================================================
DATA_RAW_PATH = os.getenv("DATA_RAW_PATH", "data/raw/")
DATA_EXTERNAL_PATH = os.getenv("DATA_EXTERNAL_PATH", "data/external/")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/")
DATA_INTERIM_PATH = os.getenv("DATA_INTERIM_PATH", "data/interim/")

# =============================================================================
# MODEL STORAGE & LOGGING
# =============================================================================
MODEL_STORAGE_PATH = Path(os.getenv("MODEL_STORAGE_PATH", "models/"))
LOG_PATH = Path(os.getenv("LOG_PATH", "logs/"))
REPORTS_PATH = Path(os.getenv("REPORTS_PATH", "reports/"))

# Create directories if they don't exist
for path in [MODEL_STORAGE_PATH, LOG_PATH, REPORTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TEMPORAL SPLIT STRATEGY
# =============================================================================
TEMPORAL_SPLIT_STRATEGY = os.getenv("TEMPORAL_SPLIT_STRATEGY", "dynamic").lower()
"""Split strategy: 'dynamic' (calc from data) or 'years' (use fixed year ranges)"""

if TEMPORAL_SPLIT_STRATEGY not in ["dynamic", "years"]:
    raise ValueError(f"TEMPORAL_SPLIT_STRATEGY must be 'dynamic' or 'years', got {TEMPORAL_SPLIT_STRATEGY}")

# For dynamic split:
DEFAULT_TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.8"))  # 80% train
DEFAULT_VAL_RATIO = float(os.getenv("VAL_RATIO", "0.1"))      # 10% val
DEFAULT_TEST_RATIO = float(os.getenv("TEST_RATIO", "0.1"))    # 10% test

if not abs((DEFAULT_TRAIN_RATIO + DEFAULT_VAL_RATIO + DEFAULT_TEST_RATIO) - 1.0) < 0.01:
    raise ValueError(
        f"Train/Val/Test ratios must sum to 1.0, got {DEFAULT_TRAIN_RATIO + DEFAULT_VAL_RATIO + DEFAULT_TEST_RATIO}"
    )

# For year-based split (legacy):
DEFAULT_TRAIN_YEARS = (2000, 2019)
DEFAULT_VAL_YEARS = (2020, 2021)
DEFAULT_TEST_YEARS = (2022, 2023)

# =============================================================================
# SARIMA MODEL CONFIGURATION
# =============================================================================
SARIMA_USE_AUTO_ARIMA = os.getenv("USE_AUTO_ARIMA", "True").lower() == "true"
"""If True, use pmdarima.auto_arima() to find optimal (p,d,q)×(P,D,Q,s).
If False, use DEFAULT_SARIMA_ORDER from sarima/config.py"""

SARIMA_MAX_ITERATIONS = int(os.getenv("SARIMA_MAX_ITERATIONS", "400"))
SARIMA_STEPWISE = os.getenv("SARIMA_STEPWISE", "True").lower() == "true"
SARIMA_TRACE = os.getenv("SARIMA_TRACE", "False").lower() == "true"

# =============================================================================
# DATABASE (if needed in future)
# =============================================================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/pneumonia")
