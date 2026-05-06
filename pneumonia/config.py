"""
Configuration module for pneumonia detection project
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/pneumonia")

# Model Configuration
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "models/")
LOG_PATH = os.getenv("LOG_PATH", "logs/")

# Data Configuration
DATA_RAW_PATH = "data/raw/"
DATA_EXTERNAL_PATH = "data/external/"
DATA_PROCESSED_PATH = "data/processed/"
DATA_INTERIM_PATH = "data/interim/"

# Training Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
