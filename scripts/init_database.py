"""
Database initialization script
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.config import DATABASE_URL
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def init_database():
    """Initialize PostgreSQL database for model storage"""
    logger.info(f"Initializing database: {DATABASE_URL}")
    
    # Implementar lógica de creación de tablas
    # CREATE TABLE models (
    #     id SERIAL PRIMARY KEY,
    #     name VARCHAR(255) NOT NULL,
    #     version INT NOT NULL,
    #     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    #     metrics JSONB,
    #     weights BYTEA,
    #     metadata JSONB
    # );
    
    logger.info("Database initialized successfully")


if __name__ == "__main__":
    init_database()
