"""
Data loading and processing module
"""

import pandas as pd
import logging
from pathlib import Path
import sys
from sqlalchemy import create_engine
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


def get_db_engine(database_url: str):
    """
    Create database connection engine
    
    Args:
        database_url: PostgreSQL connection URL
        
    Returns:
        SQLAlchemy engine
    """
    from pneumonia.config import DATABASE_URL
    return create_engine(database_url or DATABASE_URL)

def load_training_data(
    table_name: str,
    database_url: Optional[str] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load training data from PostgreSQL.
    
    Args:
        table_name: Name of the table in PostgreSQL
        database_url: PostgreSQL connection URL (uses config if not provided)
        limit: Limit number of rows
        
    Returns:
        DataFrame with training data
    """
    engine = get_db_engine(database_url)
    
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"
    
    logger.info(f"Loading data from table '{table_name}' with Query: {query}")
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from {table_name}")
    
    return df

if __name__ == "__main__":
    print(load_training_data("iras_data_raw", limit = 21))