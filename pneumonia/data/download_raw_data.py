"""
Script to download raw data from PostgreSQL and save to data/raw folder
"""

import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.data.load_data import load_training_data
from pneumonia.config import DATA_RAW_PATH
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def download_raw_data(
    table_name: str = "iras_data_raw",
    output_format: str = "csv",
    filename: str = None,
    limit: int = None
):
    """
    Download data from PostgreSQL table and save to data/raw folder.
    
    Args:
        table_name: Name of the table in PostgreSQL (default: 'iras_data_raw')
        output_format: File format - 'csv' or 'parquet' (default: 'csv')
        filename: Output filename (default: {table_name}.{format})
        limit: Optional limit on number of rows to download
        
    Returns:
        Path to saved file
    """
    try:
        # Create data/raw directory if it doesn't exist
        raw_path = Path(DATA_RAW_PATH)
        raw_path.mkdir(parents=True, exist_ok=True)
        
        # Set default filename if not provided
        if filename is None:
            filename = f"{table_name}.{output_format}"
        
        output_path = raw_path / filename
        
        logger.info(f"Downloading data from table: {table_name}")
        df = load_training_data(table_name=table_name, limit=limit)
        
        # Save based on format
        if output_format.lower() == "csv":
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to CSV: {output_path}")
        elif output_format.lower() == "parquet":
            df.to_parquet(output_path, index=False)
            logger.info(f"Data saved to Parquet: {output_path}")
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        logger.info(f"Successfully downloaded {len(df)} rows")
        print(f"✓ Data downloaded successfully: {output_path}")
        print(f"  Rows: {len(df)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        print(f"✗ Failed to download data: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download raw data from PostgreSQL to data/raw folder"
    )
    parser.add_argument(
        "--table",
        type=str,
        default="iras_data_raw",
        help="Table name in PostgreSQL (default: 'iras_data_raw')"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Output file format (default: 'csv')"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Output filename (default: {table_name}.{format})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to download"
    )
    
    args = parser.parse_args()
    
    download_raw_data(
        table_name=args.table,
        output_format=args.format,
        filename=args.filename,
        limit=args.limit
    )
