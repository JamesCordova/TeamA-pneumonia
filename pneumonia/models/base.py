"""
Base class for forecasting models

This module defines the abstract BaseForecaster class that all forecasting models
must inherit from, ensuring a consistent interface for training, prediction,
and evaluation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import json
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    This class defines the interface that all forecasting models must implement,
    including fitting, prediction, evaluation, and serialization.
    
    Attributes:
        name: Model name/identifier
        department: Department for which the model is trained
        age_group: Age group (under5 or 60plus)
        is_fitted: Whether the model has been fitted
        fitted_date: Timestamp of when model was last fitted
        metadata: Additional metadata about the model
    """
    
    def __init__(
        self,
        name: str,
        department: str,
        age_group: str,
        **kwargs
    ):
        """
        Initialize forecaster.
        
        Args:
            name: Model identifier
            department: Department name (e.g., 'LIMA')
            age_group: 'under5' or '60plus'
            **kwargs: Additional model-specific parameters
        """
        self.name = name
        self.department = department.upper()
        self.age_group = age_group.lower()
        self.is_fitted = False
        self.fitted_date = None
        self.metadata = {
            "name": name,
            "department": self.department,
            "age_group": self.age_group,
            "created": datetime.now().isoformat(),
        }
        self.metadata.update(kwargs)
        
        logger.info(f"Initialized {name} for {self.department} ({self.age_group})")
    
    @abstractmethod
    def fit(self, train_data: pd.Series, **kwargs) -> None:
        """
        Fit the forecasting model to training data.
        
        Args:
            train_data: Time series data for training
            **kwargs: Model-specific fitting parameters
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts for specified number of steps ahead.
        
        Args:
            steps: Number of forecast steps
            
        Returns:
            Array of forecasted values
        """
        pass
    
    def evaluate(
        self,
        actual: pd.Series,
        predicted: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate forecast performance using all standard metrics.

        Returns:
            Dictionary with mae, rmse, mape, smape, mda values.
        """
        from pneumonia.evaluation.metrics import compute_all_metrics
        results = compute_all_metrics(actual, predicted)
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save fitted model to disk.
        
        Saves two files:
        1. Model pickle: {name}.pkl - The actual model object
        2. Metadata JSON: {name}_metadata.json - Training info, timestamps, params
        
        Args:
            filepath: Path to save model. If None, uses default path.
                     If filepath ends with .pkl, metadata saved as _metadata.json
                     
        Returns:
            Path where model was saved (the .pkl file)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        if filepath is None:
            models_dir = Path("models") / self.department / self.age_group
            models_dir.mkdir(parents=True, exist_ok=True)
            filepath = models_dir / f"{self.name}.pkl"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure .pkl extension
        if not str(filepath).endswith('.pkl'):
            filepath = filepath.with_suffix('.pkl')
        
        try:
            # Save model pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
            
            # Save metadata JSON alongside the model
            metadata_path = filepath.with_stem(filepath.stem + '_metadata')
            metadata_path = metadata_path.with_suffix('.json')
            
            metadata_dict = {
                "name": self.name,
                "department": self.department,
                "age_group": self.age_group,
                "is_fitted": self.is_fitted,
                "fitted_date": self.fitted_date,
                "saved_date": datetime.now().isoformat(),
                "model_file": str(filepath.name),
                "metadata": self.metadata,
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    @staticmethod
    def load(filepath: Path) -> 'BaseForecaster':
        """
        Load a fitted model from disk.
        
        Loads the model pickle file. Optionally reads metadata JSON if present.
        
        Args:
            filepath: Path to saved model (.pkl file)
            
        Returns:
            Loaded forecaster instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            
            # Try to load and log metadata if available
            metadata_path = filepath.with_stem(filepath.stem + '_metadata')
            metadata_path = metadata_path.with_suffix('.json')
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    logger.debug(f"Loaded metadata: {metadata.get('saved_date', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Could not load metadata from {metadata_path}: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model metadata
        """
        metadata = self.metadata.copy()
        metadata.update({
            "is_fitted": self.is_fitted,
            "fitted_date": self.fitted_date,
        })
        return metadata
    
    def save_metadata(self, filepath: Optional[Path] = None) -> Path:
        """
        Save model metadata to JSON.
        
        Args:
            filepath: Path to save metadata. If None, uses default path.
            
        Returns:
            Path where metadata was saved
        """
        if filepath is None:
            models_dir = Path("models") / self.department / self.age_group
            models_dir.mkdir(parents=True, exist_ok=True)
            filepath = models_dir / f"{self.name}_metadata.json"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_metadata(), f, indent=2, default=str)
            logger.info(f"Metadata saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise
    
    def __repr__(self) -> str:
        """String representation of forecaster."""
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name}, dept={self.department}, age={self.age_group}, {status})"
