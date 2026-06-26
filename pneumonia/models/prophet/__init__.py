"""
Prophet module initialization

Exports Prophet model and configuration.
"""

from pneumonia.models.prophet.model import ProphetModel
from pneumonia.models.prophet import config

__all__ = ["ProphetModel", "config"]
