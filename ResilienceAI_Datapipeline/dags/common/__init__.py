"""
Common utilities package for data pipelines
Modular structure for News, arXiv, and Google Scholar pipelines
"""

# Import all utilities for convenient access
from .deduplication import DeduplicationManager
from .data_cleaning import TextCleaner
from .data_enrichment import DataEnricher, CategoryManager
from .data_validator import DataValidator
from .database_utils import DatabaseManager
from .file_management import FileManager

__all__ = [
    'DeduplicationManager',
    'TextCleaner',
    'DataEnricher',
    'CategoryManager',
    'DataValidator',
    'DatabaseManager',
    'FileManager'
]

__version__ = '1.0.0'