"""
File management utilities
Handles file system operations, JSON operations, and file cleanup
"""

import os
import json
from datetime import datetime, timedelta
from typing import Any, List, Optional


class FileManager:
    """File system operations for data pipelines"""
    
    @staticmethod
    def ensure_directories(base_dir: str = '/home/airflow/gcs/dags/') -> None:
        """
        Create standard directory structure for data pipelines
        
        Args:
            base_dir: Base directory path (default: /home/airflow/gcs/data)
            
        Creates:
            - {base_dir}/raw - Raw data files
            - {base_dir}/cleaned - Processed/cleaned data
            - {base_dir}/schema - Schema files
            - {base_dir}/hashes - Deduplication hash files
        """
        dirs = [
            f'{base_dir}/raw',
            f'{base_dir}/cleaned',
            f'{base_dir}/schema',
            f'{base_dir}/hashes',
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            try:
                os.chmod(directory, 0o777)
            except:
                pass
        
        print(f"[FILE] Ensured directories exist under {base_dir}")
    
    @staticmethod
    def save_json(data: Any, filepath: str, indent: int = 2) -> None:
        """
        Save data as JSON file
        
        Args:
            data: Data to save (must be JSON serializable)
            filepath: Full path where to save the file
            indent: JSON indentation (default: 2)
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
        print(f"[FILE] Saved to {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> Any:
        """
        Load JSON file
        
        Args:
            filepath: Full path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[FILE] Loaded from {filepath}")
        return data
    
    @staticmethod
    def cleanup_old_files(directory: str, pattern: str, days: int = 7) -> int:
        """
        Delete files older than N days matching pattern
        
        Args:
            directory: Directory to clean
            pattern: Filename pattern to match
            days: Age threshold in days (default: 7)
            
        Returns:
            Number of files deleted
        """
        if not os.path.exists(directory):
            print(f"[CLEANUP] Directory does not exist: {directory}")
            return 0
        
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        
        for filename in os.listdir(directory):
            if pattern in filename:
                filepath = os.path.join(directory, filename)
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff:
                        os.remove(filepath)
                        deleted += 1
                        print(f"[CLEANUP] Deleted: {filepath}")
                except Exception as e:
                    print(f"[CLEANUP] Error deleting {filepath}: {e}")
        
        print(f"[CLEANUP] Total files deleted from {directory}: {deleted}")
        return deleted


__all__ = ['FileManager']