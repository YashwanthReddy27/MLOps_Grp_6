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
    def ensure_directories(base_dir: str = '/opt/airflow/data') -> None:
        """
        Create standard directory structure for data pipelines
        
        Args:
            base_dir: Base directory path (default: /opt/airflow/data)
            
        Creates:
            - {base_dir}/raw - Raw data files
            - {base_dir}/cleaned - Processed/cleaned data
            - {base_dir}/schema - Schema files
            - {base_dir}/hashes - Deduplication hash files
            
        Example:
            FileManager.ensure_directories('/opt/airflow/data')
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
            
        Example:
            FileManager.save_json(
                {"articles": [...]},
                '/opt/airflow/data/raw/news.json'
            )
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
            
        Example:
            data = FileManager.load_json('/opt/airflow/data/raw/news.json')
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
            
        Example:
            deleted = FileManager.cleanup_old_files(
                '/opt/airflow/data/raw',
                'tech_news_',
                days=7
            )
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
    
    @staticmethod
    def list_files(directory: str, pattern: str = None, extension: str = None) -> List[str]:
        """
        List files in directory matching pattern and/or extension
        
        Args:
            directory: Directory to search
            pattern: Filename pattern to match (optional)
            extension: File extension to match (optional, e.g., '.json')
            
        Returns:
            Sorted list of matching file paths
            
        Example:
            json_files = FileManager.list_files(
                '/opt/airflow/data/raw',
                pattern='tech_news',
                extension='.json'
            )
        """
        if not os.path.exists(directory):
            return []
        
        files = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Skip directories
            if os.path.isdir(filepath):
                continue
            
            # Check pattern
            if pattern and pattern not in filename:
                continue
            
            # Check extension
            if extension and not filename.endswith(extension):
                continue
            
            files.append(filepath)
        
        return sorted(files)
    
    @staticmethod
    def get_file_age_days(filepath: str) -> float:
        """
        Get file age in days
        
        Args:
            filepath: Path to file
            
        Returns:
            Age in days, or -1 if file doesn't exist
            
        Example:
            age = FileManager.get_file_age_days('/path/to/file.json')
            if age > 7:
                print("File is older than a week")
        """
        if not os.path.exists(filepath):
            return -1.0
        
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        age = datetime.now() - file_time
        return age.total_seconds() / 86400  # Convert to days
    
    @staticmethod
    def get_file_size_mb(filepath: str) -> float:
        """
        Get file size in megabytes
        
        Args:
            filepath: Path to file
            
        Returns:
            Size in MB, or -1 if file doesn't exist
        """
        if not os.path.exists(filepath):
            return -1.0
        
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    @staticmethod
    def file_exists(filepath: str) -> bool:
        """
        Check if file exists
        
        Args:
            filepath: Path to check
            
        Returns:
            True if file exists
        """
        return os.path.exists(filepath) and os.path.isfile(filepath)
    
    @staticmethod
    def create_backup(filepath: str, backup_suffix: str = '.bak') -> Optional[str]:
        """
        Create backup of a file
        
        Args:
            filepath: File to backup
            backup_suffix: Suffix for backup file (default: '.bak')
            
        Returns:
            Backup file path if successful, None otherwise
            
        Example:
            backup = FileManager.create_backup('/path/to/data.json')
            # Creates: /path/to/data.json.bak
        """
        if not os.path.exists(filepath):
            print(f"[FILE] Cannot backup, file doesn't exist: {filepath}")
            return None
        
        try:
            backup_path = filepath + backup_suffix
            
            # Copy file
            import shutil
            shutil.copy2(filepath, backup_path)
            
            print(f"[FILE] Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"[FILE] Error creating backup: {e}")
            return None
    
    @staticmethod
    def safe_delete(filepath: str) -> bool:
        """
        Safely delete a file (returns success status)
        
        Args:
            filepath: File to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"[FILE] Deleted: {filepath}")
                return True
            return False
        except Exception as e:
            print(f"[FILE] Error deleting {filepath}: {e}")
            return False
    
    @staticmethod
    def get_directory_size_mb(directory: str) -> float:
        """
        Get total size of directory in megabytes
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in MB
        """
        if not os.path.exists(directory):
            return 0.0
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        
        return total_size / (1024 * 1024)  # Convert to MB


__all__ = ['FileManager']