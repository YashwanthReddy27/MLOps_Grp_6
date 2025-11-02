"""
Caching utilities - File-based only
"""
from typing import Optional, Any
import hashlib
import pickle
import logging
from pathlib import Path

from config.settings import config

logger = logging.getLogger(__name__)


class SimpleCache:
    """Simple file-based cache"""
    
    def __init__(self):
        cache_dir = config.cache.cache_dir
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized file cache at {self.cache_dir}")
    
    def _get_key_hash(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        key_hash = self._get_key_hash(key)
        cache_file = self.cache_dir / f"{key_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error reading cache: {e}")
                return None
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (not implemented for file cache)
        """
        key_hash = self._get_key_hash(key)
        cache_file = self.cache_dir / f"{key_hash}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self.logger.warning(f"Error writing cache: {e}")
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.logger.info("Cleared cache")


def get_cache():
    """Factory function to get cache"""
    return SimpleCache()