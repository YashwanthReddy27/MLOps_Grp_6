"""
Deduplication utilities for data pipelines
Manages duplicate detection across pipeline runs using hash-based tracking
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Callable
import logging
from pathlib import Path


class DeduplicationManager:
    """Manages deduplication across pipelines"""
    
    def __init__(self, pipeline_name: str, base_dir: str = '/home/airflow/gcs/data'):
        """
        Initialize deduplication manager
        
        Args:
            pipeline_name: Name of the pipeline (e.g., 'news', 'arxiv', 'scholar')
            base_dir: Base directory for data storage
        """
        self.pipeline_name = pipeline_name
        self.hash_dir = f'{base_dir}/hashes'
        self.hash_file = f'{self.hash_dir}/{pipeline_name}_hashes.json'
        os.makedirs(self.hash_dir, exist_ok=True)
        # === Logging setup ===
        base_log_dir = Path("/home/airflow/gcs/logs")
        base_log_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists
        log_file_path = base_log_dir / f"{__name__}_{datetime.now().strftime('%Y-%m-%d')}.log"
        
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"[DEDUP] Initialized DeduplicationManager for {pipeline_name}")
    
    def generate_hash(self, *fields) -> str:
        """
        Generate MD5 hash from fields
        
        Args:
            *fields: Variable number of fields to hash
            
        Returns:
            MD5 hash string
        """
        content = ''.join(str(f) for f in fields if f)
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_hashes(self) -> set:
        """
        Load existing hashes from file
        
        Returns:
            Set of existing hashes
        """
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('hashes', []))
            except Exception as e:
                self.logger.error(f"[DEDUP] Error loading hashes: {e}")
                return set()
        return set()
    
    def save_hashes(self, hashes: List[str], max_keep: int = 10000) -> None:
        """
        Save hashes to file (keep last N)
        
        Args:
            hashes: List of hashes to save
            max_keep: Maximum number of hashes to keep (default: 10000)
        """
        try:
            with open(self.hash_file, 'w') as f:
                json.dump({
                    'hashes': hashes[-max_keep:],
                    'timestamp': datetime.now().isoformat(),
                    'count': len(hashes),
                    'pipeline': self.pipeline_name
                }, f, indent=2)
            self.logger.info(f"[DEDUP] Saved {len(hashes)} hashes for {self.pipeline_name}")
        except Exception as e:
            self.logger.error(f"[DEDUP] Error saving hashes: {e}")
    
    def filter_duplicates(self, items: List[Dict], hash_key_func: Callable) -> tuple:
        """
        Filter duplicates from items
        
        Args:
            items: List of items to check
            hash_key_func: Function that takes an item and returns hash string
        
        Returns:
            (new_items, new_hashes) tuple
        """
        existing_hashes = self.load_hashes()
        new_items = []
        new_hashes = []
        
        for item in items:
            try:
                item_hash = hash_key_func(item)
                if item_hash not in existing_hashes:
                    new_items.append(item)
                    new_hashes.append(item_hash)
            except Exception as e:
                self.logger.error(f"[DEDUP] Error processing item: {e}")
                continue
        
        self.logger.info(f"[DEDUP] Filtered {len(items)} items -> {len(new_items)} new, {len(items) - len(new_items)} duplicates")
        return new_items, new_hashes
    
    def update_hashes(self, new_hashes: List[str]) -> None:
        """
        Add new hashes to existing ones
        
        Args:
            new_hashes: List of new hashes to add
        """
        existing = list(self.load_hashes())
        all_hashes = existing + new_hashes
        self.save_hashes(all_hashes)


__all__ = ['DeduplicationManager']