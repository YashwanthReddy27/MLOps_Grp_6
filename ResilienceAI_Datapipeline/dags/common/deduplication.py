"""
Deduplication utilities for data pipelines
Manages duplicate detection across pipeline runs using hash-based tracking
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Callable


class DeduplicationManager:
    """Manages deduplication across pipelines"""
    
    def __init__(self, pipeline_name: str, base_dir: str = '/opt/airflow/data'):
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
    
    def generate_hash(self, *fields) -> str:
        """
        Generate MD5 hash from fields
        
        Args:
            *fields: Variable number of fields to hash
            
        Returns:
            MD5 hash string
            
        Example:
            hash = dedup.generate_hash(title, url)
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
                print(f"[DEDUP] Error loading hashes: {e}")
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
            print(f"[DEDUP] Saved {len(hashes)} hashes for {self.pipeline_name}")
        except Exception as e:
            print(f"[DEDUP] Error saving hashes: {e}")
    
    def filter_duplicates(self, items: List[Dict], hash_key_func: Callable) -> tuple:
        """
        Filter duplicates from items
        
        Args:
            items: List of items to check
            hash_key_func: Function that takes an item and returns hash string
        
        Returns:
            (new_items, new_hashes) tuple
            
        Example:
            new_articles, new_hashes = dedup.filter_duplicates(
                articles,
                lambda article: dedup.generate_hash(
                    article.get('title'),
                    article.get('url')
                )
            )
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
                print(f"[DEDUP] Error processing item: {e}")
                continue
        
        print(f"[DEDUP] Filtered {len(items)} items -> {len(new_items)} new, {len(items) - len(new_items)} duplicates")
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
    
    def clear_hashes(self) -> None:
        """Clear all stored hashes for this pipeline"""
        if os.path.exists(self.hash_file):
            os.remove(self.hash_file)
            print(f"[DEDUP] Cleared all hashes for {self.pipeline_name}")
    
    def get_hash_count(self) -> int:
        """Get count of stored hashes"""
        return len(self.load_hashes())


__all__ = ['DeduplicationManager']