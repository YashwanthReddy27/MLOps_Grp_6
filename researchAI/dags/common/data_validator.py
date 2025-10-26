"""
Data validation utilities
Handles field validation, URL validation, format checking and data sanitization
"""

import re
from typing import Dict, List, Optional


class DataValidator:
    """Data validation utilities for pipelines"""
    
    
    @staticmethod
    def validate_arxiv_id(arxiv_id: str) -> bool:
        """
        Validate arXiv ID format
        
        Supports formats:
        - YYMM.NNNNN
        - YYMM.NNNNNN 
        - YYMM.NNNNNvN 
        - arxiv:YYMM.NNNNN
        """
        if not arxiv_id:
            return False
        
        # Remove arxiv: prefix if present
        arxiv_id = arxiv_id.replace('arxiv:', '').strip()
        
        # Remove version suffix (vN) if present
        if 'v' in arxiv_id:
            parts = arxiv_id.split('v')
            if len(parts) == 2 and parts[1].isdigit():
                arxiv_id = parts[0]
        
        # Match YYMM.NNNNN or YYMM.NNNNNN format (4 digits, dot, 4-6 digits)
        pattern = r'^\d{4}\.\d{4,6}$'
        return re.match(pattern, arxiv_id) is not None

    @staticmethod
    def clean_arxiv_id(arxiv_id: str) -> str:
        """
        Clean and normalize arXiv ID by removing prefix and version suffix
        """
        if not arxiv_id:
            return ""
        
        # Remove arxiv: prefix if present
        cleaned = arxiv_id.replace('arxiv:', '').strip()
        
        # Remove version suffix (vN) if present
        if 'v' in cleaned:
            parts = cleaned.split('v')
            if len(parts) == 2 and parts[1].isdigit():
                cleaned = parts[0]
        
        return cleaned

__all__ = ['DataValidator']