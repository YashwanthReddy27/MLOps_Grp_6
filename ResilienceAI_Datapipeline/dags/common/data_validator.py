"""
Data validation utilities
Handles field validation, URL validation, format checking, and data sanitization
"""

import re
from typing import Dict, List, Optional


class DataValidator:
    """Data validation utilities for pipelines"""

    @staticmethod
    def validate_required_fields(record: Dict, required_fields: List[str]) -> bool:
        """
        Check if record has all required non-empty fields
        
        Args:
            record: Dictionary to validate
            required_fields: List of required field names
            
        Returns:
            True if all required fields exist and are non-empty
            
        Example:
            valid = DataValidator.validate_required_fields(
                {"title": "Test", "url": "http://example.com"},
                ["title", "url"]
            )
            # Returns: True
        """
        for field in required_fields:
            if field not in record or not record[field]:
                return False
        return True
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format
        
        Args:
            url: URL string to validate
            
        Returns:
            True if URL is valid
            
        Example:
            valid = DataValidator.validate_url("https://example.com/article")
            # Returns: True
        """
        if not url:
            return False
        
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url_pattern.match(url) is not None
    
    @staticmethod
    def sanitize_for_db(text: str, max_length: int) -> str:
        """
        Sanitize text for database insertion
        
        Args:
            text: Text to sanitize
            max_length: Maximum length allowed
            
        Returns:
            Sanitized text
            
        Example:
            clean = DataValidator.sanitize_for_db("Text with \x00 null", 50)
            # Returns: "Text with  null" (null bytes removed)
        """
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    @staticmethod
    def validate_arxiv_id(arxiv_id: str) -> bool:
        """
        Validate arXiv ID format
        
        Supports formats:
        - YYMM.NNNNN (e.g., 2301.12345)
        - YYMM.NNNNNN (e.g., 2301.123456)  
        - YYMM.NNNNNvN (e.g., 2510.13809v1) - with version
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
    
    @staticmethod
    def validate_doi(doi: str) -> bool:
        """
        Validate DOI (Digital Object Identifier) format
        
        Args:
            doi: DOI string to validate
            
        Returns:
            True if DOI format is valid
            
        Example:
            valid = DataValidator.validate_doi("10.1234/example.2023")
            # Returns: True
        """
        if not doi:
            return False
        
        # Remove doi: prefix if present
        doi = doi.replace('doi:', '').strip()
        
        # Basic DOI pattern: 10.XXXX/...
        pattern = r'^10\.\d{4,}/[-._;()/:A-Za-z0-9]+$'
        return re.match(pattern, doi) is not None
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format
        
        Args:
            email: Email address to validate
            
        Returns:
            True if email format is valid
        """
        if not email:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_date_format(date_str: str, format_str: str = '%Y-%m-%d') -> bool:
        """
        Validate date string format
        
        Args:
            date_str: Date string to validate
            format_str: Expected date format (default: YYYY-MM-DD)
            
        Returns:
            True if date matches format
            
        Example:
            valid = DataValidator.validate_date_format("2023-01-15")
            # Returns: True
        """
        if not date_str:
            return False
        
        from datetime import datetime
        try:
            datetime.strptime(date_str, format_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_record_schema(record: Dict, schema: Dict[str, type]) -> tuple:
        """
        Validate record against a schema
        
        Args:
            record: Dictionary to validate
            schema: Dict mapping field names to expected types
            
        Returns:
            (is_valid, error_messages) tuple
            
        Example:
            schema = {'title': str, 'year': int, 'published': bool}
            valid, errors = DataValidator.validate_record_schema(record, schema)
        """
        errors = []
        
        for field, expected_type in schema.items():
            if field not in record:
                errors.append(f"Missing required field: {field}")
                continue
            
            value = record[field]
            if value is not None and not isinstance(value, expected_type):
                errors.append(f"Field '{field}' expected type {expected_type.__name__}, got {type(value).__name__}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_numeric_string(value: str) -> Optional[float]:
        """
        Extract numeric value from string
        
        Args:
            value: String containing numeric value
            
        Returns:
            Float value or None if not numeric
            
        Example:
            num = DataValidator.clean_numeric_string("$1,234.56")
            # Returns: 1234.56
        """
        if not value:
            return None
        
        # Remove common non-numeric characters
        cleaned = re.sub(r'[^\d.-]', '', str(value))
        
        try:
            return float(cleaned)
        except ValueError:
            return None


__all__ = ['DataValidator']