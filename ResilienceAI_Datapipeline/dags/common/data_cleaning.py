"""
Text processing and cleaning utilities
Handles text cleaning, truncation, and special format processing (HTML, LaTeX)
"""

import re
from typing import Optional


class TextCleaner:
    """A class for cleaning and processing text data"""

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """
        Clean text by removing HTML, extra spaces, and special characters
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text string
            
        Example:
            clean = TextCleaner.clean_text("<p>Hello  World!</p>")
            # Returns: "Hello World!"
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters except basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]]', '', text)

        return text
    
    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to a maximum length
        
        Args:
            text: Text to truncate
            max_length: Maximum length including suffix
            suffix: Suffix to add when truncating (default: "...")
            
        Returns:
            Truncated text
            
        Example:
            short = TextCleaner.truncate("Long text here", 10)
            # Returns: "Long te..."
        """
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def remove_latex(text: str) -> str:
        """
        Remove LaTeX commands (useful for arXiv abstracts)
        
        Args:
            text: Text containing LaTeX commands
            
        Returns:
            Text with LaTeX removed
            
        Example:
            clean = TextCleaner.remove_latex(r"\textbf{Important} text $x^2$")
            # Returns: "Important text x2"
        """
        if not text:
            return ""
        
        # Remove LaTeX commands like \textbf{}, \cite{}, etc.
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove LaTeX special characters
        text = text.replace('$', '').replace('{', '').replace('}', '')
        
        return text.strip()
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """
        Remove HTML tags from text
        
        Args:
            text: Text containing HTML
            
        Returns:
            Text without HTML tags
        """
        if not text:
            return ""
        
        return re.sub(r'<[^>]+>', '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace (multiple spaces, tabs, newlines to single space)
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return ""
        
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """
        Remove URLs from text
        
        Args:
            text: Text containing URLs
            
        Returns:
            Text without URLs
        """
        if not text:
            return ""
        
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def extract_plain_text(text: str) -> str:
        """
        Extract plain text by removing HTML, LaTeX, and normalizing whitespace
        
        Args:
            text: Text to process
            
        Returns:
            Plain text
        """
        if not text:
            return ""
        
        # Remove HTML
        text = TextCleaner.remove_html_tags(text)
        
        # Remove LaTeX
        text = TextCleaner.remove_latex(text)
        
        # Normalize whitespace
        text = TextCleaner.normalize_whitespace(text)
        
        return text


__all__ = ['TextCleaner']