"""
Text processing and cleaning utilities
Handles text cleaning, truncation and special format processing (HTML, LaTeX)
"""

import re
from typing import Optional


class TextCleaner:
    """A class for cleaning and processing text data"""

    def clean_text(self, text: Optional[str]) -> str:
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
    
    
    def remove_latex(self, text: str) -> str:
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


__all__ = ['TextCleaner']