"""
Data enrichment utilities
Handles relevance scoring, categorization, and data enhancement
"""

from datetime import datetime
from typing import Dict, List


class DataEnricher:
    """Data enrichment and scoring utilities"""
    
    @staticmethod
    def calculate_relevance_score(text: str, query_terms: List[str]) -> float:
        """
        Calculate relevance score (0-1) based on query term matches
        
        Args:
            text: Text to score (should be lowercase)
            query_terms: List of query terms/keywords
        
        Returns:
            Relevance score between 0 and 1
        """
        if not text or not query_terms:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Skip boolean operators
            if term_lower in ['or', 'and', 'not']:
                continue
            
            # Count occurrences (max contribution per term = 0.3)
            count = text_lower.count(term_lower)
            score += min(count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    @staticmethod
    def add_timestamps(record: Dict) -> Dict:
        """
        Add standard timestamp fields to record
        
        Args:
            record: Dictionary record to enhance
            
        Returns:
            Enhanced record with timestamps
        """
        now = datetime.now().isoformat()
        record['processed_at'] = now
        if 'created_at' not in record:
            record['created_at'] = now
        return record
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
        """
        Extract potential keywords from text
        
        Args:
            text: Text to extract keywords from
            min_length: Minimum keyword length (default: 3)
            max_keywords: Maximum number of keywords to return (default: 10)
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
        
        # Simple word extraction (can be enhanced with NLP)
        words = text.lower().split()
        
        # Filter by length and common stop words
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'are', 'was', 'were'}
        keywords = [
            word.strip('.,!?;:()[]{}') 
            for word in words 
            if len(word) >= min_length and word.lower() not in stop_words
        ]
        
        # Count frequency
        keyword_freq = {}
        for kw in keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        # Sort by frequency and return top N
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_keywords[:max_keywords]]


class CategoryManager:
    """Manage multi-category classification for articles/papers"""
    
    @staticmethod
    def categorize_content(text: str, categories_config: Dict[str, Dict]) -> Dict:
        """
        Categorize content based on keywords configuration
        
        Args:
            text: Text to categorize (title + description/abstract)
            categories_config: Dict with format:
                {
                    'category_name': {
                        'keywords': ['keyword1', 'keyword2'],
                        'weight': 1.0
                    }
                }
        
        Returns:
            Dict with categorization results:
                {
                    'primary_category': str,
                    'all_categories': List[str],
                    'category_scores': Dict[str, float],
                    'overall_relevance': float
                }
        """
        text_lower = text.lower()
        matched_categories = []
        category_scores = {}
        
        for category_name, category_data in categories_config.items():
            keywords = category_data.get('keywords', [])
            weight = category_data.get('weight', 1.0)
            
            # Calculate relevance score
            relevance_score = DataEnricher.calculate_relevance_score(text_lower, keywords)
            weighted_score = relevance_score * weight
            
            # If content matches category (score > threshold)
            if weighted_score > 0.1:  # Adjust threshold as needed
                matched_categories.append(category_name)
                category_scores[category_name] = round(weighted_score, 3)
        
        # Determine primary category (highest score)
        primary_category = max(category_scores, key=category_scores.get) if category_scores else 'general'
        
        # Calculate overall relevance (max score across all categories)
        overall_relevance = max(category_scores.values()) if category_scores else 0.0
        
        return {
            'primary_category': primary_category,
            'all_categories': matched_categories,
            'category_scores': category_scores,
            'overall_relevance': round(overall_relevance, 3)
        }
    
    @staticmethod
    def get_category_distribution(items: List[Dict], category_field: str = 'primary_category') -> Dict[str, int]:
        """
        Get distribution of categories across items
        
        Args:
            items: List of items with category information
            category_field: Field name containing category (default: 'primary_category')
            
        Returns:
            Dictionary with category counts
        """
        distribution = {}
        for item in items:
            category = item.get(category_field, 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution


__all__ = ['DataEnricher', 'CategoryManager']