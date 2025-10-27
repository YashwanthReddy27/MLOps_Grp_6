"""
Unit tests for data_enrichment.py module
Tests relevance scoring, categorization, and data enhancement
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dags/common'))

from data_enrichment import DataEnricher, CategoryManager


class TestDataEnricher(unittest.TestCase):
    """Test cases for DataEnricher class"""
    
    def test_calculate_relevance_score_basic(self):
        """Test basic relevance score calculation"""
        text = "machine learning and artificial intelligence are important"
        query_terms = ["machine learning", "artificial intelligence"]
        
        score = DataEnricher.calculate_relevance_score(text, query_terms)
        
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_relevance_score_empty_text(self):
        """Test relevance score with empty text"""
        score = DataEnricher.calculate_relevance_score("", ["test"])
        self.assertEqual(score, 0.0)
    
    def test_calculate_relevance_score_empty_terms(self):
        """Test relevance score with empty query terms"""
        score = DataEnricher.calculate_relevance_score("test text", [])
        self.assertEqual(score, 0.0)
    
    def test_calculate_relevance_score_none_inputs(self):
        """Test relevance score with None inputs"""
        score = DataEnricher.calculate_relevance_score(None, ["test"])
        self.assertEqual(score, 0.0)
        
        score = DataEnricher.calculate_relevance_score("test", None)
        self.assertEqual(score, 0.0)
    
    def test_calculate_relevance_score_case_insensitive(self):
        """Test that relevance scoring is case insensitive"""
        text = "Machine Learning and ARTIFICIAL INTELLIGENCE"
        query_terms = ["machine learning", "artificial intelligence"]
        
        score = DataEnricher.calculate_relevance_score(text, query_terms)
        self.assertGreater(score, 0)
    
    def test_calculate_relevance_score_boolean_operators(self):
        """Test that boolean operators are skipped"""
        text = "machine and learning or intelligence not ai"
        query_terms = ["and", "or", "not", "machine"]
        
        score = DataEnricher.calculate_relevance_score(text, query_terms)
        # Only "machine" should contribute to the score
        self.assertGreater(score, 0)
        self.assertLess(score, 0.5)  # Should be relatively low
    
    def test_calculate_relevance_score_max_contribution(self):
        """Test that max contribution per term is capped at 0.3"""
        text = "test " * 10  # "test" appears 10 times
        query_terms = ["test"]
        
        score = DataEnricher.calculate_relevance_score(text, query_terms)
        self.assertAlmostEqual(score, 0.3, places=2)
    
    def test_calculate_relevance_score_max_total(self):
        """Test that total score is capped at 1.0"""
        text = "ai machine learning deep learning neural networks " * 10
        query_terms = ["ai", "machine", "learning", "deep", "neural", "networks"]
        
        score = DataEnricher.calculate_relevance_score(text, query_terms)
        self.assertEqual(score, 1.0)
    
    def test_add_timestamps(self):
        """Test adding timestamps to record"""
        record = {"id": 1, "title": "Test"}
        
        result = DataEnricher.add_timestamps(record)
        
        self.assertIn("processed_at", result)
        self.assertIn("created_at", result)
        
        # Verify timestamp format
        datetime.fromisoformat(result["processed_at"])
        datetime.fromisoformat(result["created_at"])
    
    def test_add_timestamps_preserves_existing_created_at(self):
        """Test that existing created_at is preserved"""
        original_time = "2024-01-01T10:00:00"
        record = {"id": 1, "created_at": original_time}
        
        result = DataEnricher.add_timestamps(record)
        
        self.assertEqual(result["created_at"], original_time)
        self.assertIn("processed_at", result)
    
    def test_extract_keywords_basic(self):
        """Test basic keyword extraction"""
        text = "Machine learning is a subset of artificial intelligence"
        
        keywords = DataEnricher.extract_keywords(text)
        
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)
        self.assertLessEqual(len(keywords), 10)
    
    def test_extract_keywords_filters_stopwords(self):
        """Test that stopwords are filtered"""
        text = "the and for with this that from are was were"
        
        keywords = DataEnricher.extract_keywords(text)
        
        self.assertEqual(len(keywords), 0)
    
    def test_extract_keywords_min_length(self):
        """Test minimum keyword length filter"""
        text = "a b c ab abc abcd"
        
        keywords = DataEnricher.extract_keywords(text, min_length=3)
        
        self.assertIn("abc", keywords)
        self.assertIn("abcd", keywords)
        self.assertNotIn("a", keywords)
        self.assertNotIn("ab", keywords)
    
    def test_extract_keywords_frequency_ordering(self):
        """Test that keywords are ordered by frequency"""
        text = "python python python java java javascript"
        
        keywords = DataEnricher.extract_keywords(text)
        
        self.assertEqual(keywords[0], "python")
        self.assertEqual(keywords[1], "java")
        self.assertEqual(keywords[2], "javascript")
    
    def test_extract_keywords_max_keywords(self):
        """Test max keywords limit"""
        text = " ".join([f"word{i}" for i in range(20)])
        
        keywords = DataEnricher.extract_keywords(text, max_keywords=5)
        
        self.assertEqual(len(keywords), 5)
    
    def test_extract_keywords_punctuation_removal(self):
        """Test punctuation removal from keywords"""
        text = "python, java. javascript! c++ (golang)"
        
        keywords = DataEnricher.extract_keywords(text)
        
        self.assertIn("python", keywords)
        self.assertIn("java", keywords)
        self.assertIn("javascript", keywords)
        self.assertIn("golang", keywords)


class TestCategoryManager(unittest.TestCase):
    """Test cases for CategoryManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.categories_config = {
            'artificial_intelligence': {
                'keywords': ['ai', 'machine learning', 'deep learning'],
                'weight': 1.0
            },
            'computer_vision': {
                'keywords': ['image', 'vision', 'object detection'],
                'weight': 0.8
            },
            'nlp': {
                'keywords': ['natural language', 'text', 'nlp'],
                'weight': 0.9
            }
        }
    
    def test_categorize_content_single_match(self):
        """Test categorization with single category match"""
        text = "This article is about computer vision and image processing"
        
        result = CategoryManager.categorize_content(text, self.categories_config)
        
        self.assertEqual(result['primary_category'], 'computer_vision')
        self.assertIn('computer_vision', result['all_categories'])
        self.assertGreater(result['overall_relevance'], 0)
    
    def test_categorize_content_multiple_matches(self):
        """Test categorization with multiple category matches"""
        # Make the text more explicitly match multiple categories
        text = "AI and machine learning for natural language processing NLP text analysis"
        
        result = CategoryManager.categorize_content(text, self.categories_config)
        
        self.assertIn('artificial_intelligence', result['all_categories'])
        self.assertIn('nlp', result['all_categories'])
        self.assertGreater(len(result['all_categories']), 1)
    
    def test_categorize_content_no_match(self):
        """Test categorization with no matches"""
        text = "This is about quantum physics and chemistry"
        
        result = CategoryManager.categorize_content(text, self.categories_config)
        
        self.assertEqual(result['primary_category'], 'general')
        self.assertEqual(len(result['all_categories']), 0)
        self.assertEqual(result['overall_relevance'], 0.0)
    
    def test_categorize_content_weighted_scores(self):
        """Test that weights affect categorization"""
        text = "image processing and ai"  # Both mentioned once
        
        result = CategoryManager.categorize_content(text, self.categories_config)
        
        # AI has higher weight (1.0 vs 0.8), so should be primary
        ai_score = result['category_scores'].get('artificial_intelligence', 0)
        cv_score = result['category_scores'].get('computer_vision', 0)
        
        if ai_score > 0 and cv_score > 0:
            self.assertGreater(ai_score, cv_score)
    
    def test_categorize_content_case_insensitive(self):
        """Test case-insensitive matching"""
        text = "AI and MACHINE LEARNING with Deep Learning"
        
        result = CategoryManager.categorize_content(text, self.categories_config)
        
        self.assertIn('artificial_intelligence', result['all_categories'])
        self.assertGreater(result['overall_relevance'], 0)
    
    def test_categorize_content_threshold(self):
        """Test that low scores don't result in categorization"""
        # Create a text with very weak relevance
        text = "Some random text with minimal relevance"
        
        weak_config = {
            'test_category': {
                'keywords': ['xyz'],  # Not in text
                'weight': 1.0
            }
        }
        
        result = CategoryManager.categorize_content(text, weak_config)
        
        self.assertEqual(result['primary_category'], 'general')
        self.assertEqual(len(result['all_categories']), 0)
    
    def test_get_category_distribution(self):
        """Test category distribution calculation"""
        items = [
            {'primary_category': 'ai'},
            {'primary_category': 'ai'},
            {'primary_category': 'nlp'},
            {'primary_category': 'cv'},
            {'primary_category': 'ai'},
        ]
        
        distribution = CategoryManager.get_category_distribution(items)
        
        self.assertEqual(distribution['ai'], 3)
        self.assertEqual(distribution['nlp'], 1)
        self.assertEqual(distribution['cv'], 1)
    
    def test_get_category_distribution_empty_list(self):
        """Test distribution with empty list"""
        distribution = CategoryManager.get_category_distribution([])
        
        self.assertEqual(len(distribution), 0)
    
    def test_get_category_distribution_missing_field(self):
        """Test distribution with missing category field"""
        items = [
            {'primary_category': 'ai'},
            {},  # Missing field
            {'primary_category': 'nlp'},
        ]
        
        distribution = CategoryManager.get_category_distribution(items)
        
        self.assertEqual(distribution['ai'], 1)
        self.assertEqual(distribution['nlp'], 1)
        self.assertEqual(distribution['unknown'], 1)
    
    def test_get_category_distribution_custom_field(self):
        """Test distribution with custom field name"""
        items = [
            {'category': 'ai'},
            {'category': 'ai'},
            {'category': 'nlp'},
        ]
        
        distribution = CategoryManager.get_category_distribution(items, 'category')
        
        self.assertEqual(distribution['ai'], 2)
        self.assertEqual(distribution['nlp'], 1)


class TestIntegrationCategorization(unittest.TestCase):
    """Integration tests for categorization with real-world examples"""
    
    def setUp(self):
        """Set up real-world category configuration"""
        self.real_config = {
            'language_models': {
                'keywords': ['GPT', 'transformer', 'BERT', 'language model'],
                'weight': 0.95
            },
            'computer_vision': {
                'keywords': ['YOLO', 'object detection', 'image segmentation'],
                'weight': 0.8
            },
            'reinforcement_learning': {
                'keywords': ['Q-learning', 'PPO', 'reward', 'agent'],
                'weight': 0.85
            }
        }
    
    def test_arxiv_paper_categorization(self):
        """Test categorization of typical arXiv paper abstract"""
        abstract = """
        We present a new transformer-based language model that achieves 
        state-of-the-art results on various NLP benchmarks. Our model, 
        similar to GPT and BERT architectures, uses attention mechanisms
        to process sequential data efficiently.
        """
        
        result = CategoryManager.categorize_content(abstract, self.real_config)
        
        self.assertEqual(result['primary_category'], 'language_models')
        self.assertGreater(result['overall_relevance'], 0.1)
    
    def test_news_article_categorization(self):
        """Test categorization of tech news article"""
        article = """
        New breakthrough in object detection: Researchers develop an improved
        YOLO algorithm that can perform real-time image segmentation with
        unprecedented accuracy. This computer vision model could revolutionize
        autonomous driving.
        """
        
        result = CategoryManager.categorize_content(article, self.real_config)
        
        self.assertEqual(result['primary_category'], 'computer_vision')
        self.assertIn('computer_vision', result['all_categories'])


if __name__ == '__main__':
    unittest.main()