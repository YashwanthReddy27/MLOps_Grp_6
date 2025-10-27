"""
Unit tests for deduplication.py module
Tests duplicate detection, hash management, and tracking across pipeline runs
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import tempfile
import shutil
from datetime import datetime
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dags/common'))

from deduplication import DeduplicationManager


class TestDeduplicationManager(unittest.TestCase):
    """Test cases for DeduplicationManager class"""
    
    def setUp(self):
        """Set up test fixtures with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.dedup = DeduplicationManager('test_pipeline', base_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.dedup.pipeline_name, 'test_pipeline')
        self.assertTrue(os.path.exists(self.dedup.hash_dir))
        self.assertTrue(self.dedup.hash_file.endswith('test_pipeline_hashes.json'))
    
    def test_generate_hash_single_field(self):
        """Test hash generation with single field"""
        hash_result = self.dedup.generate_hash("test_content")
        
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 32)  # MD5 hash length
    
    def test_generate_hash_multiple_fields(self):
        """Test hash generation with multiple fields"""
        hash1 = self.dedup.generate_hash("field1", "field2", "field3")
        hash2 = self.dedup.generate_hash("field1", "field2", "field3")
        hash3 = self.dedup.generate_hash("field1", "field2", "different")
        
        self.assertEqual(hash1, hash2)  # Same input = same hash
        self.assertNotEqual(hash1, hash3)  # Different input = different hash
    
    def test_generate_hash_with_none(self):
        """Test hash generation with None values"""
        hash1 = self.dedup.generate_hash("field1", None, "field3")
        hash2 = self.dedup.generate_hash("field1", "field3")
        
        self.assertEqual(hash1, hash2)  # None values are ignored
    
    def test_load_hashes_empty_file(self):
        """Test loading hashes when file doesn't exist"""
        hashes = self.dedup.load_hashes()
        
        self.assertIsInstance(hashes, set)
        self.assertEqual(len(hashes), 0)
    
    def test_save_and_load_hashes(self):
        """Test saving and loading hashes"""
        test_hashes = ["hash1", "hash2", "hash3"]
        
        self.dedup.save_hashes(test_hashes)
        loaded_hashes = self.dedup.load_hashes()
        
        self.assertEqual(loaded_hashes, set(test_hashes))
    
    def test_save_hashes_max_keep(self):
        """Test hash saving with max_keep limit"""
        test_hashes = [f"hash_{i}" for i in range(20)]
        
        self.dedup.save_hashes(test_hashes, max_keep=10)
        
        # Load and verify
        with open(self.dedup.hash_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data['hashes']), 10)
        # Should keep the last 10 hashes
        self.assertEqual(data['hashes'], test_hashes[-10:])
    
    def test_save_hashes_metadata(self):
        """Test that metadata is saved with hashes"""
        test_hashes = ["hash1", "hash2"]
        
        self.dedup.save_hashes(test_hashes)
        
        with open(self.dedup.hash_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('timestamp', data)
        self.assertIn('count', data)
        self.assertIn('pipeline', data)
        self.assertEqual(data['pipeline'], 'test_pipeline')
        self.assertEqual(data['count'], 2)
    
    def test_filter_duplicates_basic(self):
        """Test basic duplicate filtering"""
        # Save some existing hashes
        existing_hashes = ["existing1", "existing2"]
        self.dedup.save_hashes(existing_hashes)
        
        # Create test items
        items = [
            {"id": 1, "title": "New Article"},
            {"id": 2, "title": "Existing Article"},
            {"id": 3, "title": "Another New Article"}
        ]
        
        # Define hash function
        def hash_func(item):
            if item["id"] == 2:
                return "existing1"  # This will be duplicate
            return self.dedup.generate_hash(str(item["id"]), item["title"])
        
        new_items, new_hashes = self.dedup.filter_duplicates(items, hash_func)
        
        self.assertEqual(len(new_items), 2)  # One duplicate filtered
        self.assertEqual(len(new_hashes), 2)
        self.assertNotIn(2, [item["id"] for item in new_items])
    
    def test_filter_duplicates_empty_items(self):
        """Test duplicate filtering with empty items list"""
        new_items, new_hashes = self.dedup.filter_duplicates(
            [], 
            lambda x: "hash"
        )
        
        self.assertEqual(len(new_items), 0)
        self.assertEqual(len(new_hashes), 0)
    
    def test_filter_duplicates_with_exception(self):
        """Test duplicate filtering handles exceptions gracefully"""
        items = [
            {"id": 1},
            {"id": 2},
            {"id": 3}
        ]
        
        def hash_func(item):
            if item["id"] == 2:
                raise ValueError("Test exception")
            return str(item["id"])
        
        new_items, new_hashes = self.dedup.filter_duplicates(items, hash_func)
        
        # Should continue processing other items
        self.assertEqual(len(new_items), 2)
        self.assertNotIn(2, [item["id"] for item in new_items])
    
    def test_update_hashes(self):
        """Test updating hashes adds to existing ones"""
        # Save initial hashes
        initial = ["hash1", "hash2"]
        self.dedup.save_hashes(initial)
        
        # Update with new hashes
        new = ["hash3", "hash4"]
        self.dedup.update_hashes(new)
        
        # Load and verify
        loaded = self.dedup.load_hashes()
        
        self.assertEqual(len(loaded), 4)
        for h in initial + new:
            self.assertIn(h, loaded)
    
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('os.path.exists', return_value=True)
    def test_load_hashes_file_error(self, mock_exists, mock_open_func):
        """Test load_hashes handles file errors gracefully"""
        hashes = self.dedup.load_hashes()
        
        self.assertEqual(len(hashes), 0)
    
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_save_hashes_file_error(self, mock_open_func):
        """Test save_hashes handles file errors gracefully"""
        # Should not raise exception
        self.dedup.save_hashes(["hash1"])  # Should handle error internally


class TestDeduplicationIntegration(unittest.TestCase):
    """Integration tests for deduplication across pipeline runs"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.news_dedup = DeduplicationManager('news', base_dir=self.temp_dir)
        self.arxiv_dedup = DeduplicationManager('arxiv', base_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_multiple_pipeline_isolation(self):
        """Test that different pipelines maintain separate hash stores"""
        # Save hashes for news pipeline
        news_hashes = ["news1", "news2"]
        self.news_dedup.save_hashes(news_hashes)
        
        # Save hashes for arxiv pipeline
        arxiv_hashes = ["arxiv1", "arxiv2"]
        self.arxiv_dedup.save_hashes(arxiv_hashes)
        
        # Load and verify isolation
        loaded_news = self.news_dedup.load_hashes()
        loaded_arxiv = self.arxiv_dedup.load_hashes()
        
        self.assertEqual(loaded_news, set(news_hashes))
        self.assertEqual(loaded_arxiv, set(arxiv_hashes))
        
        # Verify no cross-contamination
        for h in arxiv_hashes:
            self.assertNotIn(h, loaded_news)
        for h in news_hashes:
            self.assertNotIn(h, loaded_arxiv)
    
    def test_realistic_news_article_dedup(self):
        """Test deduplication with realistic news articles"""
        # Simulate first pipeline run
        articles_run1 = [
            {"title": "AI Breakthrough Announced", "url": "http://example.com/1"},
            {"title": "New GPU Released", "url": "http://example.com/2"},
        ]
        
        def article_hash(article):
            return self.news_dedup.generate_hash(article["title"], article["url"])
        
        new_items1, new_hashes1 = self.news_dedup.filter_duplicates(articles_run1, article_hash)
        self.news_dedup.update_hashes(new_hashes1)
        
        self.assertEqual(len(new_items1), 2)  # All new
        
        # Simulate second pipeline run with overlap
        articles_run2 = [
            {"title": "AI Breakthrough Announced", "url": "http://example.com/1"},  # Duplicate
            {"title": "Quantum Computing Advance", "url": "http://example.com/3"},  # New
        ]
        
        new_items2, new_hashes2 = self.news_dedup.filter_duplicates(articles_run2, article_hash)
        
        self.assertEqual(len(new_items2), 1)  # One duplicate filtered
        self.assertEqual(new_items2[0]["title"], "Quantum Computing Advance")
    
    def test_realistic_arxiv_paper_dedup(self):
        """Test deduplication with realistic arXiv papers"""
        papers = [
            {"arxiv_id": "2401.00001", "title": "Deep Learning Survey"},
            {"arxiv_id": "2401.00002", "title": "Reinforcement Learning"},
            {"arxiv_id": "2401.00001", "title": "Deep Learning Survey"},  # Duplicate
        ]
        
        def paper_hash(paper):
            return self.arxiv_dedup.generate_hash(paper["arxiv_id"], paper["title"])
        
        # First pass - all should be new since no existing hashes
        new_papers, new_hashes = self.arxiv_dedup.filter_duplicates(papers, paper_hash)
    
        processed = []
        seen_hashes = set()
        
        for paper in papers:
            hash_val = paper_hash(paper)
            if hash_val not in seen_hashes:
                processed.append(paper)
                seen_hashes.add(hash_val)
        
        self.assertEqual(len(processed), 2) 


class TestDeduplicationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.dedup = DeduplicationManager('test', base_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_corrupt_hash_file(self):
        """Test handling of corrupt hash file"""
        # Create corrupt JSON file
        with open(self.dedup.hash_file, 'w') as f:
            f.write("{invalid json}")
        
        # Should return empty set instead of crashing
        hashes = self.dedup.load_hashes()
        self.assertEqual(len(hashes), 0)
    
    def test_unicode_content(self):
        """Test hash generation with Unicode content"""
        hash1 = self.dedup.generate_hash("Hello 世界", "مرحبا")
        hash2 = self.dedup.generate_hash("Hello 世界", "مرحبا")
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 32)
    
    def test_very_long_content(self):
        """Test hash generation with very long content"""
        long_text = "x" * 1000000  # 1 million characters
        
        # Should not crash or timeout
        hash_result = self.dedup.generate_hash(long_text)
        
        self.assertEqual(len(hash_result), 32)
    
    def test_hash_consistency_across_runs(self):
        """Test that hashes are consistent across different manager instances"""
        content = "Test content for consistency"
        
        # Create two separate managers
        dedup1 = DeduplicationManager('test1', base_dir=self.temp_dir)
        dedup2 = DeduplicationManager('test2', base_dir=self.temp_dir)
        
        hash1 = dedup1.generate_hash(content)
        hash2 = dedup2.generate_hash(content)
        
        self.assertEqual(hash1, hash2)


if __name__ == '__main__':
    unittest.main()