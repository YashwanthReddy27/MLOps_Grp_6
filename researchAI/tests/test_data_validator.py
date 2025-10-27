"""
Unit tests for data_validator.py module
Tests field validation, URL validation, format checking and data sanitization
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dags/common'))

from data_validator import DataValidator

class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class"""
    
    def test_validate_arxiv_id_valid_formats(self):
        """Test validation of valid arXiv ID formats"""
        valid_ids = [
            "2401.00001",       # Standard format YYMM.NNNNN
            "2312.123456",      # 6 digits
            "2401.00001v1",     # With version
            "2401.00001v12",    # With double-digit version
            "arxiv:2401.00001", # With prefix
            "arxiv:2401.00001v2", # With prefix and version
            "0801.1234",        # Older format
        ]
        
        for arxiv_id in valid_ids:
            with self.subTest(arxiv_id=arxiv_id):
                result = DataValidator.validate_arxiv_id(arxiv_id)
                self.assertTrue(result, f"Failed for valid ID: {arxiv_id}")
    
    def test_validate_arxiv_id_invalid_formats(self):
        """Test validation of invalid arXiv ID formats"""
        invalid_ids = [
            "",                 # Empty
            None,              # None
            "invalid",         # Random text
            "24.00001",        # Wrong year format
            "240100001",       # Missing dot
            "2401.001",        # Too few digits after dot
            "2401.00001.123",  # Extra dot
            "2401.00001v",     # Version without number
            "2401.00001vX",    # Version with non-digit
            "24010.0001",      # Wrong position of dot
            "abcd.12345",      # Non-numeric year/month
        ]
        
        for arxiv_id in invalid_ids:
            with self.subTest(arxiv_id=arxiv_id):
                result = DataValidator.validate_arxiv_id(arxiv_id)
                self.assertFalse(result, f"Failed for invalid ID: {arxiv_id}")
    
    def test_clean_arxiv_id_removes_prefix(self):
        """Test that arXiv prefix is removed"""
        test_cases = [
            ("arxiv:2401.00001", "2401.00001"),
            ("2401.00001", "2401.00001"),  # No prefix
            ("arxiv:2401.00001v2", "2401.00001"),  # Also removes version
        ]
        
        for input_id, expected in test_cases:
            with self.subTest(input=input_id):
                result = DataValidator.clean_arxiv_id(input_id)
                self.assertEqual(result, expected)
    
    def test_clean_arxiv_id_removes_version(self):
        """Test that version suffix is removed"""
        test_cases = [
            ("2401.00001v1", "2401.00001"),
            ("2401.00001v12", "2401.00001"),
            ("2401.00001", "2401.00001"),  # No version
        ]
        
        for input_id, expected in test_cases:
            with self.subTest(input=input_id):
                result = DataValidator.clean_arxiv_id(input_id)
                self.assertEqual(result, expected)
    
    def test_clean_arxiv_id_handles_both(self):
        """Test cleaning both prefix and version"""
        input_id = "arxiv:2401.00001v3"
        expected = "2401.00001"
        
        result = DataValidator.clean_arxiv_id(input_id)
        self.assertEqual(result, expected)
    
    def test_clean_arxiv_id_handles_none(self):
        """Test cleaning None input"""
        result = DataValidator.clean_arxiv_id(None)
        self.assertEqual(result, "")
    
    def test_clean_arxiv_id_handles_empty_string(self):
        """Test cleaning empty string"""
        result = DataValidator.clean_arxiv_id("")
        self.assertEqual(result, "")
    
    def test_clean_arxiv_id_strips_whitespace(self):
        """Test that whitespace is stripped"""
        input_id = "  2401.00001  "
        expected = "2401.00001"
        
        result = DataValidator.clean_arxiv_id(input_id)
        self.assertEqual(result, expected)
    
    def test_clean_arxiv_id_edge_cases(self):
        """Test edge cases for arXiv ID cleaning"""
        test_cases = [
            ("arxiv:arxiv:2401.00001", "2401.00001"),  # Actually removes both prefixes
            ("2401.00001vv2", "2401.00001vv2"),  # Invalid version format
            ("v22401.00001", "v22401.00001"),  # Version at start
            ("2401v2.00001", "2401v2.00001"),  # Version in wrong place
        ]
        
        for input_id, expected in test_cases:
            with self.subTest(input=input_id):
                result = DataValidator.clean_arxiv_id(input_id)
                self.assertEqual(result, expected)


class TestDataValidatorIntegration(unittest.TestCase):
    """Integration tests for DataValidator with real-world scenarios"""
    
    def test_arxiv_validation_pipeline(self):
        """Test full validation pipeline for arXiv IDs"""
        # Simulate incoming data with various formats
        papers = [
            {"arxiv_id": "arxiv:2401.00001v3", "title": "Paper 1"},
            {"arxiv_id": "2312.123456", "title": "Paper 2"},
            {"arxiv_id": "invalid_id", "title": "Paper 3"},
            {"arxiv_id": None, "title": "Paper 4"},
            {"arxiv_id": "  2401.00002v1  ", "title": "Paper 5"},
        ]
        
        validated_papers = []
        for paper in papers:
            raw_id = paper.get("arxiv_id", "")
            clean_id = DataValidator.clean_arxiv_id(raw_id)
            
            if DataValidator.validate_arxiv_id(clean_id):
                validated_papers.append({
                    "arxiv_id": clean_id,
                    "arxiv_id_original": raw_id,
                    "title": paper["title"]
                })
        
        # Should have 3 valid papers (1, 2, and 5)
        self.assertEqual(len(validated_papers), 3)
        
        # Verify cleaned IDs
        self.assertEqual(validated_papers[0]["arxiv_id"], "2401.00001")
        self.assertEqual(validated_papers[1]["arxiv_id"], "2312.123456")
        self.assertEqual(validated_papers[2]["arxiv_id"], "2401.00002")
    
    def test_bulk_arxiv_validation_performance(self):
        """Test performance with large batch of IDs"""
        # Generate 1000 IDs for performance testing
        import time
        
        ids = [f"2401.{str(i).zfill(5)}" for i in range(1000)]
        
        start_time = time.time()
        
        for arxiv_id in ids:
            clean_id = DataValidator.clean_arxiv_id(arxiv_id)
            DataValidator.validate_arxiv_id(clean_id)
        
        elapsed_time = time.time() - start_time
        
        # Should process 1000 IDs quickly (under 1 second)
        self.assertLess(elapsed_time, 1.0, f"Processing took {elapsed_time:.2f}s")


if __name__ == '__main__':
    unittest.main()