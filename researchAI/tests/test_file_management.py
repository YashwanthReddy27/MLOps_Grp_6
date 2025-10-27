"""
Unit tests for file_management.py module
Tests file system operations, JSON operations, and file cleanup
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta
import sys

# Add the common directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dags/common'))

from file_management import FileManager

class TestFileManager(unittest.TestCase):
    """Test cases for FileManager class"""
    
    def setUp(self):
        """Set up test fixtures with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.file_manager = FileManager()
    
    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_ensure_directories_creates_structure(self):
        """Test that ensure_directories creates the expected structure"""
        FileManager.ensure_directories(self.temp_dir)
        
        expected_dirs = [
            os.path.join(self.temp_dir, 'raw'),
            os.path.join(self.temp_dir, 'cleaned'),
            os.path.join(self.temp_dir, 'schema'),
            os.path.join(self.temp_dir, 'hashes'),
        ]
        
        for directory in expected_dirs:
            self.assertTrue(os.path.exists(directory), f"Directory {directory} not created")
            self.assertTrue(os.path.isdir(directory), f"{directory} is not a directory")
    
    def test_ensure_directories_idempotent(self):
        """Test that ensure_directories can be called multiple times safely"""
        FileManager.ensure_directories(self.temp_dir)
        
        # Create a file in one of the directories
        test_file = os.path.join(self.temp_dir, 'raw', 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')
        
        # Call again - should not raise error
        FileManager.ensure_directories(self.temp_dir)
        
        # File should still exist
        self.assertTrue(os.path.exists(test_file))
    
    @patch('os.chmod')
    def test_ensure_directories_sets_permissions(self, mock_chmod):
        """Test that directories get correct permissions"""
        FileManager.ensure_directories(self.temp_dir)
        
        # Should attempt to set 777 permissions on each directory
        self.assertEqual(mock_chmod.call_count, 4)
        for call_args in mock_chmod.call_args_list:
            args, _ = call_args
            self.assertEqual(args[1], 0o777)
    
    def test_save_json_basic(self):
        """Test basic JSON saving"""
        data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}
        filepath = os.path.join(self.temp_dir, 'test.json')
        
        FileManager.save_json(data, filepath)
        
        self.assertTrue(os.path.exists(filepath))
        
        # Verify content
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, data)
    
    def test_save_json_creates_parent_directory(self):
        """Test that save_json creates parent directories if needed"""
        data = {'test': 'data'}
        filepath = os.path.join(self.temp_dir, 'subdir', 'nested', 'test.json')
        
        FileManager.save_json(data, filepath)
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(os.path.exists(os.path.dirname(filepath)))
    
    def test_save_json_with_datetime(self):
        """Test saving JSON with datetime objects"""
        data = {
            'timestamp': datetime.now(),
            'date': datetime.now().date()
        }
        filepath = os.path.join(self.temp_dir, 'datetime_test.json')
        
        FileManager.save_json(data, filepath)
        
        # Should not raise error due to default=str
        self.assertTrue(os.path.exists(filepath))
        
        # Load and verify it's a string now
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertIsInstance(loaded_data['timestamp'], str)
    
    def test_save_json_unicode(self):
        """Test saving JSON with Unicode characters"""
        data = {
            'chinese': '你好世界',
            'arabic': 'مرحبا بالعالم',
            'emoji': '🚀🔥'
        }
        filepath = os.path.join(self.temp_dir, 'unicode_test.json')
        
        FileManager.save_json(data, filepath)
        
        # Load and verify Unicode is preserved
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, data)
    
    def test_save_json_custom_indent(self):
        """Test saving JSON with custom indentation"""
        data = {'key': 'value'}
        filepath = os.path.join(self.temp_dir, 'indent_test.json')
        
        FileManager.save_json(data, filepath, indent=4)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check that indentation is present
        self.assertIn('    ', content)  # 4 spaces
    
    def test_load_json_basic(self):
        """Test basic JSON loading"""
        data = {'key': 'value', 'number': 42}
        filepath = os.path.join(self.temp_dir, 'test.json')
        
        # Save first
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        # Load
        loaded_data = FileManager.load_json(filepath)
        
        self.assertEqual(loaded_data, data)
    
    def test_load_json_nonexistent_file(self):
        """Test loading non-existent file raises appropriate error"""
        filepath = os.path.join(self.temp_dir, 'nonexistent.json')
        
        with self.assertRaises(FileNotFoundError):
            FileManager.load_json(filepath)
    
    def test_load_json_invalid_json(self):
        """Test loading invalid JSON raises appropriate error"""
        filepath = os.path.join(self.temp_dir, 'invalid.json')
        
        with open(filepath, 'w') as f:
            f.write('{ invalid json }')
        
        with self.assertRaises(json.JSONDecodeError):
            FileManager.load_json(filepath)
    
    def test_cleanup_old_files_basic(self):
        """Test basic file cleanup"""
        # Create test files with different ages
        now = datetime.now()
        
        # Create old file (should be deleted)
        old_file = os.path.join(self.temp_dir, 'test_old.txt')
        with open(old_file, 'w') as f:
            f.write('old content')
        
        # Set modification time to 10 days ago
        old_time = (now - timedelta(days=10)).timestamp()
        os.utime(old_file, (old_time, old_time))
        
        # Create recent file (should be kept)
        new_file = os.path.join(self.temp_dir, 'test_new.txt')
        with open(new_file, 'w') as f:
            f.write('new content')
        
        # Clean files older than 7 days
        deleted = FileManager.cleanup_old_files(self.temp_dir, 'test_', days=7)
        
        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.exists(old_file))
        self.assertTrue(os.path.exists(new_file))
    
    def test_cleanup_old_files_pattern_matching(self):
        """Test cleanup with pattern matching"""
        # Create files with different patterns
        match_file1 = os.path.join(self.temp_dir, 'news_data_1.json')
        match_file2 = os.path.join(self.temp_dir, 'news_data_2.json')
        no_match_file = os.path.join(self.temp_dir, 'arxiv_data.json')
        
        for filepath in [match_file1, match_file2, no_match_file]:
            with open(filepath, 'w') as f:
                f.write('content')
            
            # Make them old
            old_time = (datetime.now() - timedelta(days=10)).timestamp()
            os.utime(filepath, (old_time, old_time))
        
        # Clean only files matching 'news_'
        deleted = FileManager.cleanup_old_files(self.temp_dir, 'news_', days=7)
        
        self.assertEqual(deleted, 2)
        self.assertFalse(os.path.exists(match_file1))
        self.assertFalse(os.path.exists(match_file2))
        self.assertTrue(os.path.exists(no_match_file))
    
    def test_cleanup_old_files_nonexistent_directory(self):
        """Test cleanup with non-existent directory"""
        deleted = FileManager.cleanup_old_files(
            '/nonexistent/directory', 
            'pattern', 
            days=7
        )
        
        self.assertEqual(deleted, 0)
    
    def test_cleanup_old_files_permission_error(self):
        """Test cleanup handles permission errors gracefully"""
        # Create a file
        test_file = os.path.join(self.temp_dir, 'test_file.txt')
        with open(test_file, 'w') as f:
            f.write('content')
        
        # Make it old
        old_time = (datetime.now() - timedelta(days=10)).timestamp()
        os.utime(test_file, (old_time, old_time))
        
        # Mock os.remove to raise permission error
        with patch('os.remove', side_effect=PermissionError("Access denied")):
            deleted = FileManager.cleanup_old_files(self.temp_dir, 'test_', days=7)
        
        # Should handle error and return 0 deleted
        self.assertEqual(deleted, 0)
    
    def test_cleanup_old_files_empty_directory(self):
        """Test cleanup on empty directory"""
        deleted = FileManager.cleanup_old_files(self.temp_dir, 'pattern', days=7)
        self.assertEqual(deleted, 0)
    
    def test_cleanup_old_files_exact_boundary(self):
        """Test files exactly at the age boundary"""
        now = datetime.now()
        
        # Create file exactly 7 days old
        boundary_file = os.path.join(self.temp_dir, 'test_boundary.txt')
        with open(boundary_file, 'w') as f:
            f.write('content')
        
        # Set to exactly 7 days ago
        boundary_time = (now - timedelta(days=7)).timestamp()
        os.utime(boundary_file, (boundary_time, boundary_time))
        
        # Should not be deleted (< not <=)
        deleted = FileManager.cleanup_old_files(self.temp_dir, 'test_', days=7)
        
        self.assertEqual(deleted, 0)
        self.assertTrue(os.path.exists(boundary_file))


class TestFileManagerIntegration(unittest.TestCase):
    """Integration tests for FileManager with real-world scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_file_workflow(self):
        """Test typical pipeline file workflow"""
        # 1. Ensure directories
        FileManager.ensure_directories(self.temp_dir)
        
        # 2. Save raw data
        raw_data = {
            'articles': [
                {'title': 'Article 1', 'content': 'Content 1'},
                {'title': 'Article 2', 'content': 'Content 2'}
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        raw_file = os.path.join(self.temp_dir, 'raw', 'news_20240101_120000.json')
        FileManager.save_json(raw_data, raw_file)
        
        # 3. Load and process
        loaded_data = FileManager.load_json(raw_file)
        
        # 4. Save processed data
        processed_data = {
            'articles': loaded_data['articles'],
            'processed_at': datetime.now().isoformat(),
            'article_count': len(loaded_data['articles'])
        }
        
        processed_file = os.path.join(self.temp_dir, 'cleaned', 'news_processed_20240101_120000.json')
        FileManager.save_json(processed_data, processed_file)
        
        # Verify both files exist
        self.assertTrue(os.path.exists(raw_file))
        self.assertTrue(os.path.exists(processed_file))
        
        # 5. Clean old files (simulate time passing)
        # Make raw file old
        old_time = (datetime.now() - timedelta(days=10)).timestamp()
        os.utime(raw_file, (old_time, old_time))
        
        deleted = FileManager.cleanup_old_files(
            os.path.join(self.temp_dir, 'raw'),
            'news_',
            days=7
        )
        
        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.exists(raw_file))
        self.assertTrue(os.path.exists(processed_file))
    
    def test_large_json_handling(self):
        """Test handling of large JSON files"""
        # Create large data structure
        large_data = {
            'items': [
                {'id': i, 'data': 'x' * 100}  # 100 chars per item
                for i in range(10000)  # 10,000 items
            ]
        }
        
        filepath = os.path.join(self.temp_dir, 'large.json')
        
        # Save large file
        FileManager.save_json(large_data, filepath)
        
        # Load it back
        loaded = FileManager.load_json(filepath)
        
        self.assertEqual(len(loaded['items']), 10000)
        self.assertEqual(loaded['items'][0]['data'], 'x' * 100)
    
    def test_concurrent_file_operations(self):
        """Test handling concurrent-like file operations"""
        base_path = os.path.join(self.temp_dir, 'concurrent')
        
        # Simulate multiple processes writing files
        for i in range(10):
            data = {'process': i, 'timestamp': datetime.now().isoformat()}
            filepath = os.path.join(base_path, f'process_{i}.json')
            FileManager.save_json(data, filepath)
        
        # All files should exist
        for i in range(10):
            filepath = os.path.join(base_path, f'process_{i}.json')
            self.assertTrue(os.path.exists(filepath))
            
            data = FileManager.load_json(filepath)
            self.assertEqual(data['process'], i)


if __name__ == '__main__':
    unittest.main()