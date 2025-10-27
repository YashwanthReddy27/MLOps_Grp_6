"""
Unit tests for database_utils.py module
Tests PostgreSQL operations including table creation, CRUD operations, and queries
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../dags/common'))

from database_utils import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db_manager = DatabaseManager()
    
    @patch('database_utils.PostgresHook')
    def test_initialization(self, mock_postgres_hook):
        """Test database manager initialization"""
        db = DatabaseManager('custom_conn_id')
        self.assertEqual(db.conn_id, 'custom_conn_id')
    
    @patch('database_utils.PostgresHook')
    def test_get_hook(self, mock_postgres_hook):
        """Test getting PostgreSQL hook"""
        hook = self.db_manager.get_hook()
        mock_postgres_hook.assert_called_with(postgres_conn_id='postgres_default')
    
    @patch('database_utils.PostgresHook')
    def test_create_table_if_not_exists(self, mock_postgres_hook):
        """Test table creation"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        schema_sql = "CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY)"
        
        self.db_manager.create_table_if_not_exists('test_table', schema_sql)
        
        mock_hook_instance.run.assert_called_once_with(schema_sql)
    
    @patch('database_utils.PostgresHook')
    def test_create_table_error_handling(self, mock_postgres_hook):
        """Test table creation error handling"""
        mock_hook_instance = Mock()
        mock_hook_instance.run.side_effect = Exception("Database error")
        mock_postgres_hook.return_value = mock_hook_instance
        
        with self.assertRaises(Exception) as context:
            self.db_manager.create_table_if_not_exists('test_table', 'CREATE TABLE...')
        
        self.assertIn("Database error", str(context.exception))
    
    @patch('database_utils.PostgresHook')
    def test_upsert_records_empty_list(self, mock_postgres_hook):
        """Test upsert with empty records list"""
        count = self.db_manager.upsert_records('test_table', [], 'id')
        self.assertEqual(count, 0)
    
    @patch('database_utils.PostgresHook')
    def test_upsert_records_single_record(self, mock_postgres_hook):
        """Test upsert with single record"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        records = [{'id': 1, 'name': 'Test', 'value': 100}]
        
        count = self.db_manager.upsert_records('test_table', records, 'id')
        
        self.assertEqual(count, 1)
        mock_hook_instance.run.assert_called_once()
    
    @patch('database_utils.PostgresHook')
    def test_upsert_records_multiple(self, mock_postgres_hook):
        """Test upsert with multiple records"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        records = [
            {'id': 1, 'name': 'Test1'},
            {'id': 2, 'name': 'Test2'},
            {'id': 3, 'name': 'Test3'}
        ]
        
        count = self.db_manager.upsert_records('test_table', records, 'id')
        
        self.assertEqual(count, 3)
        self.assertEqual(mock_hook_instance.run.call_count, 3)
    
    @patch('database_utils.PostgresHook')
    def test_upsert_records_with_custom_update_columns(self, mock_postgres_hook):
        """Test upsert with custom update columns"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        records = [{'id': 1, 'name': 'Test', 'value': 100, 'status': 'active'}]
        
        self.db_manager.upsert_records(
            'test_table', 
            records, 
            'id',
            update_columns=['name', 'value']  # Don't update status
        )
        
        # Verify SQL contains only specified update columns
        call_args = mock_hook_instance.run.call_args
        sql = call_args[0][0]
        self.assertIn('name = EXCLUDED.name', sql)
        self.assertIn('value = EXCLUDED.value', sql)
        self.assertNotIn('status = EXCLUDED.status', sql)
    
    @patch('database_utils.PostgresHook')
    def test_execute_query(self, mock_postgres_hook):
        """Test executing SELECT query"""
        mock_hook_instance = Mock()
        mock_connection = Mock()
        mock_cursor = Mock()
        
        mock_hook_instance.get_conn.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_postgres_hook.return_value = mock_hook_instance
        
        # Mock query results
        mock_cursor.description = [('id',), ('name',), ('value',)]
        mock_cursor.fetchall.return_value = [
            (1, 'Test1', 100),
            (2, 'Test2', 200)
        ]
        
        results = self.db_manager.execute_query("SELECT * FROM test_table")
        
        expected = [
            {'id': 1, 'name': 'Test1', 'value': 100},
            {'id': 2, 'name': 'Test2', 'value': 200}
        ]
        
        self.assertEqual(results, expected)
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table", None)
        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()
    
    @patch('database_utils.PostgresHook')
    def test_execute_query_with_params(self, mock_postgres_hook):
        """Test executing query with parameters"""
        mock_hook_instance = Mock()
        mock_connection = Mock()
        mock_cursor = Mock()
        
        mock_hook_instance.get_conn.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_postgres_hook.return_value = mock_hook_instance
        
        mock_cursor.description = [('id',), ('name',)]
        mock_cursor.fetchall.return_value = [(1, 'Test')]
        
        query = "SELECT * FROM test_table WHERE id = %s"
        params = (1,)
        
        results = self.db_manager.execute_query(query, params)
        
        mock_cursor.execute.assert_called_once_with(query, params)
    
    @patch('database_utils.PostgresHook')
    def test_execute_update(self, mock_postgres_hook):
        """Test executing UPDATE query"""
        mock_hook_instance = Mock()
        mock_connection = Mock()
        mock_cursor = Mock()
        
        mock_hook_instance.get_conn.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 5
        mock_postgres_hook.return_value = mock_hook_instance
        
        rows_affected = self.db_manager.execute_update(
            "UPDATE test_table SET status = 'inactive'"
        )
        
        self.assertEqual(rows_affected, 5)
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()
    
    @patch('database_utils.PostgresHook')
    def test_get_row_count(self, mock_postgres_hook):
        """Test getting row count"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        # Mock execute_query method
        self.db_manager.execute_query = Mock(return_value=[{'count': 42}])
        
        count = self.db_manager.get_row_count('test_table')
        
        self.assertEqual(count, 42)
        self.db_manager.execute_query.assert_called_once_with(
            "SELECT COUNT(*) as count FROM test_table"
        )
    
    @patch('database_utils.PostgresHook')
    def test_get_row_count_with_where_clause(self, mock_postgres_hook):
        """Test getting row count with WHERE clause"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        self.db_manager.execute_query = Mock(return_value=[{'count': 10}])
        
        count = self.db_manager.get_row_count('test_table', "status = 'active'")
        
        self.assertEqual(count, 10)
        self.db_manager.execute_query.assert_called_once_with(
            "SELECT COUNT(*) as count FROM test_table WHERE status = 'active'"
        )
    
    @patch('database_utils.PostgresHook')
    def test_bulk_insert(self, mock_postgres_hook):
        """Test bulk insert operation"""
        mock_hook_instance = Mock()
        mock_connection = Mock()
        mock_cursor = Mock()
        
        mock_hook_instance.get_conn.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_postgres_hook.return_value = mock_hook_instance
        
        records = [
            {'id': 1, 'name': 'Test1', 'value': 100},
            {'id': 2, 'name': 'Test2', 'value': 200},
            {'id': 3, 'name': 'Test3', 'value': 300}
        ]
        
        count = self.db_manager.bulk_insert('test_table', records)
        
        self.assertEqual(count, 3)
        mock_cursor.executemany.assert_called_once()
        mock_connection.commit.assert_called_once()
    
    @patch('database_utils.PostgresHook')
    def test_bulk_insert_empty_list(self, mock_postgres_hook):
        """Test bulk insert with empty list"""
        count = self.db_manager.bulk_insert('test_table', [])
        self.assertEqual(count, 0)
    
    @patch('database_utils.PostgresHook')
    def test_table_exists(self, mock_postgres_hook):
        """Test checking if table exists"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        self.db_manager.execute_query = Mock(return_value=[{'exists': True}])
        
        exists = self.db_manager.table_exists('test_table')
        
        self.assertTrue(exists)
    
    @patch('database_utils.PostgresHook')
    def test_get_column_names(self, mock_postgres_hook):
        """Test getting column names"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        self.db_manager.execute_query = Mock(return_value=[
            {'column_name': 'id'},
            {'column_name': 'name'},
            {'column_name': 'value'}
        ])
        
        columns = self.db_manager.get_column_names('test_table')
        
        self.assertEqual(columns, ['id', 'name', 'value'])
    
    @patch('database_utils.PostgresHook')
    def test_delete_old_records(self, mock_postgres_hook):
        """Test deleting old records"""
        mock_hook_instance = Mock()
        mock_postgres_hook.return_value = mock_hook_instance
        
        self.db_manager.execute_update = Mock(return_value=15)
        
        deleted = self.db_manager.delete_old_records(
            'test_table', 
            'created_at', 
            30
        )
        
        self.assertEqual(deleted, 15)
    
    @patch('database_utils.PostgresHook')
    def test_error_handling_in_execute_query(self, mock_postgres_hook):
        """Test error handling in execute_query"""
        mock_hook_instance = Mock()
        mock_hook_instance.get_conn.side_effect = Exception("Connection failed")
        mock_postgres_hook.return_value = mock_hook_instance
        
        with self.assertRaises(Exception) as context:
            self.db_manager.execute_query("SELECT * FROM test_table")
        
        self.assertIn("Connection failed", str(context.exception))


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database operations"""
    
    @patch('database_utils.PostgresHook')
    def test_full_crud_workflow(self, mock_postgres_hook):
        """Test complete CRUD workflow"""
        mock_hook_instance = Mock()
        mock_connection = Mock()
        mock_cursor = Mock()
        
        mock_hook_instance.get_conn.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_postgres_hook.return_value = mock_hook_instance
        
        db = DatabaseManager()
        
        # Create table
        create_sql = """
        CREATE TABLE IF NOT EXISTS articles (
            id VARCHAR(32) PRIMARY KEY,
            title VARCHAR(500),
            content TEXT,
            created_at TIMESTAMP
        )
        """
        db.create_table_if_not_exists('articles', create_sql)
        
        # Insert records
        records = [
            {'id': '1', 'title': 'Article 1', 'content': 'Content 1'},
            {'id': '2', 'title': 'Article 2', 'content': 'Content 2'}
        ]
        
        count = db.upsert_records('articles', records, 'id')
        self.assertEqual(count, 2)
        
        # Query records
        mock_cursor.description = [('id',), ('title',), ('content',)]
        mock_cursor.fetchall.return_value = [
            ('1', 'Article 1', 'Content 1'),
            ('2', 'Article 2', 'Content 2')
        ]
        
        results = db.execute_query("SELECT * FROM articles")
        self.assertEqual(len(results), 2)
        
        # Update record
        mock_cursor.rowcount = 1
        updated = db.execute_update(
            "UPDATE articles SET title = 'Updated' WHERE id = '1'"
        )
        self.assertEqual(updated, 1)


if __name__ == '__main__':
    unittest.main()