"""
Database operation utilities
Handles PostgreSQL operations including table creation, CRUD operations, and queries
"""

from typing import Dict, List, Any, Optional
from airflow.providers.postgres.hooks.postgres import PostgresHook


class DatabaseManager:
    """PostgreSQL database operations wrapper"""
    
    def __init__(self, conn_id: str = 'postgres_default'):
        """
        Initialize database manager
        
        Args:
            conn_id: Airflow connection ID for PostgreSQL (default: 'postgres_default')
        """
        self.conn_id = conn_id
    
    def get_hook(self) -> PostgresHook:
        """
        Get PostgreSQL hook
        
        Returns:
            PostgresHook instance
        """
        return PostgresHook(postgres_conn_id=self.conn_id)
    
    def create_table_if_not_exists(self, table_name: str, schema_sql: str) -> None:
        """
        Create table with given schema if it doesn't exist
        
        Args:
            table_name: Name of the table
            schema_sql: SQL CREATE TABLE statement
            
        Example:
            db.create_table_if_not_exists(
                'articles',
                '''CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(500)
                );'''
            )
        """
        try:
            hook = self.get_hook()
            hook.run(schema_sql)
            print(f"[DB] Table {table_name} created/verified")
        except Exception as e:
            print(f"[DB] Error creating table {table_name}: {e}")
            raise
    
    def upsert_records(self, table_name: str, records: List[Dict], 
                      conflict_column: str, update_columns: List[str] = None) -> int:
        """
        Insert or update records using ON CONFLICT
        
        Args:
            table_name: Target table name
            records: List of dictionaries to insert/update
            conflict_column: Column to check for conflicts (usually primary key)
            update_columns: Columns to update on conflict (default: all except conflict_column)
            
        Returns:
            Number of records processed
            
        Example:
            count = db.upsert_records(
                'articles',
                [{'article_id': '123', 'title': 'New Title'}],
                conflict_column='article_id'
            )
        """
        if not records:
            print("[DB] No records to upsert")
            return 0
        
        try:
            hook = self.get_hook()
            count = 0
            
            for record in records:
                columns = list(record.keys())
                placeholders = [f'%({col})s' for col in columns]
                
                if update_columns is None:
                    update_columns = [col for col in columns if col != conflict_column]
                
                update_clause = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_columns])
                
                sql = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                    ON CONFLICT ({conflict_column}) DO UPDATE SET
                        {update_clause},
                        updated_at = CURRENT_TIMESTAMP;
                """
                
                hook.run(sql, parameters=record)
                count += 1
            
            print(f"[DB] Upserted {count} records to {table_name}")
            return count
            
        except Exception as e:
            print(f"[DB] Error upserting records: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute SELECT query and return results as list of dicts
        
        Args:
            query: SQL SELECT query
            params: Query parameters (optional)
            
        Returns:
            List of dictionaries representing rows
            
        Example:
            results = db.execute_query(
                "SELECT * FROM articles WHERE category = %s",
                ('AI',)
            )
        """
        try:
            hook = self.get_hook()
            connection = hook.get_conn()
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            connection.close()
            
            return results
            
        except Exception as e:
            print(f"[DB] Error executing query: {e}")
            raise
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        Execute UPDATE/DELETE query
        
        Args:
            query: SQL UPDATE or DELETE query
            params: Query parameters (optional)
            
        Returns:
            Number of rows affected
        """
        try:
            hook = self.get_hook()
            connection = hook.get_conn()
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            rows_affected = cursor.rowcount
            connection.commit()
            
            cursor.close()
            connection.close()
            
            return rows_affected
            
        except Exception as e:
            print(f"[DB] Error executing update: {e}")
            raise
    
    def get_row_count(self, table_name: str, where_clause: str = None) -> int:
        """
        Get row count for table
        
        Args:
            table_name: Name of the table
            where_clause: Optional WHERE clause (without WHERE keyword)
            
        Returns:
            Row count
            
        Example:
            count = db.get_row_count('articles', "category = 'AI'")
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            result = self.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            print(f"[DB] Error getting row count: {e}")
            return 0
    
    def bulk_insert(self, table_name: str, records: List[Dict]) -> int:
        """
        Bulk insert records (faster than upsert for new data)
        
        Args:
            table_name: Target table name
            records: List of dictionaries to insert
            
        Returns:
            Number of records inserted
            
        Example:
            count = db.bulk_insert('articles', article_list)
        """
        if not records:
            print("[DB] No records to insert")
            return 0
        
        try:
            hook = self.get_hook()
            columns = list(records[0].keys())
            
            # Prepare values for bulk insert
            values_list = []
            for record in records:
                values = [record.get(col) for col in columns]
                values_list.append(values)
            
            # Build SQL
            placeholders = ', '.join(['%s'] * len(columns))
            sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Execute bulk insert
            connection = hook.get_conn()
            cursor = connection.cursor()
            cursor.executemany(sql, values_list)
            connection.commit()
            
            count = len(records)
            print(f"[DB] Bulk inserted {count} records to {table_name}")
            
            cursor.close()
            connection.close()
            
            return count
            
        except Exception as e:
            print(f"[DB] Error bulk inserting records: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists
        """
        try:
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """
            result = self.execute_query(query, (table_name,))
            return result[0]['exists'] if result else False
        except Exception as e:
            print(f"[DB] Error checking table existence: {e}")
            return False
    
    def get_column_names(self, table_name: str) -> List[str]:
        """
        Get column names for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column names
        """
        try:
            query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """
            results = self.execute_query(query, (table_name,))
            return [row['column_name'] for row in results]
        except Exception as e:
            print(f"[DB] Error getting column names: {e}")
            return []
    
    def delete_old_records(self, table_name: str, date_column: str, days_old: int) -> int:
        """
        Delete records older than specified days
        
        Args:
            table_name: Target table name
            date_column: Column containing date/timestamp
            days_old: Number of days threshold
            
        Returns:
            Number of records deleted
            
        Example:
            deleted = db.delete_old_records('articles', 'created_at', 30)
        """
        try:
            query = f"""
                DELETE FROM {table_name}
                WHERE {date_column} < NOW() - INTERVAL '{days_old} days'
            """
            rows_deleted = self.execute_update(query)
            print(f"[DB] Deleted {rows_deleted} old records from {table_name}")
            return rows_deleted
        except Exception as e:
            print(f"[DB] Error deleting old records: {e}")
            return 0


__all__ = ['DatabaseManager']