from datetime import datetime, timedelta
import json
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

# Import common utilities from modular structure
from common.deduplication import DeduplicationManager
from common.data_cleaning import TextCleaner
from common.data_enrichment import DataEnricher, CategoryManager
from common.database_utils import DatabaseManager
from common.file_management import FileManager

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# CONFIGURATION: Define your preferred keywords and categories
NEWS_KEYWORDS = {
    'artificial_intelligence': {
        'keywords': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'GPT', 'LLM', 'AI', 'generative AI', 'OpenAI', 'Claude', 'ChatGPT', 'transformer'],
        'weight': 1.0
    }
}


def fetch_tech_news(**context):
    """
    Fetches tech news based on predefined keywords from NEWS_KEYWORDS configuration.
    Fetches news for ALL categories and tags each article with matched categories.
    """
    try:
        # Initialize common utilities
        file_mgr = FileManager()
        dedup = DeduplicationManager('news')
        
        # Ensure directories exist
        file_mgr.ensure_directories()
        
        # Build search query from all keywords
        all_keywords = []
        for category_data in NEWS_KEYWORDS.values():
            all_keywords.extend(category_data['keywords'])
        
        # Create OR query for NewsAPI
        search_query = ' OR '.join([f'"{kw}"' for kw in all_keywords[:10]])  # Limit to avoid too long queries
        
        print(f"[EXTRACT] Fetching news with keywords: {search_query}")
        
        # NewsAPI endpoint and parameters
        url = 'https://newsapi.org/v2/everything'
        params = {
            'apiKey': 'f9756ab031a94ffc9a4241993518a5b5',
            'q': search_query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,  # Increased to get more articles
            'domains': 'techcrunch.com,theverge.com,wired.com,arstechnica.com,zdnet.com,bleepingcomputer.com,securityweek.com,thehackernews.com,venturebeat.com,thenextweb.com',
            'from': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')  # Last 3 days
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get('articles', [])
        
        print(f"[EXTRACT] Total articles fetched: {len(articles)}")
        
        # Filter duplicates using common utility
        new_articles, new_hashes = dedup.filter_duplicates(
            articles,
            lambda article: dedup.generate_hash(
                article.get('title', ''),
                article.get('url', '')
            )
        )
        
        print(f"[EXTRACT] New unique articles: {len(new_articles)}")
        print(f"[EXTRACT] Duplicates filtered: {len(articles) - len(new_articles)}")
        
        if new_articles:
            # Get current timestamp for the filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'/opt/airflow/data/raw/tech_news_{timestamp}.json'
            
            # Prepare output data
            output_data = {
                'status': data.get('status'),
                'totalResults': len(new_articles),
                'keywords_config': NEWS_KEYWORDS,
                'fetchedAt': datetime.now().isoformat(),
                'articles': new_articles
            }
            
            # Save using common utility
            file_mgr.save_json(output_data, filename)
            
            # Update hashes using common utility
            dedup.update_hashes(new_hashes)
            
            print(f"[EXTRACT] Successfully saved {len(new_articles)} new articles to {filename}")
            
            # Push result to XCom
            context['task_instance'].xcom_push(key='result', value={
                'filename': filename,
                'article_count': len(new_articles),
                'keywords_used': list(NEWS_KEYWORDS.keys())
            })
        else:
            print("[EXTRACT] No new articles found. All articles were duplicates.")
            context['task_instance'].xcom_push(key='result', value={
                'filename': None,
                'article_count': 0,
                'message': 'No new articles found'
            })
        
        return len(new_articles)
        
    except requests.exceptions.RequestException as e:
        print(f"[EXTRACT] Error fetching news data: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[EXTRACT] Response status: {e.response.status_code}")
            print(f"[EXTRACT] Response body: {e.response.text}")
        raise
    except Exception as e:
        print(f"[EXTRACT] Unexpected error: {str(e)}")
        raise


def extract_keywords_and_categorize(**context):
    """
    NEW TASK: Extract keywords from articles and categorize them based on NEWS_KEYWORDS config.
    Uses CategoryManager from common utilities.
    """
    try:
        # Initialize common utilities
        file_mgr = FileManager()
        cleaner = TextCleaner()
        
        # Get data from previous task
        result = context['task_instance'].xcom_pull(task_ids='extract_tech_news', key='result')
        
        if not result or not result.get('filename'):
            print("[CATEGORIZE] No data to process")
            return 0
        
        raw_file = result['filename']
        print(f"[CATEGORIZE] Processing file: {raw_file}")
        
        # Load raw data
        raw_data = file_mgr.load_json(raw_file)
        articles = raw_data.get('articles', [])
        
        # Process and categorize articles using CategoryManager
        categorized_articles = []
        
        for article in articles:
            # Clean text using common utility
            title_clean = cleaner.clean_text(article.get('title', ''))
            desc_clean = cleaner.clean_text(article.get('description', ''))
            
            if not title_clean:
                continue  # Skip invalid articles
            
            combined_text = title_clean + ' ' + desc_clean
            
            # Use CategoryManager to categorize
            categorization = CategoryManager.categorize_content(combined_text, NEWS_KEYWORDS)
            
            enriched = {
                'article_id': DeduplicationManager('news').generate_hash(title_clean, article.get('url', '')),
                'title': title_clean,
                'description': desc_clean,
                'url': article.get('url'),
                'source_name': article.get('source', {}).get('name'),
                'author': article.get('author') or 'Unknown',
                'published_at': article.get('publishedAt'),
                'image_url': article.get('urlToImage'),
                'primary_category': categorization['primary_category'],
                'all_categories': categorization['all_categories'],
                'category_scores': categorization['category_scores'],
                'overall_relevance': categorization['overall_relevance'],
                'processed_at': datetime.now().isoformat()
            }
            
            categorized_articles.append(enriched)
        
        # Sort by overall relevance
        categorized_articles.sort(key=lambda x: x['overall_relevance'], reverse=True)
        
        print(f"[CATEGORIZE] Categorized {len(categorized_articles)} articles")
        
        # Print category distribution
        category_counts = {}
        for article in categorized_articles:
            cat = article['primary_category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("[CATEGORIZE] Category distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {cat}: {count} articles")
        
        # Save categorized data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        categorized_file = f'/opt/airflow/data/cleaned/tech_news_categorized_{timestamp}.json'
        
        categorized_data = {
            'keywords_config': NEWS_KEYWORDS,
            'total_articles': len(categorized_articles),
            'category_distribution': category_counts,
            'processed_at': datetime.now().isoformat(),
            'articles': categorized_articles
        }
        
        file_mgr.save_json(categorized_data, categorized_file)
        
        # Push to XCom
        context['task_instance'].xcom_push(key='categorized_result', value={
            'filename': categorized_file,
            'article_count': len(categorized_articles),
            'category_distribution': category_counts
        })
        
        print(f"[CATEGORIZE] Saved categorized data to {categorized_file}")
        
        return len(categorized_articles)
        
    except Exception as e:
        print(f"[CATEGORIZE] Error: {str(e)}")
        raise


def load_to_postgresql(**context):
    """
    Load categorized articles to PostgreSQL with multi-category support.
    """
    try:
        from common import DatabaseManager, FileManager
        
        db = DatabaseManager()
        file_mgr = FileManager()
        
        categorized_result = context['ti'].xcom_pull(task_ids='extract_keywords_and_categorize', key='categorized_result')
        
        # Check if we have new articles
        if not categorized_result or not categorized_result.get('filename'):
            print("[DB] No new articles to load")
            
            # Return existing articles from database
            try:
                existing_articles = db.execute_query("""
                    SELECT article_id, title, description, url, source_name, 
                           primary_category, all_categories, overall_relevance, published_at
                    FROM tech_news_articles
                    ORDER BY overall_relevance DESC, published_at DESC
                    LIMIT 50
                """)
                
                if existing_articles:
                    print(f"[DB] Found {len(existing_articles)} existing articles in database")
                    
                    context['ti'].xcom_push(key='result', value={
                        'status': 'no_new_articles',
                        'message': 'All articles were duplicates. Returning previously fetched articles.',
                        'article_count': len(existing_articles),
                        'articles': existing_articles
                    })
                    return len(existing_articles)
                else:
                    print("[DB] No existing articles found in database")
                    context['ti'].xcom_push(key='result', value={
                        'status': 'no_articles',
                        'message': 'No articles found',
                        'article_count': 0
                    })
                    return 0
                    
            except Exception as e:
                print(f"[DB] Error querying existing articles: {e}")
                context['ti'].xcom_push(key='result', value={
                    'status': 'no_articles',
                    'message': 'No new articles and database query failed',
                    'article_count': 0
                })
                return 0
        
        # Load new categorized articles
        categorized_file = categorized_result['filename']
        print(f"[DB] Loading new data from: {categorized_file}")
        
        categorized_data = file_mgr.load_json(categorized_file)
        articles = categorized_data.get('articles', [])
        
        # Create table with multi-category support
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS tech_news_articles (
            article_id VARCHAR(32) PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            url VARCHAR(500),
            source_name VARCHAR(100),
            author VARCHAR(200),
            published_at TIMESTAMP,
            image_url VARCHAR(500),
            primary_category VARCHAR(100),
            all_categories TEXT[],
            category_scores JSONB,
            overall_relevance DECIMAL(5,3),
            processed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_primary_category ON tech_news_articles(primary_category);
        CREATE INDEX IF NOT EXISTS idx_all_categories ON tech_news_articles USING GIN(all_categories);
        CREATE INDEX IF NOT EXISTS idx_relevance ON tech_news_articles(overall_relevance);
        CREATE INDEX IF NOT EXISTS idx_published ON tech_news_articles(published_at);
        """
        
        db.create_table_if_not_exists('tech_news_articles', create_table_sql)
        
        # Prepare and insert records
        db_records = []
        for article in articles:
            db_records.append({
                'article_id': article['article_id'],
                'title': article['title'][:500],
                'description': article.get('description', '')[:1000] if article.get('description') else None,
                'url': article.get('url', '')[:500] if article.get('url') else None,
                'source_name': article.get('source_name', '')[:100] if article.get('source_name') else None,
                'author': article.get('author', 'Unknown')[:200],
                'published_at': article.get('published_at'),
                'image_url': article.get('image_url', '')[:500] if article.get('image_url') else None,
                'primary_category': article.get('primary_category', 'general_tech'),
                'all_categories': article.get('all_categories', []),
                'category_scores': json.dumps(article.get('category_scores', {})),
                'overall_relevance': article.get('overall_relevance', 0.0),
                'processed_at': article.get('processed_at'),
            })
        
        count = db.upsert_records('tech_news_articles', db_records, 'article_id')
        print(f"[DB] Successfully loaded {count} NEW articles to PostgreSQL")
        
        # Get category distribution from database
        category_dist = db.execute_query("""
            SELECT primary_category, COUNT(*) as count
            FROM tech_news_articles
            GROUP BY primary_category
            ORDER BY count DESC
        """)
        
        context['ti'].xcom_push(key='result', value={
            'status': 'success',
            'message': f'Successfully loaded {count} new articles',
            'article_count': count,
            'category_distribution': {row[0]: row[1] for row in category_dist}
        })
        
        return count
        
    except Exception as e:
        print(f"[DB] Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def cleanup_old_files(**context):
    """
    Clean up old files using common utility.
    """
    try:
        file_mgr = FileManager()
        
        # Get retention period from Airflow Variables (default 7 days)
        retention_days = int(Variable.get("news_retention_days", default_var="7"))
        
        print(f"[CLEANUP] Starting cleanup with {retention_days} day retention")
        
        # Clean up raw files
        deleted_raw = file_mgr.cleanup_old_files(
            '/opt/airflow/data/raw',
            'tech_news_',
            days=retention_days
        )
        
        # Clean up categorized files
        deleted_categorized = file_mgr.cleanup_old_files(
            '/opt/airflow/data/cleaned',
            'tech_news_categorized_',
            days=retention_days
        )
        
        total_deleted = deleted_raw + deleted_categorized
        
        print(f"[CLEANUP] Summary: Deleted {total_deleted} files total")
        print(f"[CLEANUP]   - Raw files: {deleted_raw}")
        print(f"[CLEANUP]   - Categorized files: {deleted_categorized}")
        
        # Push cleanup results to XCom
        context['task_instance'].xcom_push(key='cleanup_result', value={
            'deleted_files': total_deleted,
            'deleted_raw': deleted_raw,
            'deleted_categorized': deleted_categorized,
            'retention_days': retention_days
        })
        
        return True
        
    except Exception as e:
        print(f"[CLEANUP] Error during cleanup: {str(e)}")
        # Don't raise - cleanup failure shouldn't fail the entire DAG
        return False


# Define the DAG - can be triggered manually or scheduled
with DAG(
    'tech_news_keyword_based',
    default_args=default_args,
    description='Fetch and categorize tech news based on predefined keywords',
    schedule_interval='0 */6 * * *',  # Run every 6 hours (or set to None for manual only)
    start_date=days_ago(1),
    catchup=False,
    tags=['tech', 'news', 'keywords', 'automated', 'categorization'],
) as dag:
    
    # Task 1: Extract news based on keywords
    extract_news_task = PythonOperator(
        task_id='extract_tech_news',
        python_callable=fetch_tech_news,
        provide_context=True,
    )
    
    # Task 2: Extract keywords and categorize articles
    categorize_task = PythonOperator(
        task_id='extract_keywords_and_categorize',
        python_callable=extract_keywords_and_categorize,
        provide_context=True,
    )
    
    # Task 3: Load to PostgreSQL
    load_db_task = PythonOperator(
        task_id='load_to_postgresql',
        python_callable=load_to_postgresql,
        provide_context=True,
    )
    
    # Task 4: Cleanup old files
    cleanup_task = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=cleanup_old_files,
        provide_context=True,
    )

    # Set task dependencies
    extract_news_task >> categorize_task >> load_db_task >> cleanup_task