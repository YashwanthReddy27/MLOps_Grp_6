"""
News API Pipeline with Great Expectations Data Validation
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Import ALL functions from your original news_api_pipeline
from news_api_pipeline import (
    fetch_tech_news,
    extract_keywords_and_categorize,
    load_to_postgresql,
    cleanup_old_files
)

from common.news_enrichment_pipe import (
    enrich_news_articles,
    extract_keywords_and_categorize_enriched
)

# Import new Great Expectations validation functions
from common.ge_validation_task import (
    validate_data_quality,
    generate_data_statistics_report
)

# Default arguments - UPDATE EMAIL ADDRESS
# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,  # Set to True and configure SMTP if needed
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ============================================================================
# MAIN DAG - Tech News with Full Content Enrichment
# ============================================================================

with DAG(
    'tech_news_enriched_with_validation',
    default_args=default_args,
    description='Fetch tech news, enrich with full article content via web scraping, categorize, validate, and store',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,  # Prevent overlapping runs
    tags=['tech', 'news', 'enrichment', 'web-scraping', 'validation'],
) as dag:
    
    # Task 1: Fetch news from News API
    # Returns: article count (int), saves data to file, pushes filename to XCom
    extract_news = PythonOperator(
        task_id='extract_tech_news',
        python_callable=fetch_tech_news,
        doc_md="""
        Fetches tech news articles from News API based on predefined keywords.
        Filters duplicates and saves new articles to JSON file.
        Output: XCom key 'result' with filename and article count
        """,
    )
    
    # Task 2: Enrich articles with full content
    # Scrapes article URLs to get complete text (500-2000+ words vs 200 chars)
    enrich_content = PythonOperator(
        task_id='enrich_article_content',
        python_callable=enrich_news_articles,
        execution_timeout=timedelta(minutes=30),  # Web scraping can take time
        doc_md="""
        Enriches articles by fetching full content from URLs via web scraping.
        Extracts complete article text, metadata, and enhanced descriptions.
        Output: XCom key 'enriched_result' with enriched data filename
        """,
    )

    # Task 3: Categorize articles using enriched content
    # Uses full article text for better categorization accuracy
    categorize = PythonOperator(
        task_id='extract_keywords_and_categorize',
        python_callable=extract_keywords_and_categorize_enriched,
        doc_md="""
        Categorizes articles using full content (if available) or descriptions.
        Assigns primary category, relevance scores, and extracts keywords.
        Output: XCom key 'categorized_result' with categorized data filename
        """,
    )
    
    # Task 4: Validate data quality
    # Ensures data meets quality standards before database insertion
    validate = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        trigger_rule='none_failed_or_skipped',  # Run even if enrichment partially fails
        doc_md="""
        Validates data quality using TFDV/Great Expectations.
        Checks schema compliance, data types, and anomalies.
        """,
    )
    
    # Task 5: Load to PostgreSQL
    # Stores enriched and categorized articles in database
    load_db = PythonOperator(
        task_id='load_to_postgresql',
        python_callable=load_to_postgresql,
        doc_md="""
        Loads categorized articles to PostgreSQL database.
        Stores full content, categories, and enrichment metadata.
        """,
    )
    
    # Task 6: Clean up old files
    # Removes files older than retention period
    cleanup = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=cleanup_old_files,
        trigger_rule='none_failed_or_skipped',  # Always run cleanup
        doc_md="""
        Cleans up old JSON files based on retention policy.
        Default: 7 days retention (configurable via Airflow Variables).
        """,
    )

    # Define task dependencies
    extract_news >> enrich_content >> categorize >> validate >> load_db >> cleanup


# ============================================================================
# WEEKLY STATISTICS REPORT DAG
# ============================================================================

with DAG(
    'news_enrichment_weekly_report',
    default_args=default_args,
    description='Weekly statistics report for news enrichment pipeline',
    schedule_interval='0 9 * * 0',  # Sundays at 9 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['news', 'report', 'statistics', 'weekly'],
) as report_dag:
    
    # Generate comprehensive statistics report
    generate_report = PythonOperator(
        task_id='generate_statistics_report',
        python_callable=generate_data_statistics_report,
        doc_md="""
        Generates weekly statistics report including:
        - Enrichment success rates
        - Average word counts
        - Category distributions
        - Data quality metrics
        """,
    )