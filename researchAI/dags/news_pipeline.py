"""
News API Pipeline with Explicit Schema Creation
Location: /app/dags/news_api_pipeline_with_validation.py
"""

from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import pipeline functions
from news_api import NewsAPIPipeline

# Import schema creation and validation functions
from common.data_schema.schema_creator_module import create_news_schema
from common.data_validation import validate_data_quality
from common.bias_detector import BiasDetector

bias_detector = BiasDetector(
    data_path="/home/airflow/gcs/data/cleaned/",
    output_dir="/home/airflow/gcs/data/bias_reports/",
    data_type="tech_news"
)

news_api = NewsAPIPipeline()

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': True,
    'email': ['anirudhshrikanth65@gmail.com'],
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}


# ============================================================================
# SCHEMA CREATION DAG - Run Once or When Schema Needs Update
# ============================================================================

with DAG(
    'news_create_schema',
    default_args=default_args,
    description='Create/Update Great Expectations schema for News API pipeline (Run manually)',
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=['news', 'schema', 'great-expectations', 'setup', 'manual'],
    doc_md=""" News API Pipeline Schema Creation""",
) as schema_dag:
    
    create_schema_task = PythonOperator(
        task_id='create_news_schema',
        python_callable=create_news_schema,
        doc_md="""
        Creates Great Expectations schema for News API pipeline.
        
        **Inputs**:
        - Training data file (auto-detected or specified in config)
        
        **Outputs**:
        - Schema file in GE artifacts directory
        - Schema file in version control directory
        - Email notification with results
        
        **XCom**:
        - Key: 'schema_creation_result'
        - Value: Dict with status, num_expectations, file paths
        """,
    )


# ============================================================================
# MAIN DAG - Tech News with Full Content Enrichment and Validation
# ============================================================================

with DAG(
    'tech_news_enriched_with_validation',
    default_args=default_args,
    description='Fetch tech news, enrich with full article content, categorize, validate, and store',
    schedule='0 */6 * * *',  # Every 6 hours
    catchup=False,
    max_active_runs=1,  # Prevent overlapping runs
    tags=['tech', 'news', 'enrichment', 'web-scraping', 'validation'],
    doc_md="""
    ## Tech News Pipeline with Content Enrichment and Validation
    
    **Prerequisites**:
    - Schema must exist (run `news_create_schema` DAG first)
    
    **Pipeline Flow**:
    1. Fetch news from News API
    2. Enrich with full content via web scraping
    3. Extract keywords and categorize
    4. **Validate data quality** (requires schema)
    5. Load to PostgreSQL (only if validation passes)
    6. Cleanup old files
    
    **Validation Behavior**:
    - ❌ **FAILS if schema doesn't exist** (by design)
    - ✅ Continues if anomalies detected (with alerts)
    - 📧 Sends alerts for missing schema or anomalies
    
    **Schedule**: Every 6 hours
    """,
) as dag:
    
    # Task 1: Fetch news from News API
    extract_news = PythonOperator(
        task_id='extract_tech_news',
        python_callable=news_api.fetch_tech_news,
        
        doc_md="""
        Fetches tech news articles from News API.
        Filters duplicates and saves new articles.
        
        **Outputs**:
        - XCom key: 'result' with filename and article count
        """,
    )

    # Task 3: Categorize articles using enriched content
    categorize = PythonOperator(
        task_id='extract_keywords_and_categorize',
        python_callable=news_api.extract_keywords_and_categorize,
        
        doc_md="""
        Categorizes articles using full content.
        Assigns categories, relevance scores, and extracts keywords.
        
        **Outputs**:
        - XCom key: 'categorized_result' with categorized data filename
        """,
    )
    
    # Task 4: Validate data quality
    validate = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        doc_md="""
        Validates data quality using Great Expectations.
        
        **IMPORTANT**: 
        - Requires schema to exist (created by news_create_schema DAG)
        - FAILS if schema not found
        - Continues if anomalies detected (with alerts)
        
        **Outputs**:
        - XCom key: 'validation_result' with status and anomaly details
        """,
    )

    detect_bias_task = PythonOperator(
        task_id='detect_bias_in_data',
        python_callable=bias_detector.detect_bias,
        doc_md="""Detects bias in the processed data using Fairlearn.""",
    )
    
    # Task 5: Load to PostgreSQL
    load_db = PythonOperator(
        task_id='load_to_postgresql',
        python_callable=news_api.load_to_postgresql,
        doc_md="""
        Loads validated data to PostgreSQL database.
        Stores full content, categories, and enrichment metadata.
        
        Only runs if validation succeeds.
        """,
    )
    
    # Task 6: Clean up old files
    cleanup = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=news_api.cleanup_old_files,
        doc_md="""
        Cleans up old JSON files based on retention policy.
        Default: 7 days retention.
        
        Runs even if earlier tasks fail.
        """,
    )

    # Define task dependencies
    extract_news >> categorize >> validate >> detect_bias_task >> load_db >> cleanup
