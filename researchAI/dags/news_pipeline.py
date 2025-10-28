"""
News API Pipeline with Explicit Schema Creation
Location: /app/dags/news_api_pipeline_with_validation.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Import pipeline functions
from news_api import NewsAPIPipeline

# Import schema creation and validation functions
from common.data_schema.schema_creator_module import create_news_schema
from common.data_validation import validate_data_quality
from common.bias_detector import BiasDetector

bias_detector = BiasDetector(
    data_path="/opt/airflow/data/cleaned/",
    output_dir="/opt/airflow/data/bias_reports/",
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
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=['news', 'schema', 'great-expectations', 'setup', 'manual'],
    doc_md="""
    ## News API Pipeline Schema Creation
    
    **Purpose**: Create or update the Great Expectations validation schema
    
    **When to Run**:
    - ✅ First-time setup (before running validation pipeline)
    - ✅ After significant changes to data structure
    - ✅ When enrichment process changes
    - ✅ When adding/removing fields
    - ✅ To update expectations based on new data patterns
    - ❌ NOT on a schedule - this is a manual maintenance task
    
    **Prerequisites**:
    - Run `tech_news_enriched_with_validation` at least once to generate training data
    - Or specify a custom training file in DAG run config
    
    **How to Run**:
    
    1. **Default (uses most recent data)**:
       - Click "Trigger DAG" button in Airflow UI
       - Uses most recent categorized news file as training data
    
    2. **Custom training file**:
       - Click "Trigger DAG w/ config"
       - Provide configuration:
       ```json
       {
         "training_file": "/opt/airflow/data/cleaned/tech_news_categorized_20250101.json"
       }
       ```
    
    3. **Overwrite existing schema**:
       - Use config:
       ```json
       {
         "overwrite_schema": true
       }
       ```
    
    **What It Does**:
    1. Loads training data from specified or most recent file
    2. Analyzes enriched content structure and statistics
    3. Generates expectation suite (schema) with expectations for:
       - Basic fields (title, description, url)
       - Enriched content fields
       - Category and keyword fields
       - Metadata fields
    4. Saves schema to:
       - `/opt/airflow/data/ge_artifacts/news_api/expectations/news_api_suite.json`
       - `/opt/airflow/data/schema/news_api_expectations.json` (for version control)
    5. Sends email notification with details
    
    **After Running**:
    1. ✓ Review generated schema file
    2. ✓ Commit schema to git: `data/schema/news_api_expectations.json`
    3. ✓ Run validation pipeline - it will now succeed
    4. ✓ Monitor future validation results
    
    **Schema Safety**:
    - Existing schemas are NOT overwritten by default
    - Must explicitly pass `overwrite_schema: true` to recreate
    - Protects against accidental schema deletion
    """,
) as schema_dag:
    
    create_schema_task = PythonOperator(
        task_id='create_news_schema',
        python_callable=create_news_schema,
        provide_context=True,
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
    schedule_interval='0 */6 * * *',  # Every 6 hours
    start_date=days_ago(1),
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
        provide_context=True,
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
        provide_context=True,
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
        trigger_rule='none_failed_or_skipped',
        provide_context=True,
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
        provide_context=True,
        doc_md="""Detects bias in the processed data using Fairlearn.""",
    )
    
    # Task 5: Load to PostgreSQL
    load_db = PythonOperator(
        task_id='load_to_postgresql',
        python_callable=news_api.load_to_postgresql,
        provide_context=True,
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
        trigger_rule='none_failed_or_skipped',
        provide_context=True,
        doc_md="""
        Cleans up old JSON files based on retention policy.
        Default: 7 days retention.
        
        Runs even if earlier tasks fail.
        """,
    )

    # Define task dependencies
    extract_news >> categorize >> validate >> detect_bias_task >> load_db >> cleanup
