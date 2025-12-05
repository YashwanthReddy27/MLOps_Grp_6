"""
News API Pipeline with Explicit Schema Creation
"""

from datetime import timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.smtp.notifications.smtp import send_smtp_notification

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
    'email': ['anirudhshrikanth65@gmail.com'],
    'retries': 0,
    'on_failure_callback': [
        send_smtp_notification(
            from_email="projectmlops@gmail.com",
            to="anirudhshrikanth65@gmail.com",
            subject="[Error] The dag {{ dag.dag_id }} failed",
            html_content="debug logs",
        )
    ],
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
    tags=['tech news schema'],
    doc_md="""News API Pipeline Schema Creation""",
) as schema_dag:
    create_schema_task = PythonOperator(
        task_id='create_news_schema',
        python_callable=create_news_schema
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
    tags=['tech news pipeline']
) as dag:
    # Task 1: Fetch news from News API
    extract_news = PythonOperator(
        task_id='extract_tech_news',
        python_callable=news_api.fetch_tech_news
    )

    # Task 2: Categorize articles using enriched content
    categorize = PythonOperator(
        task_id='extract_keywords_and_categorize',
        python_callable=news_api.extract_keywords_and_categorize
    )
    
    # Task 3: Validate data quality
    validate = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality
    )

    # Task 4: Detects bias in the processed data using Fairlearn.
    detect_bias_task = PythonOperator(
        task_id='detect_bias_in_data',
        python_callable=bias_detector.detect_bias
    )

    # Define task dependencies
    extract_news >> categorize >> validate >> detect_bias_task
