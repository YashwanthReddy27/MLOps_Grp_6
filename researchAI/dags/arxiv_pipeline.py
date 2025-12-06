"""
arXiv AI Research Pipeline
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import pipeline functions
from arxiv import ArxivPipeline

# Import schema creation and validation functions
from common.data_schema.schema_creator_module import create_arxiv_schema
from common.data_validation import validate_data_quality
from common.bias_detector import BiasDetector


bias_detector = BiasDetector(
    data_path="/home/airflow/gcs/data/cleaned/",
    output_dir="/home/airflow/gcs/data/bias_reports",
    data_type="arxiv"
)

arxiv = ArxivPipeline()

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
    'arxiv_create_schema',
    default_args=default_args,
    description='Create/Update Great Expectations schema for arXiv pipeline (Run manually)',
    schedule=None,  # Manual trigger only - no automatic schedule
    catchup=False,
    tags=['arxiv data file schema'],
) as schema_dag:
    
    create_schema_task = PythonOperator(
        task_id='create_arxiv_schema',
        python_callable=create_arxiv_schema,
        doc_md="""
        Creates Great Expectations schema for arXiv pipeline.
        
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
# MAIN DAG - Daily Pipeline with Validation
# ============================================================================

with DAG(
    'arxiv_ai_research_with_validation',
    default_args=default_args,
    description='Fetch and categorize AI research papers from arXiv with Great Expectations validation',
    schedule='0 0 * * *',  # Run daily at midnight
    catchup=False,
    max_active_runs=1,
    tags=['arxiv', 'research', 'ai', 'papers'],
) as dag:
    
    # Task 1: Fetch papers from arXiv
    fetch_papers_task = PythonOperator(
        task_id='fetch_arxiv_papers',
        python_callable=arxiv.fetch_arxiv_papers,
        
        doc_md="""
        Fetches recent AI research papers from arXiv API.
        
        **Outputs**:
        - XCom key: 'result' with filename and paper count
        """,
    )
    
    # Task 2: Process and categorize papers
    process_task = PythonOperator(
        task_id='process_and_categorize_papers',
        python_callable=arxiv.process_and_categorize_papers,
        
        doc_md="""
        Processes papers and categorizes using LLM.
        
        **Outputs**:
        - XCom key: 'processed_result' with processed filename
        """,
    )
    
    # Task 3: Validate data quality with Great Expectations
    validate_task = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        
        doc_md="""
        Validates data quality using Great Expectations.
        
        **IMPORTANT**: 
        - Requires schema to exist (created by arxiv_create_schema DAG)
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

    # Set task dependencies - validation happens BEFORE database load
    fetch_papers_task >> process_task >> validate_task >> detect_bias_task