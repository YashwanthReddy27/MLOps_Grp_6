"""
arXiv AI Research Pipeline with Great Expectations Data Validation
Location: /app/dags/arxiv_pipeline_with_validation.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Import ALL functions from your original arxiv_pipeline
from researchAI.dags.arxiv import ArxivPipeline

# Import new Great Expectations validation functions
from common.ge_validation_task import (
    validate_data_quality,
    generate_data_statistics_report
)

arxiv = ArxivPipeline()

# Default arguments - UPDATE EMAIL ADDRESS
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': True,
    'email': ['projectmlops@gmail.com'],  
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ============================================================================
# MAIN DAG - Daily Pipeline with Validation
# ============================================================================

with DAG(
    'arxiv_ai_research_with_validation',
    default_args=default_args,
    description='Fetch and categorize AI research papers from arXiv with Great Expectations validation',
    schedule_interval='0 0 * * *',  # Run daily at midnight
    start_date=days_ago(1),
    catchup=False,
    tags=['arxiv', 'research', 'ai', 'papers', 'automated', 'ge', 'validation'],
) as dag:
    
    # Task 1: Fetch papers from arXiv (from original pipeline)
    fetch_papers_task = PythonOperator(
        task_id='fetch_arxiv_papers',
        python_callable=arxiv.fetch_arxiv_papers,
        provide_context=True,
    )
    
    # Task 2: Process and categorize papers (from original pipeline)
    process_task = PythonOperator(
        task_id='process_and_categorize_papers',
        python_callable=arxiv.process_and_categorize_papers,
        provide_context=True,
    )
    
    # Task 3: ✨ NEW - Validate data quality with TFDV
    validate_task = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality, # This now uses Great Expectations
        provide_context=True,
    )
    
    # Task 4: Load to PostgreSQL (from original pipeline)
    # Only runs if validation passes
    load_db_task = PythonOperator(
        task_id='load_to_postgresql',
        python_callable=arxiv.load_to_postgresql,
        provide_context=True,
    )
    
    # Task 5: Cleanup old files (from original pipeline)
    cleanup_task = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=arxiv.cleanup_old_files,
        provide_context=True,
    )

    # Set task dependencies - validation happens BEFORE database load
    fetch_papers_task >> process_task >> validate_task >> load_db_task >> cleanup_task


# ============================================================================
# WEEKLY REPORT DAG - Comparative Statistics
# ============================================================================

with DAG(
    'arxiv_weekly_statistics_report',
    default_args=default_args,
    description='Generate weekly data quality statistics report for arXiv pipeline',
    schedule_interval='0 9 * * 0',  # Run weekly on Sunday at 9 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['arxiv', 'ge', 'report', 'weekly'],
) as report_dag:
    
    # Single task: Generate comprehensive report
    generate_report_task = PythonOperator(
        task_id='generate_statistics_report',
        python_callable=generate_data_statistics_report,
        provide_context=True,
    )