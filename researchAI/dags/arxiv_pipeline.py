"""
arXiv AI Research Pipeline with Explicit Schema Creation
Location: /app/dags/arxiv_pipeline_with_validation.py
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
    data_path="/opt/airflow/data/cleaned/",
    output_dir="/opt/airflow/data/bias_reports",
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
    tags=['arxiv', 'schema', 'great-expectations', 'setup', 'manual'],
    doc_md="""
    ## arXiv Pipeline Schema Creation
    
    **Purpose**: Create or update the Great Expectations validation schema
    
    **When to Run**:
    - ✅ First-time setup (before running validation pipeline)
    - ✅ After significant changes to data structure
    - ✅ When adding/removing columns
    - ✅ To update expectations based on new data patterns
    - ❌ NOT on a schedule - this is a manual maintenance task
    
    **Prerequisites**:
    - Run `arxiv_ai_research_with_validation` at least once to generate training data
    - Or specify a custom training file in DAG run config
    
    **How to Run**:
    
    1. **Default (uses most recent data)**:
       - Click "Trigger DAG" button in Airflow UI
       - Uses most recent processed arXiv file as training data
    
    2. **Custom training file**:
       - Click "Trigger DAG w/ config"
       - Provide configuration:
       ```json
       {
         "training_file": "/opt/airflow/data/cleaned/arxiv_papers_processed_20250101.json"
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
    2. Analyzes data structure and statistics
    3. Generates expectation suite (schema) with ~20-50 expectations
    4. Saves schema to:
       - `/opt/airflow/data/ge_artifacts/arxiv/expectations/arxiv_suite.json`
       - `/opt/airflow/data/schema/arxiv_expectations.json` (for version control)
    5. Sends email notification with details
    
    **After Running**:
    1. ✓ Review generated schema file
    2. ✓ Commit schema to git: `data/schema/arxiv_expectations.json`
    3. ✓ Run validation pipeline - it will now succeed
    4. ✓ Monitor future validation results
    
    **Schema Safety**:
    - Existing schemas are NOT overwritten by default
    - Must explicitly pass `overwrite_schema: true` to recreate
    - Protects against accidental schema deletion
    """,
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
    tags=['arxiv', 'research', 'ai', 'papers', 'automated', 'ge', 'validation'],
    doc_md="""
    ## arXiv AI Research Pipeline with Data Validation
    
    **Prerequisites**:
    - Schema must exist (run `arxiv_create_schema` DAG first)
    
    **Pipeline Flow**:
    1. Fetch papers from arXiv API
    2. Process and categorize using LLM
    3. **Validate data quality** (requires schema)
    4. Load to PostgreSQL (only if validation passes)
    5. Cleanup old files
    
    **Validation Behavior**:
    - ❌ **FAILS if schema doesn't exist** (by design)
    - ✅ Continues if anomalies detected (with alerts)
    - 📧 Sends alerts for missing schema or anomalies
    
    **Schedule**: Daily at midnight
    """,
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

    
    # Task 4: Load to PostgreSQL
    load_db_task = PythonOperator(
        task_id='load_to_postgresql',
        python_callable=arxiv.load_to_postgresql,
        
        doc_md=""" Loads validated data to PostgreSQL database. Only runs if validation succeeds.""",
    )
    
    # Task 5: Cleanup old files
    cleanup_task = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=arxiv.cleanup_old_files,
        
        doc_md="""
        Removes files older than retention period.
        """,
    )

    # Set task dependencies - validation happens BEFORE database load
    fetch_papers_task >> process_task >> validate_task >> detect_bias_task >> load_db_task >> cleanup_task