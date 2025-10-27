"""
Schema Creation Module for Great Expectations
Import this in your pipeline DAGs to create explicit schemas
Location: /app/common/create_schema.py
"""

from common.ge_validator import PipelineValidator
from common.send_email import AlertEmail
from pathlib import Path
from datetime import datetime
import json

alert_email = AlertEmail()


def create_pipeline_schema(pipeline_name: str, training_data_path: str, **context):
    """
    Create or update Great Expectations schema for a pipeline
    
    Args:
        pipeline_name: Name of the pipeline ('arxiv' or 'news_api')
        training_data_path: Path to the training data file
        context: Airflow context
        
    Returns:
        bool: True if successful
        
    Raises:
        FileNotFoundError: If training data file doesn't exist
        Exception: If schema creation fails
    """
    try:
        print(f"[SCHEMA] Creating schema for {pipeline_name} pipeline")
        print(f"[SCHEMA] Training data: {training_data_path}")
        
        # Verify training data exists
        if not Path(training_data_path).exists():
            raise FileNotFoundError(f"Training data file not found: {training_data_path}")
        
        # Initialize validator
        validator = PipelineValidator(pipeline_name=pipeline_name)
        
        # Check if schema already exists
        existing_suite = validator._get_expectation_suite()
        
        if existing_suite:
            print(f"[SCHEMA] ⚠️  WARNING: Schema already exists for {pipeline_name}")
            print(f"[SCHEMA] Existing schema location: {validator.expectations_dir / f'{validator.expectation_suite_name}.json'}")
            
            # Get overwrite flag from DAG run config or default to False
            overwrite = True
            if context.get('dag_run') and context['dag_run'].conf:
                overwrite = context['dag_run'].conf.get('overwrite_schema', False)
            
            if not overwrite:
                print(f"[SCHEMA] Skipping schema creation - schema already exists")
                print(f"[SCHEMA] To recreate, pass 'overwrite_schema': true in DAG config")
                
                context['ti'].xcom_push(key='schema_creation_result', value={
                    'status': 'skipped',
                    'message': 'Schema already exists',
                    'pipeline': pipeline_name,
                    'suite_name': validator.expectation_suite_name
                })
                return True
            
            print(f"[SCHEMA] Overwriting existing schema (overwrite_schema=true)")
        
        # Create the schema
        print(f"[SCHEMA] Creating baseline schema from training data...")
        validator.create_schema(training_data_path)
        
        # Verify schema was created
        suite = validator._get_expectation_suite()
        if not suite:
            raise Exception("Schema creation completed but suite file not found")
        
        num_expectations = len(suite.get('expectations', []))
        
        # Get file size
        schema_file = validator.expectations_dir / f"{validator.expectation_suite_name}.json"
        file_size_kb = schema_file.stat().st_size / 1024
        
        result = {
            'status': 'success',
            'message': f'Schema created successfully with {num_expectations} expectations',
            'pipeline': pipeline_name,
            'training_file': training_data_path,
            'num_expectations': num_expectations,
            'suite_name': validator.expectation_suite_name,
            'schema_file': str(schema_file),
            'file_size_kb': round(file_size_kb, 2),
            'created_at': datetime.now().isoformat()
        }
        
        context['ti'].xcom_push(key='schema_creation_result', value=result)
        
        # Send notification email
        send_schema_creation_notification(result, context)
        
        print(f"[SCHEMA] ✅ Schema created successfully")
        print(f"[SCHEMA]    - Expectations: {num_expectations}")
        print(f"[SCHEMA]    - File size: {file_size_kb:.2f} KB")
        print(f"[SCHEMA]    - Location: {schema_file}")
        
        return True
        
    except Exception as e:
        print(f"[SCHEMA] ❌ Error creating schema: {e}")
        import traceback
        traceback.print_exc()
        
        error_result = {
            'status': 'error',
            'message': str(e),
            'pipeline': pipeline_name,
            'training_file': training_data_path
        }
        context['ti'].xcom_push(key='schema_creation_result', value=error_result)
        
        send_schema_error_notification(pipeline_name, str(e), training_data_path, context)
        
        raise


def create_arxiv_schema(**context):
    """
    Create schema for arXiv pipeline
    Uses a specific training data file
    """
    pipeline_name = 'arxiv'
    
    # Get training file from DAG config or use default
    if context.get('dag_run') and context['dag_run'].conf:
        training_file = context['dag_run'].conf.get('training_file')
    else:
        training_file = None
    
    # If no specific file provided, use most recent processed file
    if not training_file:
        data_dir = Path('/opt/airflow/data/cleaned')
        files = sorted(
            data_dir.glob('arxiv_papers_processed_*.json'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not files:
            raise FileNotFoundError(
                "No training data found. Please run the arXiv pipeline first to generate data, "
                "or specify a training_file in the DAG run config."
            )
        
        training_file = str(files[0])
        print(f"[SCHEMA] Using most recent file: {training_file}")
    
    return create_pipeline_schema(pipeline_name, training_file, **context)


def create_news_schema(**context):
    """
    Create schema for News API pipeline
    Uses a specific training data file
    """
    pipeline_name = 'news_api'
    
    # Get training file from DAG config or use default
    if context.get('dag_run') and context['dag_run'].conf:
        training_file = context['dag_run'].conf.get('training_file')
    else:
        training_file = None
    
    # If no specific file provided, use most recent categorized file
    if not training_file:
        data_dir = Path('/opt/airflow/data/cleaned')
        files = sorted(
            data_dir.glob('tech_news_categorized_*.json'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not files:
            raise FileNotFoundError(
                "No training data found. Please run the News API pipeline first to generate data, "
                "or specify a training_file in the DAG run config."
            )
        
        training_file = str(files[0])
        print(f"[SCHEMA] Using most recent file: {training_file}")
    
    return create_pipeline_schema(pipeline_name, training_file, **context)


def send_schema_creation_notification(result, context):
    """Send email notification after successful schema creation"""
    try:
        pipeline = result['pipeline']
        
        body = f"""
📋 SCHEMA CREATED SUCCESSFULLY

Pipeline: {pipeline.upper()}
Status: SUCCESS
Created At: {result.get('created_at', 'N/A')}

═══════════════════════════════════════════════════════════
SCHEMA DETAILS
═══════════════════════════════════════════════════════════
Suite Name: {result.get('suite_name', 'N/A')}
Expectations: {result.get('num_expectations', 0)}
Training File: {result.get('training_file', 'N/A')}
Schema File: {result.get('schema_file', 'N/A')}
File Size: {result.get('file_size_kb', 0)} KB

═══════════════════════════════════════════════════════════
SCHEMA LOCATIONS
═══════════════════════════════════════════════════════════
1. GE Artifacts (runtime):
   {result.get('schema_file', 'N/A')}

2. Version Control (git tracked):
   /opt/airflow/data/schema/{pipeline}_expectations.json

═══════════════════════════════════════════════════════════
WHAT THIS MEANS
═══════════════════════════════════════════════════════════
✓ Schema is now active and ready for validation
✓ All future data will be validated against this baseline
✓ You'll receive alerts if data deviates from expectations
✓ Validation task will now succeed (previously would fail)

═══════════════════════════════════════════════════════════
NEXT STEPS
═══════════════════════════════════════════════════════════
1. ✓ Schema created - DONE
2. Review the schema file for correctness
3. IMPORTANT: Commit schema to version control
   git add data/schema/{pipeline}_expectations.json
   git commit -m "Add GE schema for {pipeline} pipeline"
4. Run your regular pipeline - validation will now work
5. Monitor validation results in future runs

═══════════════════════════════════════════════════════════
UPDATING THE SCHEMA
═══════════════════════════════════════════════════════════
If you need to update the schema later:
1. Trigger this DAG again with config:
   {{"overwrite_schema": true, "training_file": "/path/to/new/data.json"}}
2. Or delete the schema files and run this DAG again

═══════════════════════════════════════════════════════════
AIRFLOW DASHBOARD
═══════════════════════════════════════════════════════════
View: http://localhost:8080/dags/{context['dag'].dag_id}/grid
"""
        
        alert_email.send_email_with_attachment(
            recipient_email="anirudhshrikanth65@gmail.com",
            subject=f"📋 Schema Created: {pipeline.upper()} Pipeline",
            body=body
        )
        
        print(f"[SCHEMA] Notification email sent")
        
    except Exception as e:
        print(f"[SCHEMA] Error sending notification: {e}")


def send_schema_error_notification(pipeline, error_message, training_file, context):
    """Send error notification when schema creation fails"""
    try:
        body = f"""
❌ SCHEMA CREATION FAILED

Pipeline: {pipeline.upper()}
Training File: {training_file}
Error Time: {datetime.now().isoformat()}

═══════════════════════════════════════════════════════════
ERROR DETAILS
═══════════════════════════════════════════════════════════
{error_message}

═══════════════════════════════════════════════════════════
POSSIBLE CAUSES
═══════════════════════════════════════════════════════════
1. Training data file not found or invalid path
2. Invalid JSON format in training file
3. Insufficient permissions to write schema files
4. Empty or corrupted training data
5. Missing required columns in data

═══════════════════════════════════════════════════════════
TROUBLESHOOTING STEPS
═══════════════════════════════════════════════════════════
1. Verify training data file exists:
   ls -lh {training_file}

2. Check if file is valid JSON:
   python -m json.tool {training_file}

3. Check file contents and structure

4. Verify directory permissions:
   ls -ld /opt/airflow/data/ge_artifacts/{pipeline}

5. Check Airflow task logs for detailed traceback

6. Try with a different training file

═══════════════════════════════════════════════════════════
IMPACT
═══════════════════════════════════════════════════════════
❌ Schema NOT created
❌ Validation tasks will fail until schema is created
❌ Data quality checks are not active

═══════════════════════════════════════════════════════════
NEXT STEPS
═══════════════════════════════════════════════════════════
1. Fix the issue identified above
2. Re-trigger this schema creation DAG
3. Verify schema files are created
4. Run validation pipeline to confirm

Airflow: http://localhost:8080/dags/{context['dag'].dag_id}/grid
"""
        
        alert_email.send_email_with_attachment(
            recipient_email="anirudhshrikanth65@gmail.com",
            subject=f"❌ Schema Creation Failed: {pipeline.upper()}",
            body=body
        )
        
        print(f"[SCHEMA] Error notification sent")
        
    except Exception as e:
        print(f"[SCHEMA] Error sending error notification: {e}")
