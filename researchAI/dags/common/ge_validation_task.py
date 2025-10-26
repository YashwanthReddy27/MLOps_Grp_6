"""
Great Expectations Data Validation Tasks for Airflow DAGs
FIXED for Universal GE Validator
"""

from common.ge_validator import PipelineValidator
from common.send_email import AlertEmail
import json
from datetime import datetime
from pathlib import Path
import os


alert_email = AlertEmail()

def validate_data_quality(**context):
    """
    Validate data quality using Great Expectations
    """
    try:
        # Determine which pipeline we're running
        dag_id = context['dag'].dag_id
        
        if 'arxiv' in dag_id.lower():
            pipeline_name = 'arxiv'
            previous_task_id = 'process_and_categorize_papers'
            xcom_key = 'processed_result'
        else:
            pipeline_name = 'news_api'
            previous_task_id = 'extract_keywords_and_categorize'
            xcom_key = 'categorized_result'
        
        print(f"[VALIDATION] Starting Great Expectations validation for {pipeline_name} pipeline")
        
        # Initialize validator
        validator = PipelineValidator(pipeline_name=pipeline_name)
        
        # Get processed data file from previous task
        result = context['ti'].xcom_pull(task_ids=previous_task_id, key=xcom_key)
        
        if not result or not result.get('filename'):
            print("[VALIDATION] No data to validate")
            context['ti'].xcom_push(key='validation_result', value={
                'status': 'skipped',
                'message': 'No new data to validate'
            })
            return True
        
        data_file = result['filename']
        print(f"[VALIDATION] Validating file: {data_file}")
        
        # Check if data file exists
        if not Path(data_file).exists():
            print(f"[VALIDATION] ERROR: Data file not found: {data_file}")
            context['ti'].xcom_push(key='validation_result', value={
                'status': 'error',
                'message': f'Data file not found: {data_file}'
            })
            return True
        
        # Check if schema exists by looking for the suite file
        suite = validator._get_expectation_suite()
        schema_exists = suite is not None
        
        if schema_exists:
            print(f"[VALIDATION] Found existing schema: {validator.expectation_suite_name}")
        else:
            print(f"[VALIDATION] No existing schema found")
        
        if not schema_exists:
            # First run - create baseline schema
            print(f"[VALIDATION] Creating baseline schema from {data_file}")
            try:
                validator.create_schema(data_file)
                
                context['ti'].xcom_push(key='validation_result', value={
                    'status': 'schema_created',
                    'message': 'Baseline schema created from current data',
                    'suite_name': validator.expectation_suite_name,
                    'has_anomalies': False,
                    'data_file': data_file
                })
                
                # Send notification
                send_schema_created_email(pipeline_name, validator, context)
                
                print("[VALIDATION] Schema created successfully")
                return True
                
            except Exception as e:
                print(f"[VALIDATION] Error creating schema: {e}")
                import traceback
                traceback.print_exc()
                
                send_error_alert(pipeline_name, f"Schema creation failed: {e}", context)
                
                context['ti'].xcom_push(key='validation_result', value={
                    'status': 'error',
                    'message': f'Schema creation failed: {e}'
                })
                return True
        
        # Schema exists - validate data
        print(f"[VALIDATION] Validating against schema: {validator.expectation_suite_name}")
        
        try:
            # Use the validator's validate_data method
            # It will raise an exception if validation fails and raise_on_error=True
            # We want to catch failures and continue, so use raise_on_error=False
            validation_passed = validator.validate_data(data_file, raise_on_error=False)
            
            if validation_passed:
                print("[VALIDATION] Data validation PASSED")
                
                context['ti'].xcom_push(key='validation_result', value={
                    'status': 'success',
                    'message': 'Data validation passed',
                    'has_anomalies': False,
                    'data_file': data_file
                })
                return True
            else:
                # Validation failed - load the results to get details
                print("[VALIDATION] Data validation FAILED - Anomalies detected")
                
                # Load the most recent validation result
                validations_dir = validator.validations_dir
                validation_files = sorted(validations_dir.glob("validation_*.json"), key=os.path.getmtime, reverse=True)
                
                if validation_files:
                    with open(validation_files[0], 'r') as f:
                        results_dict = json.load(f)
                    
                    # Extract failed expectations
                    failed_expectations = [r for r in results_dict.get('results', []) if not r.get('success', True)]
                    
                    print(f"[VALIDATION] Found {len(failed_expectations)} failed expectations")
                    
                    # Prepare anomaly summary
                    anomaly_summary = {
                        'pipeline': pipeline_name,
                        'status': 'anomalies_detected',
                        'file': data_file,
                        'anomaly_count': len(failed_expectations),
                        'success_percent': results_dict.get('statistics', {}).get('success_percent', 0),
                        'anomalies': {},
                        'timestamp': context['execution_date'].isoformat()
                    }
                    
                    # Extract anomaly details
                    for res in failed_expectations[:20]:  # Limit to first 20
                        exp_config = res.get('expectation_config', {})
                        column = exp_config.get('kwargs', {}).get('column', 'TABLE')
                        expectation_type = exp_config.get('expectation_type', 'unknown')
                        key = f"{column}.{expectation_type}"
                        
                        anomaly_summary['anomalies'][key] = {
                            "short_description": f"Failed: {expectation_type}",
                            "result": str(res.get('result', 'N/A'))[:200]
                        }
                    
                    context['ti'].xcom_push(key='validation_result', value=anomaly_summary)
                    
                    # Send alert
                    send_anomaly_alert(anomaly_summary, context)
                else:
                    # No validation file found, but validation failed
                    anomaly_summary = {
                        'pipeline': pipeline_name,
                        'status': 'anomalies_detected',
                        'file': data_file,
                        'anomaly_count': 0,
                        'timestamp': context['execution_date'].isoformat()
                    }
                    context['ti'].xcom_push(key='validation_result', value=anomaly_summary)
                
                # Continue pipeline but log the issues
                print("[VALIDATION]  Continuing pipeline despite anomalies")
                return True
                
        except Exception as e:
            print(f"[VALIDATION] Error during validation: {e}")
            import traceback
            traceback.print_exc()
            
            context['ti'].xcom_push(key='validation_result', value={
                'status': 'error',
                'message': str(e),
                'has_anomalies': False
            })
            
            send_error_alert(pipeline_name, str(e), context)
            return True
        
    except Exception as e:
        print(f"[VALIDATION] Critical error: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            send_error_alert('unknown', str(e), context)
        except:
            pass
        
        return False


def send_schema_created_email(pipeline, validator, context):
    """Send email notification when a new schema is created"""
    ge_dir = validator.ge_root_dir
    ge_schema_path = ge_dir / "expectations" / f"{validator.expectation_suite_name}.json"
    
    project_root = ge_dir.parent.parent
    vc_schema_path = project_root / "data" / "schema" / f"{pipeline}_expectations.json"
    
    try:
        body = f"""
📋 NEW SCHEMA CREATED

Pipeline: {pipeline.upper()}
Execution Date: {context['execution_date']}
Created At: {datetime.now().isoformat()}

═══════════════════════════════════════════════════════════
INFORMATION
═══════════════════════════════════════════════════════════
This is the first run of Great Expectations validation for 
this pipeline. A baseline schema has been automatically 
created from your current data.

This schema will be used to validate all future data batches.

═══════════════════════════════════════════════════════════
SCHEMA LOCATIONS
═══════════════════════════════════════════════════════════
1. GE Artifacts (runtime):
   {ge_schema_path}

2. Version Control (git tracked):
   {vc_schema_path}

═══════════════════════════════════════════════════════════
NEXT STEPS
═══════════════════════════════════════════════════════════
✓ Schema is now active
✓ Future runs will validate data against this baseline
✓ You'll receive alerts if data deviates from expectations

RECOMMENDED ACTIONS:
1. Review the schema file
2. Commit the schema to version control (data/schema/)
3. Monitor future validation runs

═══════════════════════════════════════════════════════════
AIRFLOW DASHBOARD
═══════════════════════════════════════════════════════════
View: http://localhost:8080/dags/{context['dag'].dag_id}/grid

═══════════════════════════════════════════════════════════
"""
        
        alert_email.send_email_with_attachment(
            recipient_email="anirudhshrikanth65@gmail.com",
            subject=f"📋 New Schema Created: {pipeline.upper()} Pipeline",
            body=body
        )
        
        print(f"[ALERT] Schema creation notification sent")
        
    except Exception as e:
        print(f"[ALERT] Error sending schema notification: {e}")


def send_anomaly_alert(anomaly_info, context):
    """Send email alert when data anomalies are detected"""
    try:
        pipeline = anomaly_info['pipeline']
        anomaly_count = anomaly_info['anomaly_count']
        anomalies = anomaly_info.get('anomalies', {})
        success_percent = anomaly_info.get('success_percent', 0)
        
        body = f"""
🚨 DATA QUALITY ALERT - Anomalies Detected!

Pipeline: {pipeline.upper()}
Execution Date: {context['execution_date']}
Data File: {anomaly_info['file']}
Alert Time: {datetime.now().isoformat()}

═══════════════════════════════════════════════════════════
ANOMALY SUMMARY
═══════════════════════════════════════════════════════════
Total Anomalies Detected: {anomaly_count}
Validation Success Rate: {success_percent:.1f}%

═══════════════════════════════════════════════════════════
DETAILED ANOMALIES
═══════════════════════════════════════════════════════════
"""
        
        for i, (feature_name, anomaly_detail) in enumerate(list(anomalies.items())[:10], 1):
            body += f"\n{i}. Feature: {feature_name}\n"
            body += f"   Issue: {anomaly_detail.get('short_description', 'N/A')}\n"
            result = anomaly_detail.get('result', '')
            if result and result != 'N/A':
                body += f"   Details: {result}\n"
        
        if len(anomalies) > 10:
            body += f"\n... and {len(anomalies) - 10} more anomalies\n"
        
        body += f"""

═══════════════════════════════════════════════════════════
ACTION REQUIRED
═══════════════════════════════════════════════════════════
1. Review the anomalies above
2. Check if your data source has changed
3. Investigate unexpected patterns
4. Update schema if changes are intentional

═══════════════════════════════════════════════════════════
PIPELINE STATUS
═══════════════════════════════════════════════════════════
⚠️  Pipeline continued but data quality issues were detected.

Airflow: http://localhost:8080/dags/{context['dag'].dag_id}/grid
"""
        
        alert_email.send_email_with_attachment(
            recipient_email=os.getenv('RECIPIENT_EMAIL'),
            subject=f"🚨 Data Quality Alert: {pipeline.upper()} - {anomaly_count} Anomalies",
            body=body
        )
        
        print(f"[ALERT] Anomaly alert sent")
        
    except Exception as e:
        print(f"[ALERT] Error sending anomaly alert: {e}")


def send_error_alert(pipeline, error_message, context):
    """Send email alert when validation encounters an error"""
    try:
        body = f"""
⚠️  DATA VALIDATION ERROR

Pipeline: {pipeline.upper()}
Execution Date: {context['execution_date']}
Error Time: {datetime.now().isoformat()}

═══════════════════════════════════════════════════════════
ERROR DETAILS
═══════════════════════════════════════════════════════════
{error_message}

═══════════════════════════════════════════════════════════
IMPACT
═══════════════════════════════════════════════════════════
Data validation could not complete. Pipeline continued but
data quality was not verified.

⚠️  WARNING: Data processed WITHOUT quality validation!

═══════════════════════════════════════════════════════════
ACTION REQUIRED
═══════════════════════════════════════════════════════════
1. Check Airflow task logs
2. Verify GE setup
3. Check data file format
4. Fix the issue

Airflow: http://localhost:8080/dags/{context['dag'].dag_id}/grid
"""
        
        alert_email.send_email_with_attachment(
            sender_email="SENDER_EMAIL",
            sender_password="SENDER_PASSWORD",
            recipient_email="RECIPIENT_EMAIL",
            subject=f"⚠️ Validation Error: {pipeline.upper()} Pipeline",
            body=body
        )
        
        print(f"[ALERT] Error alert sent")
        
    except Exception as e:
        print(f"[ALERT] Error sending error alert: {e}")


def generate_data_statistics_report(**context):
    """Generate comprehensive data quality report"""
    try:
        dag_id = context['dag'].dag_id
        pipeline_name = 'arxiv' if 'arxiv' in dag_id.lower() else 'news_api'
        
        print(f"[REPORT] Generating report for {pipeline_name}")
        
        validator = PipelineValidator(pipeline_name=pipeline_name)
        
        suite = validator._get_expectation_suite()
        if not suite:
            print("[REPORT] No schema found")
            context['ti'].xcom_push(key='report_result', value={
                'status': 'skipped',
                'message': 'No schema found'
            })
            return True
        
        if pipeline_name == 'arxiv':
            data_dir = '/opt/airflow/data/cleaned'
            file_prefix = 'arxiv_papers_processed_'
        else:
            data_dir = '/opt/airflow/data/cleaned'
            file_prefix = 'tech_news_categorized_'
        
        data_dir_path = Path(data_dir)
        if not data_dir_path.exists():
            print(f"[REPORT] Data directory not found: {data_dir}")
            return True
        
        files = sorted(
            [f for f in data_dir_path.glob(f"{file_prefix}*.json")],
            key=os.path.getmtime,
            reverse=True
        )[:3]
        
        if not files:
            print("[REPORT] No processed files found")
            return True
        
        data_paths = {f"batch_{i+1}": str(f) for i, f in enumerate(files)}
        
        print(f"[REPORT] Generating report for {len(data_paths)} batches")
        report = validator.generate_report(data_paths)
        
        report_dir = validator.ge_root_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        report_file = report_dir / f"summary_report_{context['execution_date'].strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[REPORT] Report saved to {report_file}")
        
        context['ti'].xcom_push(key='report_result', value={
            'status': 'success',
            'report_file': str(report_file),
            'has_issues': report['has_issues']
        })
        
        return True
        
    except Exception as e:
        print(f"[REPORT] Error: {e}")
        import traceback
        traceback.print_exc()
        return False