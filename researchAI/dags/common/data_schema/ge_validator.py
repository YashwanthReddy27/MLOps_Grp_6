""" GREAT EXPECTATIONS VALIDATOR. Works with GE 0.18.8 """

import great_expectations as ge
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import os
import traceback
import csv
import sys

class PipelineValidator:
    """Manages Great Expectations validation for a specific pipeline"""
    
    def __init__(self, pipeline_name: str, base_dir: Path = Path('/home/airflow/gcs/data/ge_artifacts')):
        """Initialize validator"""
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        max_int = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)
        self.pipeline_name = pipeline_name
        self.expectation_suite_name = f"{pipeline_name}_suite"
        
        try:
            self.ge_root_dir = base_dir / pipeline_name
            
            # Attempt to create all directories
            self.ge_root_dir.mkdir(parents=True, exist_ok=True)
            self.expectations_dir = self.ge_root_dir / "expectations"
            self.expectations_dir.mkdir(parents=True, exist_ok=True)

            self.validations_dir = self.ge_root_dir / "validations"
            self.validations_dir.mkdir(parents=True, exist_ok=True)

            self.uncommitted_dir = self.ge_root_dir / "uncommitted"
            self.uncommitted_dir.mkdir(parents=True, exist_ok=True)
            
            # If we get here, directories were created successfully
            self.logger.info(f"Initializing GE validator for '{pipeline_name}' at: {self.ge_root_dir}")
            self.logger.info(f"Great Expectations version: {ge.__version__}")
            self.logger.info(f"Expectations dir: {self.expectations_dir}")
            self.logger.info(f"Validations dir: {self.validations_dir}")
            self.logger.info(f"Uncommitted dir: {self.uncommitted_dir}")

            # Success - exit the method
            return

        except Exception as e:
            # Final fallback - set attributes even if directories don't exist
            self.logger.error(f"Could not create any directories. Setting paths without creation: {e}")
            self.expectations_dir = self.ge_root_dir / "expectations"
            self.validations_dir = self.ge_root_dir / "validations"
            self.uncommitted_dir = self.ge_root_dir / "uncommitted"

    def _load_json_to_dataframe(self, json_path: str) -> pd.DataFrame:
        """Load JSON file and convert to DataFrame"""
        self.logger.info(f"[{self.pipeline_name}] Loading JSON from {json_path}")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Data file not found: {json_path}")
        
        try:
            df = pd.read_json(json_path, lines=True)
            self.logger.info(f"   Loaded as JSON Lines: {len(df)} rows, {len(df.columns)} columns")
            return df
        except:
            pass
        
        try:
            df = pd.read_json(json_path)
            self.logger.info(f"   Loaded as JSON array: {len(df)} rows, {len(df.columns)} columns")
            return df
        except:
            pass
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                for key in ['data', 'records', 'items', 'results', 'articles', 'papers']:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        self.logger.info(f"   Loaded from JSON key '{key}': {len(df)} rows, {len(df.columns)} columns")
                        return df
                df = pd.DataFrame([data])
                self.logger.info(f"   Loaded single record: {len(df.columns)} columns")
                return df
            elif isinstance(data, list):
                df = pd.DataFrame(data)
                self.logger.info(f"   Loaded list: {len(df)} rows, {len(df.columns)} columns")
                return df
        except Exception as e:
            self.logger.error(f"Failed to load JSON: {e}")
            raise ValueError(f"Could not parse JSON file {json_path}: {e}")
        
        raise ValueError(f"Could not parse JSON file {json_path}")
    
    def create_schema(self, train_data_path: str) -> bool:
        """Create baseline schema - FIXED for GE 0.18.8"""
        self.logger.info(f"📊 [{self.pipeline_name}] Creating schema from {train_data_path}")

        try:
            df = self._load_json_to_dataframe(train_data_path)

            self.logger.info(f"[{self.pipeline_name}] Creating GE DataFrame...")
            ge_df = ge.from_pandas(df)
            
            # Set suite name (property, not method in GE 0.18.x)
            ge_df.expectation_suite_name = self.expectation_suite_name
            self.logger.info(f"[{self.pipeline_name}] Set expectation suite name: {self.expectation_suite_name}")

            self.logger.info(f"[{self.pipeline_name}] Adding expectations...")

            # Table-level expectations
            ge_df.expect_table_row_count_to_be_between(min_value=1, max_value=None)
            ge_df.expect_table_column_count_to_equal(value=len(df.columns))
            
            expectations_added = 2
            
            # Column-level expectations
            for column in df.columns:
                ge_df.expect_column_to_exist(column=column)
                expectations_added += 1
                
                null_percent = df[column].isnull().sum() / len(df) * 100
                if null_percent < 100:
                    ge_df.expect_column_values_to_not_be_null(column=column, mostly=0.8)
                    expectations_added += 1
                
                dtype = str(df[column].dtype)
                if 'int' in dtype or 'float' in dtype:
                    try:
                        min_val = float(df[column].min())
                        max_val = float(df[column].max())
                        if pd.notna(min_val) and pd.notna(max_val):
                            ge_df.expect_column_values_to_be_between(
                                column=column,
                                min_value=min_val * 0.5,
                                max_value=max_val * 1.5,
                                mostly=0.95
                            )
                            expectations_added += 1
                    except Exception as e:
                        self.logger.warning(f"   Could not add range expectation for {column}: {e}")

            self.logger.info(f"[{self.pipeline_name}] Added {expectations_added} expectations")

            # Get suite and save
            suite = ge_df.get_expectation_suite(discard_failed_expectations=False)
            suite_dict = suite.to_json_dict()
            
            # Save to GE artifacts expectations folder
            ge_suite_path = self.expectations_dir / f"{self.expectation_suite_name}.json"
            self.logger.info(f"[{self.pipeline_name}] Saving suite to: {ge_suite_path}")
            with open(ge_suite_path, "w") as f:
                json.dump(suite_dict, f, indent=2)
            self.logger.info(f"[{self.pipeline_name}] ✓ Saved to expectations folder")

            # Save to version control folder
            project_root = self.ge_root_dir.parent.parent
            schema_dir = project_root / "data" / "schema"
            schema_dir.mkdir(parents=True, exist_ok=True)
            vc_suite_path = schema_dir / f"{self.pipeline_name}_expectations.json"
            self.logger.info(f"[{self.pipeline_name}] Saving suite to version control: {vc_suite_path}")
            with open(vc_suite_path, "w") as f:
                json.dump(suite_dict, f, indent=2)
            self.logger.info(f"[{self.pipeline_name}] ✓ Saved to version control")

            # Save metadata to uncommitted folder
            metadata = {
                'created_at': datetime.now().isoformat(),
                'suite_name': self.expectation_suite_name,
                'source_file': train_data_path,
                'num_expectations': len(suite_dict.get('expectations', [])),
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns)
            }
            metadata_path = self.uncommitted_dir / f"schema_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"[{self.pipeline_name}] ✓ Saved metadata to uncommitted")
            self.logger.info(f"✅ [{self.pipeline_name}] Schema created successfully")
            self.logger.info(f"   - Expectations: {len(suite_dict.get('expectations', []))}")
            self.logger.info(f"   - GE artifacts: {ge_suite_path}")
            self.logger.info(f"   - Version control: {vc_suite_path}")
            self.logger.info(f"   - Metadata: {metadata_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [{self.pipeline_name}] Schema creation failed: {e}")
            traceback.print_exc()
            raise
    
    def get_expectation_suite(self):
        """Load expectation suite from file"""
        ge_suite_path = self.expectations_dir / f"{self.expectation_suite_name}.json"
        
        if not ge_suite_path.exists():
            self.logger.info(f"[{self.pipeline_name}] Suite file not found: {ge_suite_path}")
            return None

        self.logger.info(f"[{self.pipeline_name}] Loading suite from: {ge_suite_path}")
        try:
            with open(ge_suite_path, 'r') as f:
                suite_dict = json.load(f)
            self.logger.info(f"[{self.pipeline_name}] ✓ Suite loaded with {len(suite_dict.get('expectations', []))} expectations")
            return suite_dict
        except Exception as e:
            self.logger.error(f"[{self.pipeline_name}] Error loading suite: {e}")
            return None
    
    def validate_data(self, data_path: str, raise_on_error: bool = True) -> bool:
        """Validate data and save results"""
        self.logger.info(f"🔍 [{self.pipeline_name}] Starting validation for: {data_path}")
        
        # Check if data file exists
        if not os.path.exists(data_path):
            error_msg = f"Data file not found: {data_path}"
            self.logger.error(f"❌ [{self.pipeline_name}] {error_msg}")
            if raise_on_error:
                raise FileNotFoundError(error_msg)
            return False
        
        # Load expectation suite
        suite_dict = self.get_expectation_suite()
        if not suite_dict:
            error_msg = f"Suite '{self.expectation_suite_name}' not found. Run create_schema() first!"
            self.logger.error(f"❌ [{self.pipeline_name}] {error_msg}")
            if raise_on_error:
                raise FileNotFoundError(error_msg)
            return False

        try:
            # Load data
            self.logger.info(f"[{self.pipeline_name}] Loading data for validation...")
            df = self._load_json_to_dataframe(data_path)
            
            # Create GE DataFrame with suite
            self.logger.info(f"[{self.pipeline_name}] Creating GE DataFrame with suite...")
            ge_df = ge.from_pandas(df, expectation_suite=suite_dict)
            
            # Run validation
            self.logger.info(f"[{self.pipeline_name}] Running validation...")
            results = ge_df.validate()
            
            # Prepare result filename
            result_filename = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result_file = self.validations_dir / result_filename
            
            # Save results to validations folder
            self.logger.info(f"[{self.pipeline_name}] Saving validation results to: {result_file}")
            try:
                results_dict = results.to_json_dict()
                with open(result_file, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                self.logger.info(f"[{self.pipeline_name}] ✓ Validation results saved ({os.path.getsize(result_file)} bytes)")
            except Exception as save_error:
                self.logger.error(f"[{self.pipeline_name}] ❌ Failed to save validation results: {save_error}")
                # Continue - don't fail validation because of save error
            
            # Save summary to uncommitted folder
            summary = {
                'validated_at': datetime.now().isoformat(),
                'data_file': data_path,
                'suite_name': self.expectation_suite_name,
                'success': results.success,
                'statistics': results.statistics,
                'results_evaluated': len(results.results),
                'results_passed': sum(1 for r in results.results if r.success),
                'results_failed': sum(1 for r in results.results if not r.success),
                'num_rows': len(df),
                'num_columns': len(df.columns)
            }
            summary_path = self.uncommitted_dir / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                self.logger.info(f"[{self.pipeline_name}] ✓ Summary saved to uncommitted: {summary_path}")
            except Exception as e:
                self.logger.warning(f"[{self.pipeline_name}] Could not save summary: {e}")

            # Check validation results
            if not results.success:
                self.logger.error(f"❌ [{self.pipeline_name}] VALIDATION FAILED")
                failed = [r for r in results.results if not r.success]
                self.logger.error(f"   Failed expectations: {len(failed)}")
                self.logger.error(f"   Success rate: {results.statistics.get('success_percent', 0):.1f}%")

                # Log first few failures
                for i, r in enumerate(failed[:5], 1):
                    exp_type = r.expectation_config.expectation_type
                    column = r.expectation_config.kwargs.get('column', 'TABLE')
                    self.logger.error(f"   {i}. {column} - {exp_type}")
                
                if len(failed) > 5:
                    self.logger.error(f"   ... and {len(failed) - 5} more failures")
                
                self.logger.error(f"   Full results in: {result_file}")
                
                if raise_on_error:
                    raise ValueError("Data validation failed.")
                return False
            
            self.logger.info(f"✅ [{self.pipeline_name}] Validation PASSED")
            self.logger.info(f"   Success rate: {results.statistics.get('success_percent', 100):.1f}%")
            self.logger.info(f"   Results file: {result_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ [{self.pipeline_name}] Validation error: {e}")
            traceback.print_exc()
            
            # Save error info to uncommitted
            error_info = {
                'error_at': datetime.now().isoformat(),
                'data_file': data_path,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
            error_path = self.uncommitted_dir / f"validation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(error_path, 'w') as f:
                    json.dump(error_info, f, indent=2)
            except:
                pass
            
            if raise_on_error:
                raise
            return False
    
    def generate_report(self, data_paths: Dict[str, str]) -> Dict:
        """Generate validation report for multiple datasets"""
        self.logger.info(f"📝 [{self.pipeline_name}] Generating validation report")
        
        suite_dict = self.get_expectation_suite()
        if not suite_dict:
            raise FileNotFoundError(f"Suite '{self.expectation_suite_name}' not found.")
        
        report_data = {
            'pipeline': self.pipeline_name,
            'expectation_suite_name': self.expectation_suite_name,
            'generated_at': datetime.now().isoformat(),
            'datasets': {},
            'has_issues': False
        }
        
        for name, path in data_paths.items():
            self.logger.info(f"   Checking {name}...")
            try:
                df = self._load_json_to_dataframe(path)
                ge_df = ge.from_pandas(df, expectation_suite=suite_dict)
                validation_result = ge_df.validate()
                
                failed_expectations = [res for res in validation_result.results if not res.success]
                
                report_data['datasets'][name] = {
                    'path': path,
                    'file_type': 'json',
                    'num_examples': len(df),
                    'num_columns': len(df.columns),
                    'has_anomalies': not validation_result.success,
                    'anomaly_count': len(failed_expectations),
                    'success_percent': validation_result.statistics.get('success_percent', 0),
                    'anomalies': {}
                }
                
                for res in failed_expectations:
                    column = res.expectation_config.kwargs.get('column', 'TABLE')
                    expectation_type = res.expectation_config.expectation_type
                    key = f"{column}.{expectation_type}"
                    
                    report_data['datasets'][name]['anomalies'][key] = {
                        "short_description": f"Failed: {expectation_type}",
                        "result": str(res.result)[:200] if hasattr(res, 'result') else 'N/A'
                    }
                
                if not validation_result.success:
                    report_data['has_issues'] = True
                
            except Exception as e:
                self.logger.error(f"   Error processing {name}: {e}")
                report_data['datasets'][name] = {
                    'path': path,
                    'error': str(e)
                }
        
        # Save report to uncommitted
        report_path = self.uncommitted_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            self.logger.info(f"[{self.pipeline_name}] ✓ Report saved to: {report_path}")
        except Exception as e:
            self.logger.warning(f"[{self.pipeline_name}] Could not save report: {e}")

        return report_data

