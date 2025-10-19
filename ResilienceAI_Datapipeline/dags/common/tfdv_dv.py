"""
MULTI-PIPELINE TFDV VALIDATOR
Manages separate schemas for News API and arXiv pipelines

Directory structure:
tfdv_artifacts/
├── news_api/
│   ├── schema.pbtxt
│   └── reports/
└── arxiv/
    ├── schema.pbtxt
    └── reports/
"""

import tensorflow_data_validation as tfdv
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSV field size limit handling
import csv
import sys
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)


class PipelineValidator:
    """Manages TFDV validation for a specific pipeline"""
    
    def __init__(self, pipeline_name: str, base_dir: str = "/app/tfdv_artifacts"):
        """
        Initialize validator for a specific pipeline
        
        Args:
            pipeline_name: Name of pipeline (e.g., 'news_api', 'arxiv')
            base_dir: Base directory for all TFDV artifacts (absolute path recommended)
        """
        self.pipeline_name = pipeline_name
        self.base_dir = Path(base_dir)
        self.pipeline_dir = self.base_dir / pipeline_name
        self.schema_path = self.pipeline_dir / "schema.pbtxt"
        self.reports_dir = self.pipeline_dir / "reports"
        
        logger.info(f"Initializing validator for '{pipeline_name}' at: {self.pipeline_dir}")
        
        # Create directories
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_json_to_dataframe(self, json_path: str) -> pd.DataFrame:
        """Load JSON file and convert to DataFrame"""
        logger.info(f"[{self.pipeline_name}] Loading JSON from {json_path}")
        
        try:
            # Try JSON Lines format first
            df = pd.read_json(json_path, lines=True)
            logger.info(f"   Loaded as JSON Lines: {len(df)} rows")
            return df
        except:
            pass
        
        try:
            # Try regular JSON array format
            df = pd.read_json(json_path)
            logger.info(f"   Loaded as JSON array: {len(df)} rows")
            return df
        except:
            pass
        
        try:
            # Try loading as plain JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                for key in ['data', 'records', 'items', 'results', 'articles']:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        logger.info(f"   Loaded from JSON key '{key}': {len(df)} rows")
                        return df
                
                df = pd.DataFrame([data])
                logger.info(f"   Loaded single JSON object: {len(df)} rows")
                return df
            
            elif isinstance(data, list):
                df = pd.DataFrame(data)
                logger.info(f"   Loaded from JSON list: {len(df)} rows")
                return df
            
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            raise ValueError(f"Could not parse JSON file {json_path}")
        
        raise ValueError(f"Could not parse JSON file {json_path}")
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect if file is CSV or JSON"""
        ext = Path(file_path).suffix.lower()
        if ext in ['.json', '.jsonl']:
            return 'json'
        elif ext == '.csv':
            return 'csv'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _generate_statistics(self, data_path: str):
        """Generate TFDV statistics from CSV or JSON file"""
        file_type = self._detect_file_type(data_path)
        
        if file_type == 'json':
            df = self._load_json_to_dataframe(data_path)
            temp_csv = self.pipeline_dir / "_temp.csv"
            df.to_csv(temp_csv, index=False)
            stats = tfdv.generate_statistics_from_csv(str(temp_csv))
            temp_csv.unlink()
        else:
            stats = tfdv.generate_statistics_from_csv(data_path)
        
        return stats
    
    def create_schema(self, train_data_path: str) -> bool:
        """
        Create baseline schema from training data
        
        Args:
            train_data_path: Path to training CSV or JSON file
        
        Returns:
            True if successful
        """
        logger.info(f"📊 [{self.pipeline_name}] Creating schema from {train_data_path}")
        
        try:
            stats = self._generate_statistics(train_data_path)
            schema = tfdv.infer_schema(statistics=stats)
            tfdv.write_schema_text(schema, str(self.schema_path))
            
            logger.info(f"✅ [{self.pipeline_name}] Schema created: {self.schema_path}")
            logger.info(f"   Features: {len(schema.feature)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ [{self.pipeline_name}] Schema creation failed: {e}")
            raise
    
    def validate_data(self, data_path: str, raise_on_error: bool = True) -> bool:
        """
        Validate data against schema
        
        Args:
            data_path: Path to CSV or JSON file to validate
            raise_on_error: If True, raise error on validation failure
        
        Returns:
            True if validation passes
        """
        logger.info(f"🔍 [{self.pipeline_name}] Validating {data_path}")
        
        if not self.schema_path.exists():
            error_msg = f"Schema not found at {self.schema_path}. Run create_schema() first!"
            logger.error(f"❌ [{self.pipeline_name}] {error_msg}")
            if raise_on_error:
                raise FileNotFoundError(error_msg)
            return False
        
        try:
            stats = self._generate_statistics(data_path)
            schema = tfdv.load_schema_text(str(self.schema_path))
            anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
            
            if anomalies.anomaly_info:
                logger.error(f"❌ [{self.pipeline_name}] VALIDATION FAILED: {len(anomalies.anomaly_info)} anomalies")
                
                for feature_name, anomaly_info in anomalies.anomaly_info.items():
                    logger.error(f"   • {feature_name}: {anomaly_info.short_description}")
                
                if raise_on_error:
                    raise ValueError(f"Data validation failed with {len(anomalies.anomaly_info)} anomalies")
                return False
            
            logger.info(f"✅ [{self.pipeline_name}] Validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ [{self.pipeline_name}] Validation error: {e}")
            if raise_on_error:
                raise
            return False
    
    def generate_report(self, data_paths: Dict[str, str]) -> Dict:
        """
        Generate detailed validation report
        
        Args:
            data_paths: Dict of dataset names and paths
                       Example: {'train': 'data/train.json', 'test': 'data/test.json'}
        
        Returns:
            Report dictionary with validation results
        """
        logger.info(f"📝 [{self.pipeline_name}] Generating validation report")
        
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found at {self.schema_path}")
        
        schema = tfdv.load_schema_text(str(self.schema_path))
        
        report_data = {
            'pipeline': self.pipeline_name,
            'schema_path': str(self.schema_path),
            'datasets': {},
            'has_issues': False
        }
        
        for name, path in data_paths.items():
            logger.info(f"   Checking {name}...")
            
            try:
                stats = self._generate_statistics(path)
                anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
                
                # Save statistics
                stats_file = self.reports_dir / f"{name}_stats.pb"
                tfdv.write_stats_text(stats, str(stats_file))
                
                # Save anomalies if any
                has_anomalies = bool(anomalies.anomaly_info)
                if has_anomalies:
                    anomalies_file = self.reports_dir / f"{name}_anomalies.pbtxt"
                    tfdv.write_anomalies_text(anomalies, str(anomalies_file))
                    report_data['has_issues'] = True
                
                file_type = self._detect_file_type(path)
                report_data['datasets'][name] = {
                    'path': path,
                    'file_type': file_type,
                    'num_examples': stats.datasets[0].num_examples,
                    'has_anomalies': has_anomalies,
                    'anomaly_count': len(anomalies.anomaly_info) if has_anomalies else 0,
                    'anomalies': dict(anomalies.anomaly_info) if has_anomalies else {}
                }
                
            except Exception as e:
                logger.error(f"   Error processing {name}: {e}")
                report_data['datasets'][name] = {
                    'path': path,
                    'error': str(e)
                }
        
        # Print summary
        self._print_report_summary(report_data)
        
        return report_data
    
    def _print_report_summary(self, report_data: Dict):
        """Print formatted report summary"""
        print(f"\n{'='*70}")
        print(f"TFDV VALIDATION REPORT - {self.pipeline_name.upper()}")
        print(f"{'='*70}")
        
        for name, info in report_data['datasets'].items():
            if 'error' in info:
                print(f"{name:15} | ERROR: {info['error']}")
            else:
                status = "❌ ISSUES" if info['has_anomalies'] else "✅ OK"
                print(f"{name:15} | {info['file_type']:4} | {info['num_examples']:8} rows | {status}")
                if info['has_anomalies']:
                    print(f"                | {info['anomaly_count']} anomalies:")
                    for feat, anom in list(info['anomalies'].items())[:3]:
                        print(f"                |   • {feat}: {anom.short_description}")
                    if len(info['anomalies']) > 3:
                        print(f"                |   ... and {len(info['anomalies']) - 3} more")
        
        print(f"{'='*70}\n")


# ============================================================================
# CONVENIENCE FUNCTIONS FOR MULTI-PIPELINE USAGE
# ============================================================================

class MultiPipelineValidator:
    """Manages validation for multiple pipelines"""
    
    def __init__(self, base_dir: str = "/app/tfdv_artifacts"):
        self.base_dir = Path(base_dir).resolve()  # Use absolute path
        self.validators: Dict[str, PipelineValidator] = {}
        logger.info(f"Multi-pipeline validator initialized at: {self.base_dir}")
    
    def get_validator(self, pipeline_name: str) -> PipelineValidator:
        """Get or create validator for a pipeline"""
        if pipeline_name not in self.validators:
            self.validators[pipeline_name] = PipelineValidator(pipeline_name, str(self.base_dir))
        return self.validators[pipeline_name]
    
    def create_all_schemas(self, pipeline_configs: Dict[str, str]):
        """
        Create schemas for multiple pipelines
        
        Args:
            pipeline_configs: Dict of pipeline_name -> training_data_path
                             Example: {'news_api': 'data/news_train.json',
                                      'arxiv': 'data/arxiv_train.json'}
        """
        results = {}
        for pipeline_name, train_path in pipeline_configs.items():
            validator = self.get_validator(pipeline_name)
            try:
                validator.create_schema(train_path)
                results[pipeline_name] = 'success'
            except Exception as e:
                results[pipeline_name] = f'failed: {e}'
        return results
    
    def validate_all(self, pipeline_data: Dict[str, str], raise_on_error: bool = False) -> Dict[str, bool]:
        """
        Validate data for multiple pipelines
        
        Args:
            pipeline_data: Dict of pipeline_name -> data_path
            raise_on_error: If True, raise on first failure
        
        Returns:
            Dict of pipeline_name -> validation_result
        """
        results = {}
        for pipeline_name, data_path in pipeline_data.items():
            validator = self.get_validator(pipeline_name)
            results[pipeline_name] = validator.validate_data(data_path, raise_on_error)
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Setting up validation for News API and arXiv pipelines
    """
    
    # Initialize multi-pipeline validator
    mpv = MultiPipelineValidator(base_dir="tfdv_artifacts")
    
    print("\n" + "="*70)
    print("STEP 1: Create schemas for both pipelines")
    print("="*70)
    
    # Create schemas
    schema_results = mpv.create_all_schemas({
        'news_api': '/app/data/cleaned/tech_news_categorized_20251016_222651.json',
        'arxiv': '/app/data/cleaned/arxiv_papers_processed_20251018_203954.json'
    })
    
    print("\n" + "="*70)
    print("STEP 2: Validate new data for both pipelines")
    print("="*70)
    
    # Validate data (don't raise on error, just report)
    validation_results = mpv.validate_all({
        'news_api': '/app/data/cleaned/tech_news_categorized_20251017_000141.json',
        'arxiv': '/app/data/cleaned/arxiv_papers_processed_20251017_004721.json'
    }, raise_on_error=False)
    
    print("\n" + "="*70)
    print("STEP 3: Generate detailed reports")
    print("="*70)
    
    # Generate reports for each pipeline
    news_validator = mpv.get_validator('news_api')
    news_report = news_validator.generate_report({
        'train': '/app/data/cleaned/arxiv_papers_processed_20251018_203954.json',
        'validation': '/app/data/cleaned/arxiv_papers_processed_20251017_004721.json'
    })
    
    arxiv_validator = mpv.get_validator('arxiv')
    arxiv_report = arxiv_validator.generate_report({
        'train': '/app/data/cleaned/arxiv_papers_processed_20251018_203954.json',
        'validation': '/app/data/cleaned/arxiv_papers_processed_20251017_004721.json'
    })
    
    print("\n✅ Multi-pipeline validation complete!")
    print(f"   News API reports: {news_validator.reports_dir}")
    print(f"   arXiv reports: {arxiv_validator.reports_dir}")