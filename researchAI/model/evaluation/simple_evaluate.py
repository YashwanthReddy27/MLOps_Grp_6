"""
Simple Model Evaluation with Response Comparison
Assumes indexes are already built
"""
import json
import logging
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import TechTrendsRAGPipeline
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


class SimpleModelEvaluator:
    """Simple evaluator with response comparison"""
    
    def __init__(self, pipeline: TechTrendsRAGPipeline, 
                 validation_threshold: float = 0.7,
                 fairness_threshold: float = 0.6):
        self.pipeline = pipeline
        self.logger = logging.getLogger(__name__)
        self.validation_threshold = validation_threshold
        self.fairness_threshold = fairness_threshold
    
    def calculate_response_similarity(self, generated: str, expected: str) -> float:
        """
        Calculate similarity between generated and expected response
        Simple token-based overlap similarity
        """
        if not expected or not generated:
            return 0.0
        
        # Tokenize and normalize
        gen_tokens = set(generated.lower().split())
        exp_tokens = set(expected.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 
                     'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        gen_tokens -= stop_words
        exp_tokens -= stop_words
        
        if not exp_tokens:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(gen_tokens & exp_tokens)
        union = len(gen_tokens | exp_tokens)
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_single_query(self, question: str, expected_response: str = None) -> Dict[str, Any]:
        """Evaluate a single query"""
        try:
            # Generate response
            result = self.pipeline.query(question)
            
            evaluation = {
                'question': question,
                'generated_response': result['response'],
                'expected_response': expected_response or '',
                'validation_score': result['validation']['overall_score'],
                'fairness_score': result['bias_report'].get('overall_fairness_score', 0.0),
                'num_sources': result['num_sources'],
                'response_time': result['response_time'],
                'success': True,
                'error': None
            }
            
            # Calculate response similarity if expected response provided
            if expected_response:
                similarity = self.calculate_response_similarity(
                    result['response'], 
                    expected_response
                )
                evaluation['response_similarity'] = similarity
            else:
                evaluation['response_similarity'] = None
            
            # Check thresholds
            evaluation['passes_validation'] = evaluation['validation_score'] >= self.validation_threshold
            evaluation['passes_fairness'] = evaluation['fairness_score'] >= self.fairness_threshold
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating query '{question}': {e}")
            return {
                'question': question,
                'success': False,
                'error': str(e),
                'validation_score': 0.0,
                'fairness_score': 0.0,
                'response_similarity': 0.0,
                'passes_validation': False,
                'passes_fairness': False
            }
    
    def evaluate_all(self, csv_path: str) -> Dict[str, Any]:
        """Evaluate all queries from CSV"""
        self.logger.info(f"Loading test data from {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if 'question' not in df.columns:
            raise ValueError("CSV must have 'question' column")
        
        has_expected = 'expected_response' in df.columns
        
        self.logger.info(f"Evaluating {len(df)} queries...")
        
        results = []
        for idx, row in df.iterrows():
            question = row['question']
            expected = row.get('expected_response', '') if has_expected else None
            
            self.logger.info(f"[{idx+1}/{len(df)}] {question[:60]}...")
            
            eval_result = self.evaluate_single_query(question, expected)
            results.append(eval_result)
        
        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(results)
        
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'status': 'FAILED',
                'total_queries': len(results),
                'error': 'All queries failed'
            }
        
        # Calculate averages
        avg_validation = sum(r['validation_score'] for r in successful_results) / len(successful_results)
        avg_fairness = sum(r['fairness_score'] for r in successful_results) / len(successful_results)
        
        # Calculate response similarity if available
        similarities = [r['response_similarity'] for r in successful_results 
                       if r['response_similarity'] is not None]
        avg_similarity = sum(similarities) / len(similarities) if similarities else None
        
        # Pass rates
        validation_passes = sum(1 for r in successful_results if r['passes_validation'])
        fairness_passes = sum(1 for r in successful_results if r['passes_fairness'])
        
        validation_pass_rate = validation_passes / len(successful_results)
        fairness_pass_rate = fairness_passes / len(successful_results)
        
        # Overall status
        passes_thresholds = (
            avg_validation >= self.validation_threshold and
            avg_fairness >= self.fairness_threshold and
            validation_pass_rate >= 0.8 and
            fairness_pass_rate >= 0.8
        )
        
        return {
            'status': 'PASSED' if passes_thresholds else 'FAILED',
            'total_queries': len(results),
            'successful_queries': len(successful_results),
            'failed_queries': len(results) - len(successful_results),
            'avg_validation_score': round(avg_validation, 3),
            'avg_fairness_score': round(avg_fairness, 3),
            'avg_response_similarity': round(avg_similarity, 3) if avg_similarity else None,
            'validation_pass_rate': round(validation_pass_rate, 3),
            'fairness_pass_rate': round(fairness_pass_rate, 3),
            'thresholds': {
                'validation': self.validation_threshold,
                'fairness': self.fairness_threshold
            }
        }
    
    def compare_with_previous(self, current_metrics: Dict[str, Any], 
                             previous_report_path: str) -> bool:
        """Compare current model with previous model"""
        try:
            with open(previous_report_path, 'r') as f:
                previous_results = json.load(f)
            
            current = current_metrics['aggregate_metrics']
            previous = previous_results['aggregate_metrics']
            
            print("\n" + "="*80)
            print("COMPARING WITH PREVIOUS MODEL")
            print("="*80)
            
            curr_val = current['avg_validation_score']
            prev_val = previous['avg_validation_score']
            val_diff = curr_val - prev_val
            
            curr_fair = current['avg_fairness_score']
            prev_fair = previous['avg_fairness_score']
            fair_diff = curr_fair - prev_fair
            
            print(f"\nValidation Score: {prev_val:.3f} → {curr_val:.3f} ({val_diff:+.3f})")
            print(f"Fairness Score:   {prev_fair:.3f} → {curr_fair:.3f} ({fair_diff:+.3f})")
            
            # Decision: must not regress
            is_better = curr_val >= prev_val and curr_fair >= prev_fair
            
            print(f"\nDecision: {'✅ BETTER/EQUAL' if is_better else '❌ REGRESSION'}")
            print("="*80 + "\n")
            
            return is_better
            
        except FileNotFoundError:
            self.logger.info("No previous report found - first run")
            return True
        except Exception as e:
            self.logger.warning(f"Error comparing: {e}, proceeding anyway")
            return True
    
    def save_report(self, results: Dict[str, Any], output_path: str):
        """Save evaluation report"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # Catch any other numpy scalar types
                return obj.item()
            else:
                return obj
        
        # Convert results to serializable format
        serializable_results = convert_to_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved report to {output_path}")
        
        # Print summary (use original results for display)
        agg = results['aggregate_metrics']
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Status: {agg['status']}")
        print(f"Validation Score: {agg['avg_validation_score']:.3f} (threshold: {agg['thresholds']['validation']})")
        print(f"Fairness Score: {agg['avg_fairness_score']:.3f} (threshold: {agg['thresholds']['fairness']})")
        if agg['avg_response_similarity']:
            print(f"Response Similarity: {agg['avg_response_similarity']:.3f}")
        print(f"Pass Rates: Validation {agg['validation_pass_rate']*100:.0f}%, Fairness {agg['fairness_pass_rate']*100:.0f}%")
        print("="*80 + "\n")

def push_to_artifact_registry(pipeline, project_id: str, location: str, 
                              repository: str, version: str, 
                              metrics: Dict[str, Any]) -> str:
    """Push model to GCP Artifact Registry"""
    from deployment.artifact_registry_pusher import ArtifactRegistryPusher
    
    logger.info("Pushing to Artifact Registry...")
    
    # Get sample result for bias report
    sample_result = pipeline.query("What are the latest trends in AI?")
    
    artifact_path = pipeline.push_to_artifact_registry(
        project_id=project_id,
        location=location,
        repository=repository,
        version=version,
        metrics=sample_result['metrics'],
        bias_report=sample_result['bias_report'],
        description=f"Validation: {metrics['avg_validation_score']:.3f}, Fairness: {metrics['avg_fairness_score']:.3f}"
    )
    
    logger.info(f"✅ Pushed to: {artifact_path}")
    return artifact_path


def main():
    parser = argparse.ArgumentParser(description='Simple Model Evaluation')
    
    # Required
    parser.add_argument('--test-data', type=str, required=True, help='CSV with questions and expected responses')
    
    # Thresholds
    parser.add_argument('--validation-threshold', type=float, default=0.7)
    parser.add_argument('--fairness-threshold', type=float, default=0.6)
    
    # Output
    parser.add_argument('--output', type=str, default='evaluation_report.json')
    
    # Comparison
    parser.add_argument('--previous-report', type=str, help='Previous report for comparison')
    
    # Artifact Registry (optional)
    parser.add_argument('--push-to-registry', action='store_true')
    parser.add_argument('--project-id', type=str)
    parser.add_argument('--location', type=str, default='us-central1')
    parser.add_argument('--repository', type=str, default='rag-models')
    parser.add_argument('--version', type=str)
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("SIMPLE MODEL EVALUATION")
    logger.info("="*80)
    
    try:
        # Step 1: Load indexes (assume already present)
        logger.info("Step 1: Loading existing indexes...")
        pipeline = TechTrendsRAGPipeline(enable_tracking=False)
        
        if not pipeline.load_indexes():
            logger.error("Failed to load indexes")
            sys.exit(1)
        
        logger.info("✅ Indexes loaded")
        
        # Step 2: Evaluate all queries
        logger.info("\nStep 2: Generating responses and computing metrics...")
        evaluator = SimpleModelEvaluator(
            pipeline, 
            args.validation_threshold,
            args.fairness_threshold
        )
        
        results = evaluator.evaluate_all(args.test_data)
        
        # Step 3 & 4: Metrics and bias detection (already computed)
        logger.info("✅ Metrics and bias detection completed")
        
        # Save report
        evaluator.save_report(results, args.output)
        
        # Step 5: Check thresholds
        status = results['aggregate_metrics']['status']
        
        if status == 'FAILED':
            logger.error("❌ Model FAILED thresholds")
            sys.exit(1)
        
        logger.info("✅ Model PASSED thresholds")
        
        # Step 6: Compare with previous and push if better
        if args.push_to_registry:
            should_push = True
            
            if args.previous_report:
                logger.info("\nStep 6: Comparing with previous model...")
                should_push = evaluator.compare_with_previous(results, args.previous_report)
            
            if should_push:
                logger.info("\nPushing to Artifact Registry...")
                
                project_id = args.project_id or os.getenv('GCP_PROJECT_ID')
                if not project_id:
                    logger.error("GCP project ID required")
                    sys.exit(1)
                
                artifact_path = push_to_artifact_registry(
                    pipeline,
                    project_id,
                    args.location,
                    args.repository,
                    args.version or f"v{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    results['aggregate_metrics']
                )
                
                # Save artifact path for notification
                with open('artifact_path.txt', 'w') as f:
                    f.write(artifact_path)
                
                logger.info("✅ Model pushed successfully")
            else:
                logger.error("❌ Model did not improve - NOT pushing")
                sys.exit(1)
        
        logger.info("\n✅ EVALUATION COMPLETED SUCCESSFULLY")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()