"""
Example script demonstrating GCP Monitoring setup and usage

This script shows:
1. How to initialize monitoring
2. How to set baseline statistics
3. How to check for model decay and data drift
4. How to trigger retraining based on monitoring
"""

import os
import json
import time
import argparse
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import TechTrendsRAGPipeline
from monitoring_integration import add_monitoring_to_pipeline
from utils.logger import setup_logging


def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "="*80)
    if title:
        print(f"{title}")
        print("="*80)


def run_initial_queries(pipeline, num_queries: int = 50):
    """
    Run initial queries to establish baseline
    
    Args:
        pipeline: RAG pipeline with monitoring
        num_queries: Number of queries to run
    """
    print_separator("RUNNING INITIAL QUERIES FOR BASELINE")
    
    # Sample queries covering different topics
    queries = [
        "What are the latest developments in large language models?",
        "How is reinforcement learning being used in robotics?",
        "What are the recent advances in quantum computing?",
        "Explain the latest cybersecurity threats and defenses",
        "What is the current state of edge computing?",
        "How is blockchain being used in supply chain management?",
        "What are the latest trends in cloud computing?",
        "Explain recent developments in computer vision",
        "What are the challenges in deploying AI at scale?",
        "How is federated learning changing machine learning?",
    ]
    
    for i in range(num_queries):
        query = queries[i % len(queries)]
        print(f"\n[{i+1}/{num_queries}] Query: {query[:50]}...")
        
        try:
            result = pipeline.query(query)
            print(f"  ✓ Validation: {result['validation']['overall_score']:.3f}, "
                  f"Fairness: {result['bias_report']['overall_fairness_score']:.3f}")
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n✓ Initial queries completed")


def set_baseline(pipeline):
    """Set baseline statistics"""
    print_separator("SETTING BASELINE STATISTICS")
    
    if not hasattr(pipeline, 'monitoring') or pipeline.monitoring is None:
        print("✗ Monitoring not enabled")
        return False
    
    try:
        pipeline.monitoring.set_baseline()
        print("✓ Baseline statistics set successfully")
        
        # Display baseline info
        monitor = pipeline.monitoring.monitor
        print(f"\nBaseline features tracked: {len(monitor.baseline_stats)}")
        
        for feature, stats in list(monitor.baseline_stats.items())[:5]:
            print(f"  {feature}:")
            print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error setting baseline: {e}")
        return False


def check_model_health(pipeline):
    """Check current model health"""
    print_separator("CHECKING MODEL HEALTH")
    
    if not hasattr(pipeline, 'monitoring') or pipeline.monitoring is None:
        print("✗ Monitoring not enabled")
        return
    
    try:
        health = pipeline.monitoring.check_model_health()
        
        print(f"Model: {health['model_name']}")
        print(f"Timestamp: {health['timestamp']}")
        
        print("\nCurrent Metrics:")
        metrics = health['current_metrics']
        print(f"  Validation Score: {metrics.get('recent_validation_score', 0):.3f}")
        print(f"  Fairness Score: {metrics.get('recent_fairness_score', 0):.3f}")
        print(f"  Decay Score: {metrics.get('overall_decay_score', 0):.3f}")
        print(f"  Drift Score: {metrics.get('overall_drift_score', 0):.3f}")
        
        print("\nThresholds:")
        thresholds = health['thresholds']
        print(f"  Performance: {thresholds['performance']}")
        print(f"  Fairness: {thresholds['fairness']}")
        print(f"  Data Drift: {thresholds['data_drift']}")
        
        # Model Decay
        print("\nModel Decay Status:")
        decay = health['model_decay']
        print(f"  Status: {decay['status']}")
        
        if decay.get('status') != 'insufficient_data':
            print(f"  Validation Below Threshold: {decay.get('validation_below_threshold', False)}")
            print(f"  Fairness Below Threshold: {decay.get('fairness_below_threshold', False)}")
        
        # Data Drift
        print("\nData Drift Status:")
        drift = health['data_drift']
        print(f"  Status: {drift['status']}")
        
        if drift.get('status') != 'no_baseline':
            print(f"  Features Checked: {drift.get('num_features_checked', 0)}")
            print(f"  Drifted Features: {drift.get('num_drifted_features', 0)}")
            
            drifted = drift.get('drifted_features', [])
            if drifted:
                print("\n  Drifted Features:")
                for feature_info in drifted[:3]:  # Show first 3
                    print(f"    - {feature_info['feature']}: "
                          f"drift={feature_info['drift_score']:.3f}")
        
    except Exception as e:
        print(f"✗ Error checking health: {e}")


def check_retraining_trigger(pipeline):
    """Check if retraining should be triggered"""
    print_separator("CHECKING RETRAINING TRIGGER")
    
    if not hasattr(pipeline, 'monitoring') or pipeline.monitoring is None:
        print("✗ Monitoring not enabled")
        return False
    
    try:
        decision = pipeline.monitoring.should_trigger_retraining()
        
        print(f"Should Retrain: {decision['should_retrain']}")
        print(f"Timestamp: {decision['timestamp']}")
        
        if decision['should_retrain']:
            print("\n⚠️  RETRAINING RECOMMENDED")
            print("\nReasons:")
            for reason in decision['reasons']:
                print(f"  - {reason}")
        else:
            print("\n✓ Model is healthy, no retraining needed")
        
        return decision['should_retrain']
        
    except Exception as e:
        print(f"✗ Error checking retraining trigger: {e}")
        return False


def simulate_model_decay(pipeline, num_queries: int = 20):
    """
    Simulate model decay by running queries that might perform poorly
    
    Args:
        pipeline: RAG pipeline
        num_queries: Number of queries to run
    """
    print_separator("SIMULATING MODEL DECAY (Running Low-Quality Queries)")
    
    # Queries that might be harder or out-of-distribution
    difficult_queries = [
        "Tell me about extremely obscure research from 1995",
        "What are the quantum mechanical principles of consciousness?",
        "Explain the relationship between dark matter and AI",
        "What are the philosophical implications of AGI?",
        "How does quantum entanglement affect neural networks?",
    ]
    
    for i in range(num_queries):
        query = difficult_queries[i % len(difficult_queries)]
        print(f"\n[{i+1}/{num_queries}] Query: {query[:50]}...")
        
        try:
            result = pipeline.query(query)
            print(f"  Validation: {result['validation']['overall_score']:.3f}, "
                  f"Fairness: {result['bias_report']['overall_fairness_score']:.3f}")
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n⚠️  Decay simulation completed")


def main():
    """Main monitoring demonstration"""
    parser = argparse.ArgumentParser(
        description='GCP Monitoring Setup and Demonstration'
    )
    
    parser.add_argument(
        '--project-id',
        type=str,
        help='GCP project ID (default: from GCP_PROJECT_ID env var)'
    )
    
    parser.add_argument(
        '--skip-baseline-setup',
        action='store_true',
        help='Skip baseline setup (use existing baseline)'
    )
    
    parser.add_argument(
        '--simulate-decay',
        action='store_true',
        help='Simulate model decay by running difficult queries'
    )
    
    parser.add_argument(
        '--num-baseline-queries',
        type=int,
        default=50,
        help='Number of queries for baseline setup (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print_separator("GCP MONITORING SETUP AND DEMONSTRATION")
    
    # Get project ID
    project_id = args.project_id or os.getenv('GCP_PROJECT_ID')
    
    if not project_id:
        print("✗ Error: GCP_PROJECT_ID not set")
        print("  Set it using: export GCP_PROJECT_ID=your-project-id")
        print("  Or pass --project-id your-project-id")
        return
    
    print(f"Using GCP Project: {project_id}")
    
    # Initialize pipeline
    print("\nStep 1: Initializing RAG Pipeline...")
    pipeline = TechTrendsRAGPipeline(enable_tracking=True)
    
    # Load indexes
    print("Step 2: Loading indexes...")
    if not pipeline.load_indexes():
        print("✗ Failed to load indexes")
        print("  Please build indexes first using example.py")
        return
    
    print("✓ Indexes loaded")
    
    # Add monitoring
    print("\nStep 3: Enabling GCP Monitoring...")
    monitoring = add_monitoring_to_pipeline(pipeline, project_id=project_id)
    
    if not monitoring.enable_monitoring:
        print("✗ Failed to enable monitoring")
        return
    
    print("✓ Monitoring enabled")
    
    # Setup baseline (if needed)
    if not args.skip_baseline_setup:
        print("\nStep 4: Setting up baseline statistics...")
        print(f"  Running {args.num_baseline_queries} queries to establish baseline...")
        
        run_initial_queries(pipeline, args.num_baseline_queries)
        
        # Set baseline
        if not set_baseline(pipeline):
            print("✗ Failed to set baseline")
            return
    else:
        print("\nStep 4: Skipping baseline setup (using existing)")
        
        # Try to load existing baseline
        baseline_path = Path("./baseline_stats.json")
        if not baseline_path.exists():
            print("✗ No existing baseline found")
            print("  Run without --skip-baseline-setup to create one")
            return
    
    # Check initial health
    print("\nStep 5: Checking initial model health...")
    check_model_health(pipeline)
    
    # Check retraining trigger
    print("\nStep 6: Checking retraining trigger...")
    check_retraining_trigger(pipeline)
    
    # Simulate decay if requested
    if args.simulate_decay:
        print("\nStep 7: Simulating model decay...")
        simulate_model_decay(pipeline, num_queries=30)
        
        # Check health after decay
        print("\nStep 8: Checking health after simulated decay...")
        check_model_health(pipeline)
        
        # Check retraining trigger after decay
        print("\nStep 9: Checking retraining trigger after decay...")
        should_retrain = check_retraining_trigger(pipeline)
        
        if should_retrain:
            print("\n⚠️  MODEL RETRAINING TRIGGERED")
            print("\nNext steps:")
            print("  1. Pull latest data from data pipeline")
            print("  2. Run model retraining pipeline")
            print("  3. Evaluate new model")
            print("  4. Deploy if performance improves")
    
    print_separator("MONITORING SETUP COMPLETE")
    
    print("\nMonitoring is now active!")
    print("\nYou can:")
    print("  1. Check health: GET /api/monitoring/health")
    print("  2. Check retraining: GET /api/monitoring/retraining-check")
    print("  3. Get metrics: GET /api/monitoring/metrics")
    print("  4. Set baseline: POST /api/monitoring/set-baseline")
    
    print("\nView metrics in GCP Console:")
    print(f"  https://console.cloud.google.com/monitoring/dashboards?project={project_id}")


if __name__ == "__main__":
    main()