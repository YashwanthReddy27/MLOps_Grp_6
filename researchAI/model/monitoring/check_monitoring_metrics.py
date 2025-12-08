"""
Script to fetch monitoring metrics from GCP Cloud Monitoring Dashboard
and determine if model retraining is needed based on data drift and model decay thresholds.

Compatible with:
- Python 3.11
- google-cloud-monitoring>=2.21.0,<2.26.0
"""

import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from google.cloud import monitoring_v3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringMetricsFetcher:
    """Fetch and analyze metrics from GCP Cloud Monitoring"""
    
    def __init__(self, project_id: str, model_name: str = "rag-model", lookback_hours: int = 24):
        """
        Initialize metrics fetcher
        
        Args:
            project_id: GCP project ID
            model_name: Model name (must match label in gcp_monitoring.py)
            lookback_hours: How many hours of data to analyze
        """
        self.project_id = project_id
        self.model_name = model_name
        self.project_name = f"projects/{project_id}"
        self.client = monitoring_v3.MetricServiceClient()
        self.lookback_hours = lookback_hours
        
        # Thresholds (must match gcp_monitoring.py)
        self.DATA_DRIFT_THRESHOLD = 0.15
        self.MODEL_DECAY_THRESHOLD = 0.7  # Below this is problematic
        
        logger.info(f"Initialized metrics fetcher for project: {project_id}, model: {model_name}")
    
    def _fetch_metric_data(self, metric_type: str, model_name: str = "rag-model") -> Optional[float]:
        """
        Fetch recent data for a specific metric
        
        Args:
            metric_type: The metric type to fetch (e.g., 'data_drift_score')
            model_name: Model name label to filter by
            
        Returns:
            Latest metric value or None
        """
        try:
            # Time range: last N hours
            now = datetime.utcnow()
            start_time = now - timedelta(hours=self.lookback_hours)
            
            # Build the metric filter with model_name label
            # Format matches gcp_monitoring.py: metric.type and metric.labels.model_name
            metric_filter = (
                f'metric.type = "custom.googleapis.com/rag/{metric_type}" '
                f'AND metric.labels.model_name = "{model_name}"'
            )
            
            # Create interval
            interval = monitoring_v3.TimeInterval(
                {
                    "end_time": {"seconds": int(now.timestamp())},
                    "start_time": {"seconds": int(start_time.timestamp())},
                }
            )
            
            # Aggregate: get the mean of recent values
            aggregation = monitoring_v3.Aggregation(
                {
                    "alignment_period": {"seconds": 3600},  # 1 hour buckets
                    "per_series_aligner": monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                    "cross_series_reducer": monitoring_v3.Aggregation.Reducer.REDUCE_MEAN,
                }
            )
            
            # List time series
            results = self.client.list_time_series(
                request={
                    "name": self.project_name,
                    "filter": metric_filter,
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                    "aggregation": aggregation,
                }
            )
            
            # Extract the most recent value
            latest_value = None
            latest_time = None
            
            for result in results:
                for point in result.points:
                    point_time = point.interval.end_time.timestamp()
                    value = point.value.double_value
                    
                    if latest_time is None or point_time > latest_time:
                        latest_time = point_time
                        latest_value = value
            
            if latest_value is not None:
                logger.info(f"✓ Fetched {metric_type}: {latest_value:.4f}")
            else:
                logger.warning(f"⚠️  No data found for {metric_type}")
            
            return latest_value
            
        except Exception as e:
            logger.error(f"Error fetching {metric_type}: {e}")
            return None
    
    def check_data_drift(self) -> Dict[str, Any]:
        """
        Check current data drift score
        
        Returns:
            Dictionary with drift status and value
        """
        logger.info("Checking data drift...")
        
        drift_score = self._fetch_metric_data("data_drift_score", self.model_name)
        
        if drift_score is None:
            return {
                "status": "UNKNOWN",
                "drift_score": None,
                "threshold": self.DATA_DRIFT_THRESHOLD,
                "drift_detected": False,
                "message": "No data drift metrics available"
            }
        
        drift_detected = drift_score > self.DATA_DRIFT_THRESHOLD
        
        result = {
            "status": "DRIFT_DETECTED" if drift_detected else "HEALTHY",
            "drift_score": drift_score,
            "threshold": self.DATA_DRIFT_THRESHOLD,
            "drift_detected": drift_detected,
            "message": f"Data drift score: {drift_score:.4f} (threshold: {self.DATA_DRIFT_THRESHOLD})"
        }
        
        if drift_detected:
            logger.warning(f"⚠️  DATA DRIFT DETECTED: {drift_score:.4f} > {self.DATA_DRIFT_THRESHOLD}")
        else:
            logger.info(f"✓ Data drift within acceptable range: {drift_score:.4f}")
        
        return result
    
    def check_model_decay(self) -> Dict[str, Any]:
        """
        Check current model decay score
        
        Returns:
            Dictionary with decay status and value
        """
        logger.info("Checking model decay...")
        
        decay_score = self._fetch_metric_data("model_decay_score", self.model_name)
        
        if decay_score is None:
            return {
                "status": "UNKNOWN",
                "decay_score": None,
                "threshold": self.MODEL_DECAY_THRESHOLD,
                "decay_detected": False,
                "message": "No model decay metrics available"
            }
        
        # Decay score < threshold means performance has dropped
        decay_detected = decay_score < self.MODEL_DECAY_THRESHOLD
        
        result = {
            "status": "DECAY_DETECTED" if decay_detected else "HEALTHY",
            "decay_score": decay_score,
            "threshold": self.MODEL_DECAY_THRESHOLD,
            "decay_detected": decay_detected,
            "message": f"Model decay score: {decay_score:.4f} (threshold: {self.MODEL_DECAY_THRESHOLD})"
        }
        
        if decay_detected:
            logger.warning(f"⚠️  MODEL DECAY DETECTED: {decay_score:.4f} < {self.MODEL_DECAY_THRESHOLD}")
        else:
            logger.info(f"✓ Model performance healthy: {decay_score:.4f}")
        
        return result
    
    def should_trigger_retraining(self) -> Dict[str, Any]:
        """
        Main decision function: Should we retrain?
        
        Returns:
            Dictionary with decision and supporting data
        """
        logger.info("=" * 80)
        logger.info("CHECKING MONITORING METRICS FOR RETRAINING DECISION")
        logger.info("=" * 80)
        
        # Check both drift and decay
        drift_result = self.check_data_drift()
        decay_result = self.check_model_decay()
        
        # Decision: retrain if EITHER drift or decay detected
        should_retrain = drift_result["drift_detected"] or decay_result["decay_detected"]
        
        reasons = []
        if drift_result["drift_detected"]:
            reasons.append("Data drift exceeds threshold")
        if decay_result["decay_detected"]:
            reasons.append("Model performance decay detected")
        
        result = {
            "should_retrain": should_retrain,
            "timestamp": datetime.utcnow().isoformat(),
            "lookback_hours": self.lookback_hours,
            "data_drift": drift_result,
            "model_decay": decay_result,
            "reasons": reasons,
            "decision": "RETRAIN_REQUIRED" if should_retrain else "NO_ACTION_NEEDED"
        }
        
        logger.info("\n" + "=" * 80)
        if should_retrain:
            logger.warning("🔴 RETRAINING REQUIRED")
            for reason in reasons:
                logger.warning(f"   - {reason}")
        else:
            logger.info("✅ NO RETRAINING NEEDED - All metrics healthy")
        logger.info("=" * 80)
        
        return result


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check GCP monitoring metrics for retraining decision")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--model-name", default="rag-model", help="Model name (must match label in gcp_monitoring.py)")
    parser.add_argument("--lookback-hours", type=int, default=24, help="Hours of data to analyze")
    parser.add_argument("--output", default="monitoring_decision.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    try:
        fetcher = MonitoringMetricsFetcher(
            project_id=args.project_id,
            model_name=args.model_name,
            lookback_hours=args.lookback_hours
        )
        
        result = fetcher.should_trigger_retraining()
        
        # Save result to file
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {args.output}")
        
        # Exit code: 0 if no retraining needed, 1 if retraining required
        exit_code = 1 if result["should_retrain"] else 0
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()