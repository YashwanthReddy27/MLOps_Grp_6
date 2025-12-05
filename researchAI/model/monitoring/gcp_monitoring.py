"""
Google Cloud Monitoring for RAG Model Performance and Data Drift Detection

This module integrates with Google Cloud Monitoring to:
1. Track model performance metrics over time
2. Detect model decay
3. Detect data distribution shifts
4. Trigger alerts when thresholds are exceeded
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import query
import numpy as np
from collections import deque
import json

logger = logging.getLogger(__name__)


class GCPModelMonitor:
    """
    Monitor RAG model performance using Google Cloud Monitoring
    
    Tracks:
    - Validation scores (quality metrics)
    - Fairness scores
    - Response times
    - Error rates
    - Data distribution statistics
    """
    
    def __init__(
        self,
        project_id: str,
        model_name: str = "rag-model",
        metric_prefix: str = "custom.googleapis.com/rag",
        window_size: int = 100,
        performance_threshold: float = 0.70,
        fairness_threshold: float = 0.60,
        data_drift_threshold: float = 0.15
    ):
        """
        Initialize GCP Model Monitor
        
        Args:
            project_id: GCP project ID
            model_name: Name of the model being monitored
            metric_prefix: Prefix for custom metrics
            window_size: Number of recent predictions to track for drift detection
            performance_threshold: Minimum acceptable validation score
            fairness_threshold: Minimum acceptable fairness score
            data_drift_threshold: Maximum acceptable drift score
        """
        self.project_id = project_id
        self.model_name = model_name
        self.metric_prefix = metric_prefix
        self.window_size = window_size
        
        # Thresholds
        self.performance_threshold = performance_threshold
        self.fairness_threshold = fairness_threshold
        self.data_drift_threshold = data_drift_threshold
        
        # Initialize GCP clients
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"
        
        # Local tracking for drift detection
        self.validation_scores = deque(maxlen=window_size)
        self.fairness_scores = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        
        # Feature statistics for drift detection
        self.baseline_stats = {}
        self.current_stats = {}
        
        logger.info(
            f"Initialized GCP Model Monitor for {model_name} in project {project_id}"
        )
    
    def create_custom_metric_descriptor(self, metric_type: str, 
                                       description: str,
                                       value_type: str = "DOUBLE",
                                       metric_kind: str = "GAUGE"):
        """
        Create a custom metric descriptor in GCP
        
        Args:
            metric_type: Type of metric (e.g., 'validation_score')
            description: Metric description
            value_type: Value type (DOUBLE, INT64, etc.)
            metric_kind: Metric kind (GAUGE, CUMULATIVE, etc.)
        """
        try:
            descriptor = monitoring_v3.MetricDescriptor()
            descriptor.type = f"{self.metric_prefix}/{metric_type}"
            descriptor.metric_kind = getattr(
                monitoring_v3.MetricDescriptor.MetricKind, 
                metric_kind
            )
            descriptor.value_type = getattr(
                monitoring_v3.MetricDescriptor.ValueType, 
                value_type
            )
            descriptor.description = description
            
            # Add labels
            descriptor.labels.append(
                monitoring_v3.LabelDescriptor(
                    key="model_name",
                    value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
                    description="Name of the RAG model"
                )
            )
            
            descriptor = self.client.create_metric_descriptor(
                name=self.project_name,
                metric_descriptor=descriptor
            )
            
            logger.info(f"Created metric descriptor: {descriptor.type}")
            return descriptor
            
        except Exception as e:
            logger.warning(f"Metric descriptor may already exist: {e}")
            return None
    
    def initialize_metrics(self):
        """Create all required metric descriptors"""
        metrics = [
            ("validation_score", "Model validation score (0-1)"),
            ("fairness_score", "Model fairness score (0-1)"),
            ("response_time", "Query response time in seconds"),
            ("error_rate", "Error rate (0-1)"),
            ("num_sources", "Number of sources retrieved"),
            ("data_drift_score", "Data distribution drift score (0-1)"),
            ("model_decay_score", "Model performance decay score (0-1)"),
        ]
        
        for metric_type, description in metrics:
            self.create_custom_metric_descriptor(metric_type, description)
        
        logger.info("Initialized all metric descriptors")
    
    def write_time_series(self, metric_type: str, value: float, 
                         labels: Optional[Dict[str, str]] = None):
        """
        Write a time series data point to GCP Monitoring
        
        Args:
            metric_type: Type of metric
            value: Metric value
            labels: Optional metric labels
        """
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = f"{self.metric_prefix}/{metric_type}"
            
            # Add labels
            if labels is None:
                labels = {}
            labels['model_name'] = self.model_name
            
            for key, val in labels.items():
                series.metric.labels[key] = val
            
            series.resource.type = "global"
            series.resource.labels["project_id"] = self.project_id
            
            # Create data point
            now = datetime.utcnow()
            seconds = int(now.timestamp())
            nanos = int((now.timestamp() - seconds) * 10**9)
            
            interval = monitoring_v3.TimeInterval(
                {"end_time": {"seconds": seconds, "nanos": nanos}}
            )
            
            point = monitoring_v3.Point(
                {"interval": interval, "value": {"double_value": value}}
            )
            
            series.points = [point]
            
            # Write to GCP
            self.client.create_time_series(
                name=self.project_name,
                time_series=[series]
            )
            
            logger.debug(f"Wrote metric {metric_type}={value}")
            
        except Exception as e:
            logger.error(f"Error writing time series: {e}")
    
    def log_query_metrics(self, query_result: Dict[str, Any]):
        """
        Log metrics from a query result
        
        Args:
            query_result: Result from pipeline.query()
        """
        # Extract metrics
        validation_score = query_result.get('validation', {}).get('overall_score', 0.0)
        fairness_score = query_result.get('bias_report', {}).get('overall_fairness_score', 0.0)
        response_time = query_result.get('response_time', 0.0)
        num_sources = query_result.get('num_sources', 0)
        
        # Write to GCP Monitoring
        self.write_time_series("validation_score", validation_score)
        self.write_time_series("fairness_score", fairness_score)
        self.write_time_series("response_time", response_time)
        self.write_time_series("num_sources", float(num_sources))
        
        # Track locally for decay detection
        self.validation_scores.append(validation_score)
        self.fairness_scores.append(fairness_score)
        self.response_times.append(response_time)
        
        # Check for decay
        self.check_model_decay()
    
    def check_model_decay(self) -> Dict[str, Any]:
        """
        Check if model performance has decayed
        
        Returns:
            Dictionary with decay status and metrics
        """
        if len(self.validation_scores) < 10:
            return {"status": "insufficient_data"}
        
        # Calculate recent performance
        recent_validation = np.mean(list(self.validation_scores)[-20:])
        recent_fairness = np.mean(list(self.fairness_scores)[-20:])
        
        # Calculate baseline (older data)
        if len(self.validation_scores) >= 50:
            baseline_validation = np.mean(list(self.validation_scores)[:30])
            baseline_fairness = np.mean(list(self.fairness_scores)[:30])
            
            # Calculate decay
            validation_decay = baseline_validation - recent_validation
            fairness_decay = baseline_fairness - recent_fairness
            
            # Overall decay score (0-1, higher = more decay)
            decay_score = max(0, (validation_decay + fairness_decay) / 2)
        else:
            validation_decay = 0
            fairness_decay = 0
            decay_score = 0
        
        # Write decay metric
        self.write_time_series("model_decay_score", decay_score)
        
        # Check thresholds
        validation_below_threshold = recent_validation < self.performance_threshold
        fairness_below_threshold = recent_fairness < self.fairness_threshold
        
        decay_detected = validation_below_threshold or fairness_below_threshold
        
        result = {
            "status": "decay_detected" if decay_detected else "healthy",
            "recent_validation_score": recent_validation,
            "recent_fairness_score": recent_fairness,
            "validation_decay": validation_decay,
            "fairness_decay": fairness_decay,
            "overall_decay_score": decay_score,
            "validation_below_threshold": validation_below_threshold,
            "fairness_below_threshold": fairness_below_threshold,
            "thresholds": {
                "performance": self.performance_threshold,
                "fairness": self.fairness_threshold
            }
        }
        
        if decay_detected:
            logger.warning(f"Model decay detected: {result}")
        
        return result
    
    def log_data_statistics(self, query: str, retrieved_docs: List[Dict[str, Any]]):
        """
        Log data statistics for drift detection
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
        """
        # Extract features for drift detection
        features = self._extract_features(query, retrieved_docs)
        
        # Update current statistics
        for feature_name, value in features.items():
            if feature_name not in self.current_stats:
                self.current_stats[feature_name] = []
            self.current_stats[feature_name].append(value)
            
            # Keep only recent window
            if len(self.current_stats[feature_name]) > self.window_size:
                self.current_stats[feature_name] = self.current_stats[feature_name][-self.window_size:]
    
    def _extract_features(self, query: str, 
                         retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract features for drift detection
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Query features
        features['query_length'] = len(query.split())
        features['query_char_length'] = len(query)
        
        # Retrieval features
        if retrieved_docs:
            scores = [doc.get('score', 0.0) for doc in retrieved_docs]
            features['avg_retrieval_score'] = np.mean(scores)
            features['max_retrieval_score'] = np.max(scores)
            features['min_retrieval_score'] = np.min(scores)
            features['score_std'] = np.std(scores)
            
            # Source diversity
            sources = set()
            categories = set()
            for doc in retrieved_docs:
                metadata = doc.get('metadata', {})
                source = metadata.get('source_name') or metadata.get('arxiv_id', 'unknown')
                sources.add(source)
                categories.update(metadata.get('categories', []))
            
            features['num_unique_sources'] = len(sources)
            features['num_unique_categories'] = len(categories)
            features['source_diversity_ratio'] = len(sources) / len(retrieved_docs)
        
        return features
    
    def set_baseline_statistics(self):
        """
        Set current statistics as baseline for drift detection
        
        Should be called after initial model deployment with representative data
        """
        self.baseline_stats = {
            feature: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for feature, values in self.current_stats.items()
            if len(values) >= 20
        }
        
        logger.info(f"Set baseline statistics for {len(self.baseline_stats)} features")
        
        # Save baseline
        self._save_baseline()
    
    def detect_data_drift(self) -> Dict[str, Any]:
        """
        Detect data distribution drift using statistical tests
        
        Returns:
            Dictionary with drift detection results
        """
        if not self.baseline_stats:
            return {"status": "no_baseline"}
        
        drift_scores = {}
        drifted_features = []
        
        for feature, baseline in self.baseline_stats.items():
            if feature not in self.current_stats or len(self.current_stats[feature]) < 20:
                continue
            
            current_values = self.current_stats[feature][-50:]  # Recent window
            
            # Calculate statistics
            current_mean = np.mean(current_values)
            current_std = np.std(current_values)
            
            # Z-score for mean shift
            if baseline['std'] > 0:
                z_score = abs(current_mean - baseline['mean']) / baseline['std']
            else:
                z_score = 0
            
            # Population Stability Index (PSI)
            psi = self._calculate_psi(baseline, current_values)
            
            # Combined drift score
            drift_score = min(1.0, (z_score / 3 + psi) / 2)
            drift_scores[feature] = drift_score
            
            # Check threshold
            if drift_score > self.data_drift_threshold:
                drifted_features.append({
                    'feature': feature,
                    'drift_score': drift_score,
                    'z_score': z_score,
                    'psi': psi,
                    'baseline_mean': baseline['mean'],
                    'current_mean': current_mean
                })
        
        # Overall drift score
        overall_drift = np.mean(list(drift_scores.values())) if drift_scores else 0
        
        # Write to GCP
        self.write_time_series("data_drift_score", overall_drift)
        
        result = {
            "status": "drift_detected" if drifted_features else "healthy",
            "overall_drift_score": overall_drift,
            "drift_threshold": self.data_drift_threshold,
            "num_features_checked": len(drift_scores),
            "num_drifted_features": len(drifted_features),
            "drifted_features": drifted_features,
            "all_drift_scores": drift_scores
        }
        
        if drifted_features:
            logger.warning(f"Data drift detected in {len(drifted_features)} features: {result}")
        
        return result
    
    def _calculate_psi(self, baseline: Dict[str, float], 
                      current_values: List[float]) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            baseline: Baseline statistics
            current_values: Current values
            
        Returns:
            PSI score
        """
        try:
            # Create bins based on baseline
            num_bins = 10
            baseline_min = baseline['min']
            baseline_max = baseline['max']
            
            if baseline_max <= baseline_min:
                return 0.0
            
            bins = np.linspace(baseline_min, baseline_max, num_bins + 1)
            
            # Expected proportions (uniform for simplicity)
            expected_props = np.ones(num_bins) / num_bins
            
            # Actual proportions from current data
            actual_counts, _ = np.histogram(current_values, bins=bins)
            actual_props = actual_counts / len(current_values)
            
            # Calculate PSI
            psi = 0
            for expected, actual in zip(expected_props, actual_props):
                if actual > 0 and expected > 0:
                    psi += (actual - expected) * np.log(actual / expected)
            
            return psi
            
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0
    
    def should_trigger_retraining(self) -> Dict[str, Any]:
        """
        Determine if model retraining should be triggered
        
        Returns:
            Dictionary with decision and reasoning
        """
        # Check model decay
        decay_result = self.check_model_decay()
        
        # Check data drift
        drift_result = self.detect_data_drift()
        
        # Decision logic
        trigger_retraining = False
        reasons = []
        
        # Check decay
        if decay_result.get('status') == 'decay_detected':
            trigger_retraining = True
            if decay_result.get('validation_below_threshold'):
                reasons.append(
                    f"Validation score ({decay_result['recent_validation_score']:.3f}) "
                    f"below threshold ({self.performance_threshold})"
                )
            if decay_result.get('fairness_below_threshold'):
                reasons.append(
                    f"Fairness score ({decay_result['recent_fairness_score']:.3f}) "
                    f"below threshold ({self.fairness_threshold})"
                )
        
        # Check drift
        if drift_result.get('status') == 'drift_detected':
            trigger_retraining = True
            reasons.append(
                f"Data drift detected in {drift_result['num_drifted_features']} features "
                f"(overall drift: {drift_result['overall_drift_score']:.3f})"
            )
        
        result = {
            "should_retrain": trigger_retraining,
            "reasons": reasons,
            "decay_result": decay_result,
            "drift_result": drift_result,
            "timestamp": datetime.now().isoformat()
        }
        
        if trigger_retraining:
            logger.warning(f"Retraining triggered: {reasons}")
        
        return result
    
    def _save_baseline(self, filepath: str = "./baseline_stats.json"):
        """Save baseline statistics to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.baseline_stats, f, indent=2)
            logger.info(f"Saved baseline statistics to {filepath}")
        except Exception as e:
            logger.error(f"Error saving baseline: {e}")
    
    def _load_baseline(self, filepath: str = "./baseline_stats.json"):
        """Load baseline statistics from file"""
        try:
            with open(filepath, 'r') as f:
                self.baseline_stats = json.load(f)
            logger.info(f"Loaded baseline statistics from {filepath}")
        except Exception as e:
            logger.warning(f"Error loading baseline: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary
        
        Returns:
            Summary of all monitoring metrics
        """
        decay_result = self.check_model_decay()
        drift_result = self.detect_data_drift()
        retraining_decision = self.should_trigger_retraining()
        
        return {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "thresholds": {
                "performance": self.performance_threshold,
                "fairness": self.fairness_threshold,
                "data_drift": self.data_drift_threshold
            },
            "current_metrics": {
                "recent_validation_score": decay_result.get('recent_validation_score'),
                "recent_fairness_score": decay_result.get('recent_fairness_score'),
                "overall_decay_score": decay_result.get('overall_decay_score'),
                "overall_drift_score": drift_result.get('overall_drift_score')
            },
            "model_decay": decay_result,
            "data_drift": drift_result,
            "retraining_decision": retraining_decision,
            "data_points_tracked": len(self.validation_scores)
        }