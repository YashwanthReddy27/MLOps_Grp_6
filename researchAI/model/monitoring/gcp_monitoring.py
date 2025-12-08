"""
Monitoring for the RAG pipeline: model decay + data drift (with Evidently).

Compatible with:
- Python 3.11
- google-cloud-monitoring>=2.21.0,<2.26.0

Public API:

    monitor = HybridMonitor(...)
    monitor.log_query(query_result: dict)
    monitor.check_health() -> dict
    monitor.should_trigger_retraining() -> dict
    monitor.set_baseline() -> None
    monitor.get_status() -> dict
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from google.cloud.monitoring_v3 import MetricServiceClient
    from google.cloud.monitoring_v3.types import TimeSeries, Point, TimeInterval
    import google.protobuf.timestamp_pb2 as timestamp_pb2
    from google.api import metric_pb2 as ga_metric
    from google.api import label_pb2 as ga_label
    _GCP_MONITORING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Google Cloud Monitoring not available: {e}")
    _GCP_MONITORING_AVAILABLE = False
    MetricServiceClient = None

logger = logging.getLogger(__name__)

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    _EVIDENTLY_AVAILABLE = True
except Exception:
    _EVIDENTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Low-level GCP monitor
# ---------------------------------------------------------------------------

@dataclass
class GCPModelMonitor:
    project_id: str
    model_name: str = "rag-model"
    metric_prefix: str = "custom.googleapis.com/rag"
    window_size: int = 200

    performance_threshold: float = 0.7
    fairness_threshold: float = 0.6
    data_drift_threshold: float = 0.15

    client: Any = field(init=False, default=None)
    project_name: str = field(init=False)

    validation_scores: List[float] = field(default_factory=list, init=False)
    fairness_scores: List[float] = field(default_factory=list, init=False)

    # Data drift storage
    current_batch: List[Dict[str, Any]] = field(default_factory=list, init=False)
    reference_df: Optional[pd.DataFrame] = field(default=None, init=False)
    
    # Rate limiting for GCP metric writes
    last_write_time: Dict[str, datetime] = field(default_factory=dict, init=False)
    min_write_interval_seconds: int = 60  # Minimum 60 seconds between writes

    def __post_init__(self) -> None:
        if not _GCP_MONITORING_AVAILABLE:
            raise ImportError(
                "google-cloud-monitoring is not installed or not properly configured. "
                "Install with: pip install google-cloud-monitoring>=2.21.0,<2.26.0"
            )
        
        try:
            self.client = MetricServiceClient()
            self.project_name = f"projects/{self.project_id}"
            logger.info(
                "Initialized GCPModelMonitor(project=%s, model=%s)",
                self.project_id,
                self.model_name,
            )
            self._ensure_metric_descriptors()
        except Exception as e:
            logger.error(f"Failed to initialize MetricServiceClient: {e}")
            logger.error("Ensure GOOGLE_APPLICATION_CREDENTIALS is set or running on GCP")
            raise

    # ----------------- public API -----------------

    def log_query_metrics(self, query_result: Dict[str, Any]) -> None:
        """
        Ingest a query result and update:
        - validation/fairness scores (decay)
        - feature batch (drift)
        - All RAG metrics to GCP Cloud Monitoring
        """
        now = datetime.now(timezone.utc)
        metrics = query_result.get("metrics", {}) or {}

        # Try to get validation score from multiple possible locations
        validation_score = (
            query_result.get("validation", {}).get("overall_score")
            or query_result.get("validation_score")
            or metrics.get("overall_score")
            or metrics.get("validation_score")
        )
        
        # Try to get fairness score from multiple possible locations
        fairness_score = (
            query_result.get("bias_report", {}).get("overall_fairness_score")
            or query_result.get("fairness_score")
            or metrics.get("fairness_score")
        )

        if validation_score is not None:
            self._append_window(self.validation_scores, float(validation_score))
            self.write_time_series("validation_score", float(validation_score), now)
            
        if fairness_score is not None:
            self._append_window(self.fairness_scores, float(fairness_score))
            self.write_time_series("fairness_score", float(fairness_score), now)

        # Push retrieval metrics to GCP
        retrieval_metrics = metrics.get("retrieval_metrics", {})
        if retrieval_metrics:
            self._log_retrieval_metrics(retrieval_metrics, now)
        
        # Push generation metrics to GCP
        generation_metrics = metrics.get("generation_metrics", {})
        if generation_metrics:
            self._log_generation_metrics(generation_metrics, now)
        
        # Push end-to-end metrics
        if "response_time_seconds" in metrics:
            self.write_time_series("response_time", float(metrics["response_time_seconds"]), now)
        if "tokens_per_second" in metrics:
            self.write_time_series("tokens_per_second", float(metrics["tokens_per_second"]), now)
        if "query_length" in metrics:
            self.write_time_series("query_length", float(metrics["query_length"]), now)
        
        # Calculate and log model decay score
        decay_info = self._check_model_decay()
        decay_score = decay_info.get("overall_decay_score")
        if decay_score is not None:
            self.write_time_series("model_decay_score", float(decay_score), now)

        # Log data features for drift detection
        features = self._extract_features_from_query(query_result)
        if features:
            self._log_data_features(features)
        
        # Calculate and log drift score periodically
        # Only compute drift if we have enough data points
        if len(self.current_batch) >= 10:
            try:
                drift_result = self._detect_data_drift()
                drift_score = drift_result.get("overall_drift_score")
                if drift_score is not None:
                    # Note: write_time_series has built-in rate limiting
                    self.write_time_series("data_drift_score", float(drift_score), now)
            except Exception as e:
                logger.debug(f"Failed to calculate drift score: {e}")

    def _log_retrieval_metrics(self, retrieval_metrics: Dict[str, Any], timestamp: datetime) -> None:
        """Push retrieval metrics to GCP Cloud Monitoring"""
        metric_mappings = {
            "num_retrieved": "retrieval_num_docs",
            "avg_score": "retrieval_avg_score",
            "max_score": "retrieval_max_score",
            "min_score": "retrieval_min_score",
            "score_std": "retrieval_score_std",
            "source_diversity": "retrieval_source_diversity",
            "category_diversity": "retrieval_category_diversity",
            "source_entropy": "retrieval_source_entropy",
            "precision": "retrieval_precision",
            "recall": "retrieval_recall",
            "f1_score": "retrieval_f1_score",
        }
        
        for metric_key, gcp_metric_name in metric_mappings.items():
            value = retrieval_metrics.get(metric_key)
            if value is not None:
                try:
                    self.write_time_series(gcp_metric_name, float(value), timestamp)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not log {gcp_metric_name}: {e}")

    def _log_generation_metrics(self, generation_metrics: Dict[str, Any], timestamp: datetime) -> None:
        """Push generation metrics to GCP Cloud Monitoring"""
        metric_mappings = {
            "response_length": "generation_response_length",
            "num_sentences": "generation_num_sentences",
            "avg_sentence_length": "generation_avg_sentence_length",
            "num_citations": "generation_num_citations",
            "unique_citations": "generation_unique_citations",
            "similarity_score": "generation_similarity_score",
        }
        
        for metric_key, gcp_metric_name in metric_mappings.items():
            value = generation_metrics.get(metric_key)
            if value is not None:
                try:
                    self.write_time_series(gcp_metric_name, float(value), timestamp)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not log {gcp_metric_name}: {e}")

    def _extract_features_from_query(self, query_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract numerical features from query result for drift detection"""
        features = {}
        
        # Extract from metrics if available
        metrics = query_result.get("metrics", {})
        if metrics:
            # Retrieval features
            retrieval = metrics.get("retrieval_metrics", {})
            if retrieval:
                for key in ["avg_score", "max_score", "min_score", "score_std", 
                           "source_diversity", "category_diversity", "source_entropy"]:
                    if key in retrieval:
                        features[f"retrieval_{key}"] = retrieval[key]
            
            # Generation features
            generation = metrics.get("generation_metrics", {})
            if generation:
                for key in ["response_length", "num_sentences", "avg_sentence_length",
                           "num_citations", "unique_citations"]:
                    if key in generation:
                        features[f"generation_{key}"] = generation[key]
            
            # Top-level metrics
            for key in ["response_time_seconds", "tokens_per_second", "query_length"]:
                if key in metrics:
                    features[key] = metrics[key]
        
        # Extract validation and fairness scores
        validation = query_result.get("validation", {})
        if validation and "overall_score" in validation:
            features["validation_score"] = validation["overall_score"]
        
        bias_report = query_result.get("bias_report", {})
        if bias_report and "overall_fairness_score" in bias_report:
            features["fairness_score"] = bias_report["overall_fairness_score"]
        
        # Return None if no features extracted
        return features if features else None

    def get_monitoring_summary(self) -> Dict[str, Any]:
        decay = self._check_model_decay()
        drift = self._detect_data_drift()

        return {
            "model_name": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thresholds": {
                "performance": self.performance_threshold,
                "fairness": self.fairness_threshold,
                "data_drift": self.data_drift_threshold,
            },
            "current_metrics": {
                "recent_validation_score": decay.get("recent_validation_score"),
                "recent_fairness_score": decay.get("recent_fairness_score"),
                "overall_decay_score": decay.get("overall_decay_score"),
                "overall_drift_score": drift.get("overall_drift_score"),
            },
            "model_decay": decay,
            "data_drift": drift,
        }

    def should_trigger_retraining(self) -> Dict[str, Any]:
        decay = self._check_model_decay()
        drift = self._detect_data_drift()

        reasons: List[str] = []
        should_retrain = False

        recent_val = decay.get("recent_validation_score")
        recent_fairness = decay.get("recent_fairness_score")
        drift_score = drift.get("overall_drift_score")

        if recent_val is not None and recent_val < self.performance_threshold:
            should_retrain = True
            reasons.append(
                f"Validation score {recent_val:.3f} < {self.performance_threshold:.3f}"
            )

        if recent_fairness is not None and recent_fairness < self.fairness_threshold:
            should_retrain = True
            reasons.append(
                f"Fairness score {recent_fairness:.3f} < {self.fairness_threshold:.3f}"
            )

        if drift_score is not None and drift_score > self.data_drift_threshold:
            should_retrain = True
            reasons.append(
                f"Data drift score {drift_score:.3f} > {self.data_drift_threshold:.3f}"
            )

        return {
            "should_retrain": should_retrain,
            "reasons": reasons or ["Within thresholds"],
            "decay": decay,
            "drift": drift,
        }

    def set_evidently_baseline(self) -> None:
        if not self.current_batch:
            logger.warning("No data to build Evidently baseline")
            return
        if not _EVIDENTLY_AVAILABLE:
            logger.warning(
                "Evidently not installed; cannot set baseline. "
                "Install with `pip install evidently`."
            )
            return
        self.reference_df = pd.DataFrame(self.current_batch)
        logger.info("Evidently baseline set with %d rows", len(self.reference_df))

    # ----------------- internal helpers -----------------

    def _append_window(self, arr: List[float], value: float) -> None:
        arr.append(value)
        if len(arr) > self.window_size:
            del arr[:-self.window_size]

    def _log_data_features(self, features: Dict[str, Any]) -> None:
        self.current_batch.append(features)
        if len(self.current_batch) > self.window_size:
            self.current_batch = self.current_batch[-self.window_size:]

    def _ensure_metric_descriptors(self) -> None:
        """Ensure all metric descriptors are created in GCP"""
        # Core metrics
        core_metrics = [
            "validation_score",
            "fairness_score",
            "data_drift_score",
            "model_decay_score",
        ]
        
        # Retrieval metrics
        retrieval_metrics = [
            "retrieval_num_docs",
            "retrieval_avg_score",
            "retrieval_max_score",
            "retrieval_min_score",
            "retrieval_score_std",
            "retrieval_source_diversity",
            "retrieval_category_diversity",
            "retrieval_source_entropy",
            "retrieval_precision",
            "retrieval_recall",
            "retrieval_f1_score",
        ]
        
        # Generation metrics
        generation_metrics = [
            "generation_response_length",
            "generation_num_sentences",
            "generation_avg_sentence_length",
            "generation_num_citations",
            "generation_unique_citations",
            "generation_similarity_score",
        ]
        
        # System metrics
        system_metrics = [
            "response_time",
            "tokens_per_second",
            "query_length",
        ]
        
        all_metrics = core_metrics + retrieval_metrics + generation_metrics + system_metrics
        
        for metric_name in all_metrics:
            full_name = f"{self.metric_prefix}/{metric_name}"
            descriptor = ga_metric.MetricDescriptor(
                type=full_name,
                metric_kind=ga_metric.MetricDescriptor.MetricKind.GAUGE,
                value_type=ga_metric.MetricDescriptor.ValueType.DOUBLE,
                description=f"RAG {metric_name.replace('_', ' ')}",
                labels=[
                    ga_label.LabelDescriptor(
                        key="model_name",
                        value_type=ga_label.LabelDescriptor.ValueType.STRING,
                        description="RAG model name",
                    )
                ],
            )
            try:
                self.client.create_metric_descriptor(
                    name=self.project_name,
                    metric_descriptor=descriptor,
                )
                logger.debug(f"Created metric descriptor: {metric_name}")
            except Exception as e:
                # Most often AlreadyExists; keep quiet unless it's a real error
                if "already exists" not in str(e).lower():
                    logger.debug(f"Note for {metric_name}: {e}")

    def write_time_series(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Check if we've written this metric too recently
        if metric_name in self.last_write_time:
            time_since_last = (timestamp - self.last_write_time[metric_name]).total_seconds()
            if time_since_last < self.min_write_interval_seconds:
                logger.debug(
                    f"Skipping write for {metric_name}: only {time_since_last:.1f}s since last write "
                    f"(minimum {self.min_write_interval_seconds}s)"
                )
                return

        try:
            # Create the time series object
            series = TimeSeries()
            series.metric.type = f"{self.metric_prefix}/{metric_name}"
            series.resource.type = "global"
            series.metric.labels["model_name"] = self.model_name

            # Create interval with end_time
            now = timestamp_pb2.Timestamp()
            now.FromDatetime(timestamp)
            
            interval = TimeInterval(end_time=now)
            
            # Create point with interval
            point = Point(
                interval=interval,
                value={"double_value": float(value)}
            )
            
            series.points.append(point)

            self.client.create_time_series(
                name=self.project_name,
                time_series=[series],
            )
            
            # Track successful write
            self.last_write_time[metric_name] = timestamp
            logger.debug(f"Successfully wrote metric {metric_name}={value}")
            
        except Exception as e:
            logger.warning("Failed to write metric %s: %s", metric_name, e)

    def _check_model_decay(self) -> Dict[str, Any]:
        """Check for model performance decay"""
        if not self.validation_scores and not self.fairness_scores:
            return {
                "recent_validation_score": None,
                "recent_fairness_score": None,
                "overall_decay_score": None,
            }

        recent_val = (
            float(np.mean(self.validation_scores)) if self.validation_scores else None
        )
        recent_fair = (
            float(np.mean(self.fairness_scores)) if self.fairness_scores else None
        )

        if recent_val is None and recent_fair is None:
            decay_score = None
        else:
            parts = []
            if recent_val is not None and self.performance_threshold > 0:
                parts.append(recent_val / self.performance_threshold)
            if recent_fair is not None and self.fairness_threshold > 0:
                parts.append(recent_fair / self.fairness_threshold)
            decay_score = float(min(parts)) if parts else None

        return {
            "recent_validation_score": recent_val,
            "recent_fairness_score": recent_fair,
            "overall_decay_score": decay_score,
        }

    def _detect_data_drift(self) -> Dict[str, Any]:
        """Detect data distribution drift using Evidently or fallback heuristic"""
        if not self.current_batch:
            return {
                "status": "no_current_data",
                "overall_drift_score": None,
                "drift_threshold": self.data_drift_threshold,
                "num_features_checked": None,
                "num_drifted_features": None,
                "drifted_features": None,
            }

        # Evidently if baseline exists
        if _EVIDENTLY_AVAILABLE and self.reference_df is not None:
            try:
                current_df = pd.DataFrame(self.current_batch)
                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=self.reference_df, current_data=current_df)
                result = report.as_dict()

                data_result = result["metrics"][0]["result"]
                dataset_drift = data_result["dataset_drift"]

                share_drifted = dataset_drift["share_drifted_features"]
                n_drifted = dataset_drift["number_of_drifted_features"]
                n_features = dataset_drift["number_of_features"]

                drifted_features = []
                for col_name, col_info in data_result["columns"].items():
                    if col_info.get("drift_detected"):
                        drifted_features.append(
                            {
                                "feature": col_name,
                                "drift_score": col_info.get("drift_score"),
                                "stattest_name": col_info.get("stattest_name"),
                            }
                        )

                overall_drift = float(share_drifted)

                status = "drift_detected" if n_drifted > 0 else "healthy"

                return {
                    "status": status,
                    "overall_drift_score": overall_drift,
                    "drift_threshold": self.data_drift_threshold,
                    "num_features_checked": n_features,
                    "num_drifted_features": n_drifted,
                    "drifted_features": drifted_features,
                }
            except Exception as e:
                logger.warning("Evidently drift detection failed: %s", e)

        # Fallback: simple numeric variance heuristic
        try:
            current_df = pd.DataFrame(self.current_batch)
            numeric_df = current_df.select_dtypes(include=["number"])
            if numeric_df.empty:
                return {
                    "status": "no_numeric_features",
                    "overall_drift_score": None,
                    "drift_threshold": self.data_drift_threshold,
                    "num_features_checked": 0,
                    "num_drifted_features": 0,
                    "drifted_features": [],
                }

            stds = numeric_df.std().fillna(0.0)
            max_std = float(stds.max()) if not stds.empty else 1.0
            overall_drift = float(stds.mean() / max_std) if max_std > 0 else 0.0

            n_features = int(len(stds))
            n_drifted = int((stds > stds.mean()).sum())
            drifted_features = [
                {"feature": name, "std": float(val)}
                for name, val in stds[stds > stds.mean()].items()
            ]

            status = "drift_detected" if overall_drift > self.data_drift_threshold else "healthy"

            return {
                "status": status,
                "overall_drift_score": overall_drift,
                "drift_threshold": self.data_drift_threshold,
                "num_features_checked": n_features,
                "num_drifted_features": n_drifted,
                "drifted_features": drifted_features,
            }
        except Exception as e:
            logger.warning("Fallback drift detection failed: %s", e)
            return {
                "status": "error",
                "overall_drift_score": None,
                "drift_threshold": self.data_drift_threshold,
                "num_features_checked": None,
                "num_drifted_features": None,
                "drifted_features": None,
            }


# ---------------------------------------------------------------------------
# High-level wrapper used by pipeline / API
# ---------------------------------------------------------------------------

class HybridMonitor:
    """
    High-level monitoring wrapper for RAG pipeline.
    
    Provides a simple interface for:
    - Logging query metrics
    - Checking model health
    - Detecting when retraining is needed
    - Setting baselines for drift detection
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        model_name: str = "rag-model",
        enable_gcp: bool = True,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.gcp_monitor: Optional[GCPModelMonitor] = None

        if enable_gcp:
            self._initialize_gcp(project_id, model_name)
        else:
            self.logger.info("GCP monitoring explicitly disabled")

    def _initialize_gcp(self, project_id: Optional[str], model_name: str) -> None:
        project_id = project_id or os.getenv("GCP_PROJECT_ID")
        if not project_id:
            self.logger.warning(
                "GCP_PROJECT_ID not set. GCP monitoring will not be enabled."
            )
            return
        try:
            self.gcp_monitor = GCPModelMonitor(
                project_id=project_id,
                model_name=model_name,
            )
            self.logger.info("✓ GCP monitoring enabled for project %s", project_id)
        except Exception as e:
            self.logger.error("Failed to initialize GCPModelMonitor: %s", e)
            self.gcp_monitor = None

    def log_query(self, query_result: Dict[str, Any]) -> None:
        """Log metrics from a query result"""
        if not self.gcp_monitor:
            return
        try:
            self.gcp_monitor.log_query_metrics(query_result)
        except Exception as e:
            self.logger.warning("GCP monitor logging failed: %s", e)

    def check_health(self) -> Dict[str, Any]:
        """Check overall model health"""
        if not self.gcp_monitor:
            return {
                "status": "ERROR",
                "source": "none",
                "message": "GCP monitoring not available",
            }
        try:
            summary = self.gcp_monitor.get_monitoring_summary()
            summary["status"] = "OK"
            summary["source"] = "gcp"
            return summary
        except Exception as e:
            self.logger.error("GCP health check failed: %s", e)
            return {
                "status": "ERROR",
                "source": "gcp",
                "message": f"GCP monitoring error: {e}",
            }

    def should_trigger_retraining(self) -> Dict[str, Any]:
        """Check if model retraining should be triggered"""
        if not self.gcp_monitor:
            return {
                "should_retrain": False,
                "reasons": ["GCP monitoring not available"],
                "source": "none",
            }
        try:
            decision = self.gcp_monitor.should_trigger_retraining()
            decision["source"] = "gcp"
            return decision
        except Exception as e:
            self.logger.error("GCP retraining check failed: %s", e)
            return {
                "should_retrain": False,
                "reasons": [f"GCP monitoring error: {e}"],
                "source": "gcp",
            }

    def set_baseline(self) -> None:
        """Set baseline for drift detection using Evidently"""
        if not self.gcp_monitor:
            self.logger.warning("Cannot set baseline: GCP monitor not available")
            return
        try:
            self.gcp_monitor.set_evidently_baseline()
            self.logger.info("✓ GCP Evidently baseline set")
        except Exception as e:
            self.logger.warning("GCP baseline failed: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        return {
            "gcp_monitor": {
                "available": self.gcp_monitor is not None,
                "status": "active" if self.gcp_monitor else "unavailable",
                "project_id": os.getenv("GCP_PROJECT_ID", "not_set"),
            },
            "mode": "gcp_only" if self.gcp_monitor else "none",
        }


def add_monitoring_to_pipeline(
    pipeline: Any,
    project_id: Optional[str] = None,
    model_name: str = "rag-model",
    enable_gcp: bool = True,
) -> HybridMonitor:
    """
    Add monitoring to an existing RAG pipeline.
    
    This wraps the pipeline's query method to automatically log metrics
    after each query.
    
    Args:
        pipeline: The RAG pipeline instance
        project_id: GCP project ID (or from GCP_PROJECT_ID env var)
        model_name: Name for the model in metrics
        enable_gcp: Whether to enable GCP Cloud Monitoring
        
    Returns:
        HybridMonitor instance attached to the pipeline
    """
    monitoring = HybridMonitor(
        project_id=project_id,
        model_name=model_name,
        enable_gcp=enable_gcp,
    )

    pipeline.monitoring = monitoring

    if hasattr(pipeline, "query"):
        original_query = pipeline.query

        def monitored_query(*args, **kwargs):
            result = original_query(*args, **kwargs)
            try:
                monitoring.log_query(result)
            except Exception as e:
                logger.error("Error logging query to monitoring: %s", e)
            return result

        pipeline.query = monitored_query  # type: ignore[assignment]

    status = monitoring.get_status()
    logger.info("Monitoring attached to pipeline (mode=%s)", status["mode"])
    return monitoring