"""
Hybrid Monitoring Solution - Best of Both Worlds

Combines simple local monitoring (your current monitor.py) with 
GCP Cloud Monitoring for production-ready monitoring with failover.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HybridMonitor:
    """
    Hybrid monitoring that uses both local and GCP monitoring
    
    Benefits:
    - Local backup (always works, no dependencies)
    - GCP advanced features (when available)
    - Automatic failover
    - Zero downtime migration path
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        enable_gcp: bool = True
    ):
        """
        Initialize hybrid monitoring
        
        Args:
            project_id: GCP project ID (optional, from env if not provided)
            enable_gcp: Whether to enable GCP monitoring
        """
        self.logger = logging.getLogger(__name__)
        
        # Always initialize simple monitor (backup)
        try:
            from monitor import SimpleMonitor
            self.simple_monitor = SimpleMonitor()
            self.logger.info("✓ Simple monitor initialized (local backup)")
        except Exception as e:
            self.logger.error(f"Failed to initialize simple monitor: {e}")
            self.simple_monitor = None
        
        # Try to initialize GCP monitor
        self.gcp_monitor = None
        if enable_gcp:
            self._initialize_gcp(project_id)
    
    def _initialize_gcp(self, project_id: Optional[str]):
        """Initialize GCP monitoring if available"""
        try:
            from monitoring.monitoring_integration import MonitoringIntegration
            
            project_id = project_id or os.getenv('GCP_PROJECT_ID')
            
            if not project_id:
                self.logger.info(
                    "GCP_PROJECT_ID not set. Using simple monitor only. "
                    "Set GCP_PROJECT_ID to enable cloud monitoring."
                )
                return
            
            self.gcp_monitor = MonitoringIntegration(
                project_id=project_id,
                enable_monitoring=True
            )
            
            if self.gcp_monitor.enable_monitoring:
                self.logger.info("✓ GCP monitoring enabled")
            else:
                self.logger.warning("GCP monitoring failed to initialize")
                self.gcp_monitor = None
                
        except ImportError:
            self.logger.info(
                "GCP monitoring dependencies not installed. Using simple monitor. "
                "To enable: pip install google-cloud-monitoring"
            )
        except Exception as e:
            self.logger.warning(f"GCP monitoring unavailable: {e}")
    
    def log_query(self, query_result: Dict[str, Any]):
        """
        Log query to both monitoring systems
        
        Args:
            query_result: Result from pipeline.query()
        """
        # Always log to simple monitor (backup)
        if self.simple_monitor:
            try:
                self.simple_monitor.log_query(query_result)
            except Exception as e:
                self.logger.error(f"Simple monitor logging failed: {e}")
        
        # Also log to GCP if available
        if self.gcp_monitor:
            try:
                self.gcp_monitor.log_query(query_result)
            except Exception as e:
                self.logger.warning(f"GCP monitor logging failed: {e}")
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check model health
        
        Uses GCP if available, falls back to simple monitor
        
        Returns:
            Health status dictionary
        """
        # Prefer GCP (more detailed)
        if self.gcp_monitor:
            try:
                health = self.gcp_monitor.check_model_health()
                health['source'] = 'gcp'
                return health
            except Exception as e:
                self.logger.warning(f"GCP health check failed: {e}, using simple monitor")
        
        # Fallback to simple monitor
        if self.simple_monitor:
            try:
                health = self.simple_monitor.check_health()
                health['source'] = 'local'
                return health
            except Exception as e:
                self.logger.error(f"Simple monitor health check failed: {e}")
                return {
                    'status': 'ERROR',
                    'source': 'none',
                    'message': 'All monitoring systems failed'
                }
        
        return {
            'status': 'ERROR',
            'source': 'none',
            'message': 'No monitoring system available'
        }
    
    def should_trigger_retraining(self) -> Dict[str, Any]:
        """
        Check if retraining should be triggered
        
        Uses GCP if available (more accurate), falls back to simple monitor
        
        Returns:
            Retraining decision
        """
        # Prefer GCP (more sophisticated analysis)
        if self.gcp_monitor:
            try:
                decision = self.gcp_monitor.should_trigger_retraining()
                decision['source'] = 'gcp'
                return decision
            except Exception as e:
                self.logger.warning(f"GCP retraining check failed: {e}")
        
        # Fallback to simple monitor
        if self.simple_monitor:
            try:
                health = self.simple_monitor.check_health()
                
                # Convert simple monitor format to standard format
                decision = {
                    'should_retrain': health.get('needs_retraining', False),
                    'reasons': health.get('reasons', []),
                    'status': health.get('status', 'UNKNOWN'),
                    'source': 'local',
                    'metrics': {
                        'validation_score': health.get('avg_validation_score'),
                        'fairness_score': health.get('avg_fairness_score')
                    }
                }
                return decision
            except Exception as e:
                self.logger.error(f"Simple monitor retraining check failed: {e}")
        
        return {
            'should_retrain': False,
            'reasons': ['Monitoring unavailable'],
            'source': 'none'
        }
    
    def set_baseline(self):
        """
        Set baseline statistics
        
        Sets baseline in both systems
        """
        # Set baseline in simple monitor
        if self.simple_monitor:
            try:
                baseline = self.simple_monitor.establish_baseline()
                if baseline:
                    self.logger.info("✓ Simple monitor baseline set")
            except Exception as e:
                self.logger.warning(f"Simple monitor baseline failed: {e}")
        
        # Set baseline in GCP monitor
        if self.gcp_monitor:
            try:
                self.gcp_monitor.set_baseline()
                self.logger.info("✓ GCP monitor baseline set")
            except Exception as e:
                self.logger.warning(f"GCP baseline failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get monitoring system status
        
        Returns:
            Status of both monitoring systems
        """
        return {
            'simple_monitor': {
                'available': self.simple_monitor is not None,
                'status': 'active' if self.simple_monitor else 'unavailable'
            },
            'gcp_monitor': {
                'available': self.gcp_monitor is not None,
                'status': 'active' if self.gcp_monitor else 'unavailable',
                'project_id': os.getenv('GCP_PROJECT_ID', 'not_set')
            },
            'mode': self._get_mode()
        }
    
    def _get_mode(self) -> str:
        """Determine current operating mode"""
        if self.gcp_monitor and self.simple_monitor:
            return 'hybrid'
        elif self.gcp_monitor:
            return 'gcp_only'
        elif self.simple_monitor:
            return 'local_only'
        else:
            return 'none'


def add_hybrid_monitoring_to_pipeline(
    pipeline,
    project_id: Optional[str] = None,
    enable_gcp: bool = True
):
    """
    Add hybrid monitoring to pipeline
    
    Args:
        pipeline: TechTrendsRAGPipeline instance
        project_id: GCP project ID (optional)
        enable_gcp: Whether to enable GCP monitoring
        
    Returns:
        HybridMonitor instance
    """
    monitoring = HybridMonitor(
        project_id=project_id,
        enable_gcp=enable_gcp
    )
    
    # Store in pipeline
    pipeline.monitoring = monitoring
    
    # Wrap query method to log automatically
    original_query = pipeline.query
    
    def monitored_query(*args, **kwargs):
        result = original_query(*args, **kwargs)
        
        # Log to monitoring
        try:
            monitoring.log_query(result)
        except Exception as e:
            logger.error(f"Error logging query: {e}")
        
        return result
    
    pipeline.query = monitored_query
    
    # Log status
    status = monitoring.get_status()
    logger.info(f"Hybrid monitoring enabled in {status['mode']} mode")
    
    return monitoring