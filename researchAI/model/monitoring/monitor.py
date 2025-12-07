"""
Simple monitoring for model performance with drift detection
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

class SimpleMonitor:
    """Monitor model performance and detect issues"""
    
    def __init__(self):
        self.metrics_file = Path("monitoring/metrics.json")
        self.baseline_file = Path("monitoring/baseline.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_query(self, query_result: dict):
        """Save query metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "validation_score": query_result['validation']['overall_score'],
            "fairness_score": query_result['bias_report']['overall_fairness_score'],
            "response_time": query_result['response_time'],
            "num_sources": query_result.get('num_sources', 0)
        }
        
        metrics = self._load_metrics()
        metrics.append(metric)
        
        # Keep last 1000
        if len(metrics) > 1000:
            metrics = metrics[-1000:]
        
        self._save_metrics(metrics)
    
    def check_health(self) -> dict:
        """Check if model needs retraining"""
        metrics = self._load_metrics()
        
        if len(metrics) < 10:
            return {
                "status": "OK",
                "needs_retraining": False,
                "message": "Not enough data yet",
                "total_queries": len(metrics)
            }
        
        # Get last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        recent = [
            m for m in metrics 
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
        
        if not recent:
            return {
                "status": "OK",
                "needs_retraining": False,
                "message": "No recent queries",
                "total_queries": len(metrics)
            }
        
        # Calculate averages
        avg_validation = sum(m['validation_score'] for m in recent) / len(recent)
        avg_fairness = sum(m['fairness_score'] for m in recent) / len(recent)
        avg_response_time = sum(m['response_time'] for m in recent) / len(recent)
        
        # Check for data drift
        baseline = self._load_baseline()
        drift_detected = False
        drift_message = ""
        
        if baseline:
            val_drop = baseline['avg_validation'] - avg_validation
            fair_drop = baseline['avg_fairness'] - avg_fairness
            
            # Drift if >10% drop from baseline
            if val_drop > 0.1 or fair_drop > 0.1:
                drift_detected = True
                drift_message = f"Performance drop detected: Validation -{val_drop:.2%}, Fairness -{fair_drop:.2%}"
        
        # Check thresholds
        validation_ok = avg_validation >= 0.7
        fairness_ok = avg_fairness >= 0.6
        
        needs_retraining = not (validation_ok and fairness_ok) or drift_detected
        
        reasons = []
        if not validation_ok:
            reasons.append(f"Validation score {avg_validation:.3f} below 0.7")
        if not fairness_ok:
            reasons.append(f"Fairness score {avg_fairness:.3f} below 0.6")
        if drift_detected:
            reasons.append(drift_message)
        
        return {
            "status": "UNHEALTHY" if needs_retraining else "HEALTHY",
            "needs_retraining": needs_retraining,
            "avg_validation_score": round(avg_validation, 3),
            "avg_fairness_score": round(avg_fairness, 3),
            "avg_response_time": round(avg_response_time, 2),
            "total_queries": len(recent),
            "drift_detected": drift_detected,
            "reasons": reasons,
            "message": "; ".join(reasons) if reasons else "Model performing well"
        }
    
    def establish_baseline(self):
        """Create baseline from current metrics"""
        metrics = self._load_metrics()
        
        if len(metrics) < 50:
            return None
        
        # Use last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        baseline_data = [
            m for m in metrics 
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
        
        if not baseline_data:
            baseline_data = metrics[-50:]
        
        baseline = {
            "created_at": datetime.now().isoformat(),
            "avg_validation": sum(m['validation_score'] for m in baseline_data) / len(baseline_data),
            "avg_fairness": sum(m['fairness_score'] for m in baseline_data) / len(baseline_data),
            "num_queries": len(baseline_data)
        }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        return baseline
    
    def _load_metrics(self):
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_metrics(self, metrics):
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _load_baseline(self):
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return None