from typing import Dict, Any, Optional
import logging
import mlflow
from datetime import datetime

from config.settings import config

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Track RAG experiments with MLflow"""
    
    def __init__(self):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        
        self.logger.info(f"MLflow tracking: {self.config.mlflow_tracking_uri}")
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"rag_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = mlflow.start_run(run_name=run_name)
        self.logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_config(self):
        """Log configuration parameters"""
        mlflow.log_param("embedding_model", config.embedding.model_name)
        mlflow.log_param("embedding_dim", config.embedding.dimension)
        
        mlflow.log_param("dense_weight", config.retrieval.dense_weight)
        mlflow.log_param("sparse_weight", config.retrieval.sparse_weight)
        mlflow.log_param("top_k", config.retrieval.top_k)
        mlflow.log_param("rerank_top_k", config.retrieval.rerank_top_k)
        
        mlflow.log_param("llm_model", config.generation.model_name)
        mlflow.log_param("temperature", config.generation.temperature)
        mlflow.log_param("max_tokens", config.generation.max_tokens)
        
        mlflow.log_param("vector_store", config.vector_store.provider)
        mlflow.log_param("index_type", config.vector_store.index_type)
    
    def log_query_result(self, 
                        query: str,
                        response: str,
                        metrics: Dict[str, Any],
                        retrieved_docs: list,
                        bias_report: Optional[Dict[str, Any]] = None):
        """
        Log a single query-response pair with metrics
        
        Args:
            query: User query
            response: Generated response
            metrics: Metrics dictionary
            retrieved_docs: Retrieved documents
            bias_report: Optional bias detection report
        """
        mlflow.log_param("query", query[:100])  
        mlflow.log_param("num_retrieved", len(retrieved_docs))
        
        if 'retrieval_metrics' in metrics:
            for key, value in metrics['retrieval_metrics'].items():
                mlflow.log_metric(f"retrieval_{key}", value)
        
        if 'generation_metrics' in metrics:
            for key, value in metrics['generation_metrics'].items():
                mlflow.log_metric(f"generation_{key}", value)
        
        if 'response_time_seconds' in metrics:
            mlflow.log_metric("response_time_seconds", metrics['response_time_seconds'])
        
        if bias_report:
            if 'overall_fairness_score' in bias_report:
                mlflow.log_metric("fairness_overall_score", bias_report['overall_fairness_score'])
                
                diversity_metrics = bias_report.get('diversity_metrics', {})
                if diversity_metrics:
                    mlflow.log_metric("fairness_source_diversity", 
                                    diversity_metrics.get('source_diversity', 0.0))
                    mlflow.log_metric("fairness_category_diversity", 
                                    diversity_metrics.get('category_diversity', 0.0))
                    mlflow.log_metric("fairness_num_unique_sources", 
                                    diversity_metrics.get('num_unique_sources', 0))
            
            elif 'overall_bias_score' in bias_report:
                mlflow.log_metric("bias_overall_score", bias_report['overall_bias_score'])
                if 'source_bias' in bias_report:
                    mlflow.log_metric("source_diversity", 
                                    bias_report['source_bias'].get('source_diversity', 0.0))
        
        mlflow.log_text(response, "response.txt")
        
        import json
        mlflow.log_dict(metrics, "metrics.json")
        if bias_report:
            mlflow.log_dict(bias_report, "fairness_report.json")
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        self.logger.info("Ended MLflow run")