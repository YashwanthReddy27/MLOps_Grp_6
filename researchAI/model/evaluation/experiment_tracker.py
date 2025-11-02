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
        
        # Set up MLflow
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
        # Log embedding config
        mlflow.log_param("embedding_model", config.embedding.model_name)
        mlflow.log_param("embedding_dim", config.embedding.dimension)
        
        # Log retrieval config
        mlflow.log_param("dense_weight", config.retrieval.dense_weight)
        mlflow.log_param("sparse_weight", config.retrieval.sparse_weight)
        mlflow.log_param("top_k", config.retrieval.top_k)
        mlflow.log_param("rerank_top_k", config.retrieval.rerank_top_k)
        
        # Log generation config
        mlflow.log_param("llm_model", config.generation.model_name)
        mlflow.log_param("temperature", config.generation.temperature)
        mlflow.log_param("max_tokens", config.generation.max_tokens)
        
        # Log vector store config
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
        # Log query metadata
        mlflow.log_param("query", query[:100])  # Truncate long queries
        mlflow.log_param("num_retrieved", len(retrieved_docs))
        
        # Log retrieval metrics
        if 'retrieval_metrics' in metrics:
            for key, value in metrics['retrieval_metrics'].items():
                mlflow.log_metric(f"retrieval_{key}", value)
        
        # Log generation metrics
        if 'generation_metrics' in metrics:
            for key, value in metrics['generation_metrics'].items():
                mlflow.log_metric(f"generation_{key}", value)
        
        # Log response time
        if 'response_time_seconds' in metrics:
            mlflow.log_metric("response_time_seconds", metrics['response_time_seconds'])
        
        # Log fairness metrics (updated structure)
        if bias_report:
            # Handle both old bias_detector format and new fairness_detector format
            if 'overall_fairness_score' in bias_report:
                # New fairness detector format
                mlflow.log_metric("fairness_overall_score", bias_report['overall_fairness_score'])
                
                # Log diversity metrics if available
                diversity_metrics = bias_report.get('diversity_metrics', {})
                if diversity_metrics:
                    mlflow.log_metric("fairness_source_diversity", 
                                    diversity_metrics.get('source_diversity', 0.0))
                    mlflow.log_metric("fairness_category_diversity", 
                                    diversity_metrics.get('category_diversity', 0.0))
                    mlflow.log_metric("fairness_num_unique_sources", 
                                    diversity_metrics.get('num_unique_sources', 0))
            
            elif 'overall_bias_score' in bias_report:
                # Legacy bias detector format (for backward compatibility)
                mlflow.log_metric("bias_overall_score", bias_report['overall_bias_score'])
                if 'source_bias' in bias_report:
                    mlflow.log_metric("source_diversity", 
                                    bias_report['source_bias'].get('source_diversity', 0.0))
        
        # Log artifacts
        mlflow.log_text(response, "response.txt")
        
        import json
        mlflow.log_dict(metrics, "metrics.json")
        if bias_report:
            # Log as fairness_report.json (updated name)
            mlflow.log_dict(bias_report, "fairness_report.json")
    
    def log_batch_metrics(self, metrics: Dict[str, Any]):
        """
        Log aggregated metrics for a batch of queries
        
        Args:
            metrics: Aggregated metrics
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
    
    def log_fairness_evaluation(self, fairness_evaluation: Dict[str, Any]):
        """
            Log comprehensive fairness evaluation results to MLflow
            
            This method logs results from the RAGBiasDetector which uses 
            slicing techniques to evaluate fairness across multiple dimensions:
            - Query complexity (simple/intermediate/advanced)
            - Domain type (AI/ML, cybersecurity, cloud, web3, robotics)
            - User experience level (beginner/intermediate/expert)
            - Geographic context (regions)
            - Source type (academic/industry/news/community)
            
            Args:
                fairness_evaluation: Comprehensive fairness evaluation results from 
                                    evaluate_fairness() method
        """
        
        # Log overall fairness metrics
        summary = fairness_evaluation.get('evaluation_summary', {})
        mlflow.log_metric("fairness_overall_score", summary.get('overall_fairness_score', 0.0))
        mlflow.log_metric("fairness_query_count", summary.get('query_count', 0))
        mlflow.log_metric("fairness_significant_biases_count", summary.get('significant_biases_count', 0))
        mlflow.log_param("fairness_mitigation_applied", summary.get('mitigation_applied', False))
        
        # Log bias analysis metrics
        bias_report = fairness_evaluation.get('bias_report', {})
        if bias_report:
            # Log slice metrics
            slice_metrics = bias_report.get('slice_metrics', {})
            for slice_name, categories in slice_metrics.items():
                for category, data in categories.items():
                    metrics = data.get('metrics', {})
                    sample_size = data.get('sample_size', 0)
                    
                    # Log key metrics for each slice category
                    mlflow.log_metric(f"fairness_{slice_name}_{category}_retrieval_score", 
                                    metrics.get('avg_retrieval_score', 0.0))
                    mlflow.log_metric(f"fairness_{slice_name}_{category}_response_quality", 
                                    metrics.get('avg_response_quality', 0.0))
                    mlflow.log_metric(f"fairness_{slice_name}_{category}_source_diversity", 
                                    metrics.get('source_diversity', 0.0))
                    mlflow.log_metric(f"fairness_{slice_name}_{category}_sample_size", sample_size)
            
            # Log significant biases
            significant_biases = bias_report.get('bias_analysis', {}).get('significant_biases', [])
            for i, bias in enumerate(significant_biases):
                mlflow.log_param(f"bias_{i}_slice", bias.get('slice', 'unknown'))
                mlflow.log_param(f"bias_{i}_type", bias.get('bias_type', 'unknown'))
                mlflow.log_param(f"bias_{i}_severity", bias.get('severity', 'unknown'))
                mlflow.log_param(f"bias_{i}_disadvantaged_group", 
                               bias.get('affected_groups', {}).get('disadvantaged', 'unknown'))
                mlflow.log_param(f"bias_{i}_advantaged_group", 
                               bias.get('affected_groups', {}).get('advantaged', 'unknown'))
        
        # Log mitigation results if available
        mitigation_result = fairness_evaluation.get('mitigation_result')
        if mitigation_result and mitigation_result.get('status') == 'success':
            mlflow.log_param("mitigation_strategy", mitigation_result.get('strategy', 'unknown'))
            
            # Log effectiveness metrics
            effectiveness = mitigation_result.get('effectiveness', {})
            if 'overall_improvement' in effectiveness:
                mlflow.log_metric("mitigation_overall_improvement", effectiveness['overall_improvement'])
            if 'overall_balance_improvement' in effectiveness:
                mlflow.log_metric("mitigation_balance_improvement", effectiveness['overall_balance_improvement'])
            
            # Log post-mitigation fairness score if available
            if 'post_mitigation_bias_report' in mitigation_result:
                post_score = mitigation_result['post_mitigation_bias_report'].get('overall_fairness_score', 0.0)
                mlflow.log_metric("fairness_post_mitigation_score", post_score)
                
                # Calculate and log improvement
                original_score = summary.get('overall_fairness_score', 0.0)
                improvement = post_score - original_score
                mlflow.log_metric("fairness_score_improvement", improvement)
    
        
        # Log detailed results as artifacts
        try:
            # Log full fairness evaluation as JSON
            fairness_data = fairness_evaluation.copy()
            # Remove evaluation_data to avoid large artifacts (keep only summary)
            if 'evaluation_data' in fairness_data:
                fairness_data['evaluation_data_count'] = len(fairness_data['evaluation_data'])
                del fairness_data['evaluation_data']
            
            mlflow.log_dict(fairness_data, "fairness_evaluation.json")
            
            # Log recommendations as text
            if recommendations:
                rec_text = "\n".join([
                    f"[{rec.get('priority', 'unknown').upper()}] {rec.get('description', 'No description')}"
                    for rec in recommendations
                ])
                mlflow.log_text(rec_text, "fairness_recommendations.txt")
            
            self.logger.info(f"Logged fairness evaluation with score: {summary.get('overall_fairness_score', 0.0):.2f}")
            
        except Exception as e:
            self.logger.warning(f"Failed to log fairness evaluation artifacts: {e}")
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        self.logger.info("Ended MLflow run")