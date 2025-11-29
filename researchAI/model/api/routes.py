"""
API Routes for RAG Pipeline
"""
import logging
import time
import uuid
from typing import Callable, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.schemas import (
    QueryRequest, QueryResponse, IndexUpdateRequest, IndexUpdateResponse,
    HealthResponse, MetricsResponse, StatsResponse, FeedbackRequest, FeedbackResponse,
    Source, BiasReport
)

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# In-memory metrics storage (in production, use Redis or database)
metrics_store = {
    "total_queries": 0,
    "total_response_time": 0.0,
    "total_validation_score": 0.0,
    "total_fairness_score": 0.0,
    "cache_hits": 0,
    "start_time": time.time()
}

# In-memory feedback storage (in production, use database)
feedback_store = []


def create_api_router(get_pipeline: Callable) -> APIRouter:
    """
    Create API router with dependency injection for pipeline
    
    Args:
        get_pipeline: Function that returns the pipeline instance
        
    Returns:
        Configured APIRouter
    """
    router = APIRouter()
    
    # ===== Health Check Endpoint =====
    @router.get("/health", response_model=HealthResponse, tags=["monitoring"])
    async def health_check():
        """
        Health check endpoint for Kubernetes liveness/readiness probes
        """
        pipeline = get_pipeline()
        
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Get index stats if available
        index_stats = None
        indexes_loaded = False
        
        try:
            index_stats = pipeline.retriever.get_index_stats()
            indexes_loaded = True
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
        
        return HealthResponse(
            status="healthy",
            pipeline_loaded=True,
            indexes_loaded=indexes_loaded,
            index_stats=index_stats
        )

    @router.post("/monitor/establish-baseline", tags=["monitoring"])
    async def establish_baseline():
        """Create performance baseline"""
        try:
            monitor = SimpleMonitor()
            baseline = monitor.establish_baseline()
            
            if baseline:
                return {
                    "status": "success",
                    "baseline": baseline
                }
            else:
                return {
                    "status": "failed",
                    "message": "Need at least 50 queries"
                }
        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")
            return {"status": "error", "message": str(e)}
    
    
    # ===== Query Endpoint =====
    @router.post("/query", response_model=QueryResponse, tags=["query"])
    @limiter.limit("10/minute")  # Rate limit: 10 requests per minute
    async def query(
        request: Request,  # ADDED: Required for SlowAPI
        query_request: QueryRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Process a RAG query
        
        Rate limited to 10 requests per minute per IP.
        """
        pipeline = get_pipeline()
        
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        logger.info(f"Processing query: {query_request.query[:100]}...")
        
        try:
            # Process query
            result = pipeline.query(
                query=query_request.query,
                filters=query_request.filters,
                enable_streaming=query_request.enable_streaming
            )
            
            # Debug: Log what keys are actually in the result
            logger.debug(f"Pipeline result keys: {list(result.keys())}")
            
            # Extract values with safe defaults
            response_time = result.get('response_time', 0.0)
            validation = result.get('validation', {})
            validation_score = validation.get('overall_score', 0.0)
            bias_report = result.get('bias_report', {})
            fairness_score = bias_report.get('overall_fairness_score', 0.0)
            from_cache = result.get('from_cache', False)
            
            # Update metrics in background
            background_tasks.add_task(
                update_metrics,
                response_time,
                validation_score,
                fairness_score,
                from_cache
            )
            
            # Add monitoring in background (add this line)
            background_tasks.add_task(log_to_monitor, response)


            # Format response with safe defaults
            # Handle sources safely
            sources_list = []
            for source_data in result.get('sources', []):
                try:
                    # Ensure source has all required fields
                    source_obj = Source(
                        number=source_data.get('number', 0),
                        title=source_data.get('title', 'Unknown'),
                        source=source_data.get('source', 'Unknown'),
                        date=source_data.get('date', 'N/A'),
                        url=source_data.get('url', ''),
                        relevance_score=source_data.get('relevance_score', 0.0),
                        excerpt=source_data.get('excerpt', '')
                    )
                    sources_list.append(source_obj)
                except Exception as e:
                    logger.warning(f"Failed to parse source: {e}")
                    continue
            
            # Handle BiasReport with all required fields
            if bias_report and isinstance(bias_report, dict):
                # Add required fields if missing
                bias_report_data = {
                    'overall_fairness_score': bias_report.get('overall_fairness_score', 0.0),
                    'diversity_metrics': bias_report.get('diversity_metrics', {}),
                    'warnings': bias_report.get('warnings', []),
                    'query_characteristics': bias_report.get('query_characteristics', {}),
                    'fairness_indicators': bias_report.get('fairness_indicators', {}),
                    'evaluation_type': bias_report.get('evaluation_type', 'standard'),
                    'timestamp': bias_report.get('timestamp', datetime.now().isoformat())
                }
                bias_report_obj = BiasReport(**bias_report_data)
            else:
                # Create default BiasReport
                bias_report_obj = BiasReport(
                    overall_fairness_score=0.0,
                    diversity_metrics={},
                    warnings=[],
                    query_characteristics={},
                    fairness_indicators={},
                    evaluation_type='standard',
                    timestamp=datetime.now().isoformat()
                )
            
            response = QueryResponse(
                query=result.get('query', query_request.query),
                response=result.get('response', ''),
                sources=sources_list,
                num_sources=len(sources_list),
                bias_report=bias_report_obj,
                validation=validation,
                metrics=result.get('metrics', {}),
                response_time=response_time,
                from_cache=from_cache
            )
            
            logger.info(f"Query processed successfully in {response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    # ===== Metrics Endpoint =====
    @router.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
    async def get_metrics():
        """
        Get aggregated system metrics
        """
        total_queries = metrics_store["total_queries"]
        
        if total_queries == 0:
            return MetricsResponse(
                total_queries=0,
                avg_response_time=0.0,
                avg_validation_score=0.0,
                avg_fairness_score=0.0,
                cache_hit_rate=0.0,
                uptime=time.time() - metrics_store["start_time"]
            )
        
        return MetricsResponse(
            total_queries=total_queries,
            avg_response_time=metrics_store["total_response_time"] / total_queries,
            avg_validation_score=metrics_store["total_validation_score"] / total_queries,
            avg_fairness_score=metrics_store["total_fairness_score"] / total_queries,
            cache_hit_rate=metrics_store["cache_hits"] / total_queries,
            uptime=time.time() - metrics_store["start_time"]
        )
    

    @router.get("/monitor/health", tags=["monitoring"])
    async def monitor_health():
        """Check model health"""
        try:
            monitor = SimpleMonitor()
            health = monitor.check_health()
            return health
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {"status": "ERROR", "message": str(e)}
    
    
    # ===== Stats Endpoint =====
    @router.get("/stats", response_model=StatsResponse, tags=["monitoring"])
    async def get_stats():
        """
        Get index statistics
        """
        pipeline = get_pipeline()
        
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        try:
            stats = pipeline.retriever.get_index_stats()
            
            total_docs = (
                stats['papers']['dense']['total_vectors'] +
                stats['news']['dense']['total_vectors']
            )
            
            return StatsResponse(
                papers_index=stats['papers'],
                news_index=stats['news'],
                total_documents=total_docs
            )
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
    
    
    # ===== Feedback Endpoint =====
    @router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
    @limiter.limit("20/hour")
    async def submit_feedback(
        request: Request,  # ADDED: Required for SlowAPI
        feedback_request: FeedbackRequest
    ):
        """
        Submit user feedback on a response
        
        Rate limited to 20 requests per hour.
        """
        feedback_id = str(uuid.uuid4())
        
        feedback_entry = {
            "feedback_id": feedback_id,
            "query": feedback_request.query,
            "response_id": feedback_request.response_id,
            "rating": feedback_request.rating,
            "feedback_text": feedback_request.feedback_text,
            "issues": feedback_request.issues,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store feedback (in production, use database)
        feedback_store.append(feedback_entry)
        
        logger.info(f"Feedback received: {feedback_id} - Rating: {feedback_request.rating}")
        
        return FeedbackResponse(
            status="success",
            message="Feedback submitted successfully",
            feedback_id=feedback_id
        )
    
    
    return router

def log_to_monitor(result: dict):
    """Log query result to monitoring"""
    try:
        from monitoring.monitor import SimpleMonitor
        monitor = SimpleMonitor()
        monitor.log_query(result)
    except Exception as e:
        logger.error(f"Error logging to monitor: {e}")

def update_metrics(response_time: float, validation_score: float, 
                   fairness_score: float, from_cache: bool):
    """
    Update metrics store
    
    Args:
        response_time: Query response time
        validation_score: Validation score
        fairness_score: Fairness score
        from_cache: Whether result was from cache
    """
    metrics_store["total_queries"] += 1
    metrics_store["total_response_time"] += response_time
    metrics_store["total_validation_score"] += validation_score
    metrics_store["total_fairness_score"] += fairness_score
    
    if from_cache:
        metrics_store["cache_hits"] += 1