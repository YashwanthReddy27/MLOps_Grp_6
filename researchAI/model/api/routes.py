"""
API Routes for RAG Pipeline
"""
import logging
import time
import uuid
from typing import Callable, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
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
    
    
    # ===== Query Endpoint =====
    @router.post("/query", response_model=QueryResponse, tags=["query"])
    @limiter.limit("10/minute")  # Rate limit: 10 requests per minute
    async def query(request: QueryRequest, background_tasks: BackgroundTasks):
        """
        Process a RAG query
        
        Rate limited to 10 requests per minute per IP.
        """
        pipeline = get_pipeline()
        
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        try:
            # Process query
            result = pipeline.query(
                query=request.query,
                filters=request.filters,
                enable_streaming=request.enable_streaming
            )
            
            # Update metrics in background
            background_tasks.add_task(
                update_metrics,
                result['response_time'],
                result['validation']['overall_score'],
                result['bias_report'].get('overall_fairness_score', 0.0),
                result.get('from_cache', False)
            )
            
            # Format response
            response = QueryResponse(
                query=result['query'],
                response=result['response'],
                sources=[Source(**s) for s in result['sources']],
                num_sources=result['num_sources'],
                bias_report=BiasReport(**result['bias_report']),
                validation=result['validation'],
                metrics=result['metrics'],
                response_time=result['response_time'],
                from_cache=result.get('from_cache', False)
            )
            
            logger.info(f"Query processed successfully in {result['response_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    
    # ===== Index Update Endpoint =====
    @router.post("/index/update", response_model=IndexUpdateResponse, tags=["indexing"])
    @limiter.limit("5/hour")  # Rate limit: 5 requests per hour
    async def update_index(request: IndexUpdateRequest):
        """
        Update indexes with new documents
        
        Rate limited to 5 requests per hour.
        """
        pipeline = get_pipeline()
        
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        if not request.papers and not request.news:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        logger.info(
            f"Index update requested - Papers: {len(request.papers or [])}, "
            f"News: {len(request.news or [])}, Mode: {request.mode}"
        )
        
        start_time = time.time()
        
        try:
            if request.mode == "rebuild":
                # Rebuild indexes from scratch
                pipeline.index_documents(
                    papers=request.papers,
                    news=request.news
                )
            else:
                # Update existing indexes
                pipeline.update_indexes(
                    papers=request.papers,
                    news=request.news
                )
            
            duration = time.time() - start_time
            
            # Get updated stats
            stats = pipeline.retriever.get_index_stats()
            
            response = IndexUpdateResponse(
                status="success",
                message="Indexes updated successfully",
                papers_indexed=len(request.papers or []),
                news_indexed=len(request.news or []),
                total_chunks=stats['papers']['dense']['total_vectors'] + stats['news']['dense']['total_vectors'],
                duration=duration
            )
            
            logger.info(f"Index update completed in {duration:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error updating indexes: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error updating indexes: {str(e)}")
    
    
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
    async def submit_feedback(request: FeedbackRequest):
        """
        Submit user feedback on a response
        
        Rate limited to 20 requests per hour.
        """
        feedback_id = str(uuid.uuid4())
        
        feedback_entry = {
            "feedback_id": feedback_id,
            "query": request.query,
            "response_id": request.response_id,
            "rating": request.rating,
            "feedback_text": request.feedback_text,
            "issues": request.issues,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store feedback (in production, use database)
        feedback_store.append(feedback_entry)
        
        logger.info(f"Feedback received: {feedback_id} - Rating: {request.rating}")
        
        return FeedbackResponse(
            status="success",
            message="Feedback submitted successfully",
            feedback_id=feedback_id
        )
    
    
    return router


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