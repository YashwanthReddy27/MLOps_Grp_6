"""
Pydantic schemas for API requests and responses
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# Query schemas
class QueryRequest(BaseModel):
    """Request schema for query endpoint"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters (categories, date range, etc.)")
    enable_streaming: bool = Field(False, description="Enable streaming response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the latest developments in reinforcement learning?",
                "filters": {
                    "categories": ["artificial_intelligence", "machine_learning"]
                },
                "enable_streaming": False
            }
        }


class Source(BaseModel):
    """Source citation schema"""
    number: int
    title: str
    source: str
    url: str
    date: str


class BiasReport(BaseModel):
    """Bias/Fairness report schema"""
    overall_fairness_score: float
    query_characteristics: Dict[str, Any]
    diversity_metrics: Dict[str, Any]
    fairness_indicators: Dict[str, Any]
    warnings: List[Dict[str, Any]]
    evaluation_type: str
    timestamp: str


class QueryResponse(BaseModel):
    """Response schema for query endpoint"""
    query: str
    response: str
    sources: List[Source]
    num_sources: int
    bias_report: BiasReport
    validation: Dict[str, Any]
    metrics: Dict[str, Any]
    response_time: float
    from_cache: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the latest developments in AI?",
                "response": "Recent developments in AI include...",
                "sources": [
                    {
                        "number": 1,
                        "title": "Advances in Large Language Models",
                        "source": "arXiv",
                        "url": "https://arxiv.org/...",
                        "date": "2024-01-15"
                    }
                ],
                "num_sources": 5,
                "response_time": 2.34,
                "from_cache": False
            }
        }


# Index update schemas
class IndexUpdateRequest(BaseModel):
    """Request schema for index update endpoint"""
    papers: Optional[List[Dict[str, Any]]] = Field(None, description="List of papers to index")
    news: Optional[List[Dict[str, Any]]] = Field(None, description="List of news articles to index")
    mode: str = Field("update", description="'update' or 'rebuild'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "papers": [],
                "news": [],
                "mode": "update"
            }
        }


class IndexUpdateResponse(BaseModel):
    """Response schema for index update endpoint"""
    status: str
    message: str
    papers_indexed: int
    news_indexed: int
    total_chunks: int
    duration: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Health check schemas
class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    status: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    pipeline_loaded: bool
    indexes_loaded: bool
    index_stats: Optional[Dict[str, Any]] = None
    version: str = "1.0.0"
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "pipeline_loaded": True,
                "indexes_loaded": True,
                "version": "1.0.0"
            }
        }


# Metrics schemas
class MetricsResponse(BaseModel):
    """Response schema for metrics endpoint"""
    total_queries: int
    avg_response_time: float
    avg_validation_score: float
    avg_fairness_score: float
    cache_hit_rate: float
    uptime: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Stats schemas
class StatsResponse(BaseModel):
    """Response schema for stats endpoint"""
    papers_index: Dict[str, Any]
    news_index: Dict[str, Any]
    total_documents: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "papers_index": {
                    "dense": {"total_vectors": 1500},
                    "sparse": {"total_documents": 1500}
                },
                "news_index": {
                    "dense": {"total_vectors": 2000},
                    "sparse": {"total_documents": 2000}
                },
                "total_documents": 3500
            }
        }


# Feedback schemas
class FeedbackRequest(BaseModel):
    """Request schema for user feedback"""
    query: str
    response_id: Optional[str] = None
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, max_length=1000)
    issues: Optional[List[str]] = Field(None, description="List of issues: 'inaccurate', 'biased', 'incomplete', etc.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the latest AI trends?",
                "rating": 4,
                "feedback_text": "Good response but could include more recent sources",
                "issues": ["incomplete"]
            }
        }


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission"""
    status: str
    message: str
    feedback_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())