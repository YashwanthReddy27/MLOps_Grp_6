import os
import sys
import logging
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

# # Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import TechTrendsRAGPipeline
from api.routes import create_api_router
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[TechTrendsRAGPipeline] = None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global pipeline
    logger.info("Starting up RAG Pipeline API...")
    
    try:
        # Initialize pipeline
        pipeline = TechTrendsRAGPipeline(enable_tracking=True)
        
        # Load existing indexes
        logger.info("Loading indexes...")
        if pipeline.load_indexes():
            logger.info("✓ Indexes loaded successfully")
            stats = pipeline.retriever.get_index_stats()
            logger.info(f"Index stats: {stats}")
        else:
            logger.warning("⚠ No indexes found. You need to index documents first.")
        
        logger.info("✓ RAG Pipeline API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Pipeline API...")
    pipeline = None
    logger.info("✓ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Tech Trends RAG API",
    description="REST API for Technology Trends RAG Pipeline with Fairness Detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"← {request.method} {request.url.path} "
        f"Status: {response.status_code} Duration: {duration:.2f}s"
    )
    
    return response


# Include API routes
app.include_router(
    create_api_router(lambda: pipeline),
    prefix="/api",
    tags=["api"]
)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Tech Trends RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health"
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


def main():
    """Run the API server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )


if __name__ == "__main__":
    main()