"""
Tech Trends RAG System
"""

__version__ = "1.0.0"

from .pipeline import TechTrendsRAGPipeline
from .data_processing.document_processor import DocumentProcessor
from .data_processing.chunking import DocumentChunker
from .data_processing.embedding import HybridEmbedder
from .retrieval.retriever import HybridRetriever
from .generation.generator import ResponseGenerator
from .config.settings import config

__all__ = [
    'TechTrendsRAGPipeline',
    'DocumentProcessor',
    'DocumentChunker',
    'HybridEmbedder',
    'HybridRetriever',
    'ResponseGenerator',
    'config'
]