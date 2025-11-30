
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

class FastAPISettings(BaseModel):
    rate_limit_query: str = "10/minute"
    rate_limit_feedback: str = "20/hour"

class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  
    dimension: int = 384  
    normalize: bool = True
    batch_size: int = 32


class ChunkingConfig(BaseModel):
    """Document chunking configuration"""
    paper_chunk_size: int = 512
    paper_chunk_overlap: int = 50
    news_chunk_size: int = 1024
    news_chunk_overlap: int = 50
    separators: List[str] = ["\n\n", "\n", ". ", " "]


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    # Hybrid retrieval weights
    dense_weight: float = 0.7  
    sparse_weight: float = 0.3 
    
    # Retrieval parameters
    top_k: int = 20
    rerank_top_k: int = 10
    diversity_top_k: int = 8
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_weight: float = 0.6
    original_score_weight: float = 0.4


class GenerationConfig(BaseModel):
    """LLM generation configuration"""
    model_name: str = "gemini-2.0-flash-001" 
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.9
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))  


class VectorStoreConfig(BaseModel):
    """FAISS vector database configuration"""
    provider: str = "faiss"
    persist_directory: str = "./faiss_indexes"
    papers_index_name: str = "tech-papers"
    news_index_name: str = "tech-news"
    
    # FAISS parameters
    index_type: str = "Flat" 
    nlist: int = 100
    nprobe: int = 10
    
    # flat parameters
    flat_m: int = 32  # Number of connections per layer
    flat_ef_construction: int = 40  # Size of dynamic candidate list during construction
    flat_ef_search: int = 16  # Size of dynamic candidate list during search
    
    # Metadata storage
    use_metadata_db: bool = True


class BM25Config(BaseModel):
    """BM25 configuration"""
    persist_directory: str = "./bm25_indexes"
    k1: float = 1.5
    b: float = 0.75
    tokenizer: str = "nltk"
    

class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = True
    cache_dir: str = "./cache"  # File-based cache only
    ttl: int = 3600


class BiasDetectionConfig(BaseModel):
    """Bias detection configuration"""
    max_source_ratio: float = 0.5
    min_source_diversity: int = 3
    min_temporal_span_days: int = 7

class GCPConfig(BaseModel):
    """GCP configuration for Artifact Registry"""
    project_id: str = Field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "")
    )
    location: str = Field(
        default_factory=lambda: os.getenv("GCP_LOCATION", "us-central1")
    )
    artifact_repository: str = Field(
        default_factory=lambda: os.getenv("GCP_ARTIFACT_REPOSITORY", "rag-models")
    )

class RAGConfig(BaseModel):
    """Main RAG pipeline configuration"""
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    bm25: BM25Config = BM25Config()
    cache: CacheConfig = CacheConfig()
    bias_detection: BiasDetectionConfig = BiasDetectionConfig()
    fastapi: FastAPISettings = FastAPISettings()
    
    # MLflow tracking - LOCAL FILE STORAGE
    mlflow_tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "./mlruns") 
    )
    mlflow_experiment_name: str = "tech-trends-rag"
    
    # Logging
    log_level: str = "INFO"

# Global config instance
config = RAGConfig()