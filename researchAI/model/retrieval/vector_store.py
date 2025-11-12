from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import pickle
from pathlib import Path
import logging

from config.settings import config

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS vector database with metadata storage"""
    
    def __init__(self, index_name: str):
        self.config = config.vector_store
        self.index_name = index_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.index = None
        self.chunk_ids = []
        self.metadata = []
        self.dimension = config.embedding.dimension
        
        # Paths
        self.persist_dir = Path(self.config.persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.persist_dir / f"{index_name}.faiss" 
        self.metadata_path = self.persist_dir / f"{index_name}_metadata.pkl"
        
        self.logger.info(f"Initialized FAISS vector store: {index_name}")
    
    def create_index(self):
        """Create a new FAISS flat index"""
        if self.config.index_type == "Flat":
            # Flat index - exact search
            self.index = faiss.IndexFlatIP(self.dimension)
            
        elif self.config.index_type == "IVFFlat":
            # IVF index - approximate search with clustering
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                self.config.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = self.config.nprobe
            
        elif self.config.index_type == "flat":
            # flat index - hierarchical navigable small world
            self.index = faiss.IndexFlatL2(self.dimension, self.config.flat_m)
            self.index.flat.efConstruction = self.config.flat_ef_construction
            self.index.flat.efSearch = self.config.flat_ef_search
            
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        self.logger.info(f"Created FAISS index: {self.config.index_type}")
    
    def train_index(self, embeddings: np.ndarray):
        """
        Train index if needed (for IVF-based indexes)
        
        Args:
            embeddings: Training embeddings
        """
        if self.config.index_type == "IVFFlat":
            if not self.index.is_trained:
                self.logger.info(f"Training IVF index with {len(embeddings)} vectors")
                self.index.train(embeddings)
                self.logger.info("Index training completed")
    
    def add(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks to the index
        
        Args:
            chunks: List of chunks with embeddings
        """
        if self.index is None:
            self.create_index()
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train if needed
        if self.config.index_type == "IVFFlat" and not self.index.is_trained:
            self.train_index(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        for chunk in chunks:
            self.chunk_ids.append(chunk['chunk_id'])
            self.metadata.append({
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'doc_type': chunk['doc_type'],
                'text': chunk['text'],
                'chunk_index': chunk['chunk_index'],
                **chunk.get('metadata', {})
            })
        
        self.logger.info(f"Added {len(chunks)} chunks to index (total: {len(self.chunk_ids)})")
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        if self.index is None or self.index.ntotal == 0:
            self.logger.warning("Index is empty")
            return []
        
        # Prepare query
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            
            metadata = self.metadata[idx].copy()
            
            # Apply filters
            if filters and not self._apply_filters(metadata, filters):
                continue
            
            results.append({
                'chunk_id': self.chunk_ids[idx],
                'score': float(distances[0][i]),
                'metadata': metadata
            })
            
            if len(results) >= top_k:
                break
        
        self.logger.debug(f"FAISS search returned {len(results)} results")
        return results
    
    def _apply_filters(self, metadata: Dict[str, Any], 
                      filters: Dict[str, Any]) -> bool:
        """Apply filters to metadata"""
        if 'categories' in filters:
            doc_categories = metadata.get('categories', [])
            if not any(cat in doc_categories for cat in filters['categories']):
                return False
        
        if 'doc_type' in filters:
            if metadata.get('doc_type') != filters['doc_type']:
                return False
        
        if 'min_relevance' in filters:
            if metadata.get('relevance_score', 0) < filters['min_relevance']:
                return False
        
        return True
    
    def save(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'chunk_ids': self.chunk_ids,
                    'metadata': self.metadata
                }, f)
            
            self.logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            self.logger.error(f"Error saving FAISS index: {e}")
    
    def load(self) -> bool:
        """
        Load index and metadata from disk
        
        Returns:
            True if loaded successfully
        """
        if not self.index_path.exists():
            self.logger.warning(f"Index not found at {self.index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.chunk_ids = data['chunk_ids']
                self.metadata = data['metadata']
            
            self.logger.info(
                f"Loaded FAISS index from {self.index_path} "
                f"({self.index.ntotal} vectors)"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if self.index is None:
            return {'status': 'not_initialized'}
        
        return {
            'index_type': self.config.index_type,
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
    
    def update(self, chunks: List[Dict[str, Any]]):
        """
        Update existing index with new chunks
        
        Args:
            chunks: List of new chunks with embeddings
        """
        if self.index is None:
            self.logger.warning("No existing index found. Creating new index...")
            self.create_index()
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train if needed (for IVF indexes)
        if self.config.index_type == "IVFFlat" and not self.index.is_trained:
            self.train_index(embeddings)
        
        # Add to existing index
        self.index.add(embeddings)
        
        # Append metadata
        for chunk in chunks:
            self.chunk_ids.append(chunk['chunk_id'])
            self.metadata.append({
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'doc_type': chunk['doc_type'],
                'text': chunk['text'],
                'chunk_index': chunk['chunk_index'],
                **chunk.get('metadata', {})
            })
        
        self.logger.info(f"Updated index with {len(chunks)} new chunks (total: {len(self.chunk_ids)})")