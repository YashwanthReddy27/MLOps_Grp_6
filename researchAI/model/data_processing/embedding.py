from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging

from config.settings import config

logger = logging.getLogger(__name__)


class HybridEmbedder:
    """Generate hybrid embeddings for documents"""
    
    def __init__(self):
        self.config = config.embedding
        self.logger = logging.getLogger(__name__)
        
        # Load embedding model
        self.logger.info(f"Loading embedding model: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True
        )
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings
    
    def embed_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced embedding for a chunk
        
        Args:
            chunk: Chunk object
            
        Returns:
            Chunk with embedding added
        """
        # Main content embedding
        content_emb = self.embed_text(chunk['text'])
        
        # Category embedding (for semantic boost)
        categories = chunk['metadata'].get('categories', [])
        if categories:
            category_text = " ".join(categories[:3])  # Top 3 categories
            category_emb = self.embed_text(category_text)
            
            # Combine: 70% content, 30% category
            hybrid_emb = 0.7 * content_emb + 0.3 * category_emb
        else:
            hybrid_emb = content_emb
        
        # Normalize if needed
        if self.config.normalize:
            hybrid_emb = hybrid_emb / np.linalg.norm(hybrid_emb)
        
        chunk['embedding'] = hybrid_emb.tolist()
        return chunk
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple chunks
        
        Args:
            chunks: List of chunk objects
            
        Returns:
            Chunks with embeddings added
        """
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate content embeddings in batch
        content_embeddings = self.embed_batch(texts)
        
        # Process category embeddings and combine
        for idx, chunk in enumerate(chunks):
            categories = chunk['metadata'].get('categories', [])
            
            if categories:
                category_text = " ".join(categories[:3])
                category_emb = self.embed_text(category_text)
                
                # Combine embeddings -> hybrid embeddings
                hybrid_emb = 0.7 * content_embeddings[idx] + 0.3 * category_emb
                
                # Normalize
                if self.config.normalize:
                    hybrid_emb = hybrid_emb / np.linalg.norm(hybrid_emb)
                
                chunk['embedding'] = hybrid_emb.tolist()
            else:
                chunk['embedding'] = content_embeddings[idx].tolist()
        
        self.logger.info(f"Generated {len(chunks)} embeddings")
        return chunks
    
    def get_query_embedding(self, query: str, 
                           query_context: Dict[str, Any] = None) -> np.ndarray:
        """
        Generate embedding for a query with optional context
        
        Args:
            query: Query text
            query_context: Optional context (e.g., user preferences, filters)
            
        Returns:
            Query embedding
        """
        # Enhance query with context if available
        if query_context:
            categories = query_context.get('categories', [])
            if categories:
                # Add category hint to query
                enhanced_query = f"{query} {' '.join(categories)}"
                return self.embed_text(enhanced_query)
        
        return self.embed_text(query)