from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path
import logging
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from config.settings import config

logger = logging.getLogger(__name__)

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class BM25Index:
    """BM25 sparse retrieval index"""
    
    def __init__(self, index_name: str):
        self.config = config.bm25
        self.index_name = index_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.bm25 = None
        self.chunk_ids = []
        self.chunks_metadata = []
        
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Paths
        self.persist_dir = Path(self.config.persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.persist_dir / f"{index_name}_bm25.pkl"
        self.metadata_path = self.persist_dir / f"{index_name}_metadata.pkl"
        
        self.logger.info(f"Initialized BM25 index: {index_name}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 index from chunks
        
        Args:
            chunks: List of chunk dictionaries
        """
        self.logger.info(f"Building BM25 index for {len(chunks)} chunks")
        
        # Extract and tokenize texts
        tokenized_corpus = []
        self.chunk_ids = []
        self.chunks_metadata = []
        
        for chunk in chunks:
            # Tokenize text
            tokens = self.tokenize(chunk['text'])
            tokenized_corpus.append(tokens)
            
            # Store metadata
            self.chunk_ids.append(chunk['chunk_id'])
            self.chunks_metadata.append({
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'text': chunk['text'],
                **chunk.get('metadata', {})
            })
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.config.k1,
            b=self.config.b
        )
        
        self.logger.info(f"Built BM25 index with {len(self.chunk_ids)} documents")
    
    def search(self, query: str, top_k: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (categories, date, etc.)
            
        Returns:
            List of search results with scores
        """
        if self.bm25 is None:
            self.logger.error("BM25 index not built. Call build_index first.")
            return []
        
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            
            # Skip low scores
            if score < 0.1:
                continue
            
            metadata = self.chunks_metadata[idx].copy()
            
            # Apply filters
            if filters:
                if not self._apply_filters(metadata, filters):
                    continue
            
            results.append({
                'chunk_id': self.chunk_ids[idx],
                'score': score,
                'metadata': metadata
            })
            
            if len(results) >= top_k:
                break
        
        self.logger.debug(f"BM25 search returned {len(results)} results")
        return results
    
    def _apply_filters(self, metadata: Dict[str, Any], 
                      filters: Dict[str, Any]) -> bool:
        """
        Apply filters to a document
        
        Args:
            metadata: Document metadata
            filters: Filter criteria
            
        Returns:
            True if document passes filters
        """
        # Category filter
        if 'categories' in filters:
            doc_categories = metadata.get('categories', [])
            if not any(cat in doc_categories for cat in filters['categories']):
                return False
        
        # Document type filter
        if 'doc_type' in filters:
            if metadata.get('doc_type') != filters['doc_type']:
                return False
        
        # Minimum relevance filter
        if 'min_relevance' in filters:
            if metadata.get('relevance_score', 0) < filters['min_relevance']:
                return False
        
        return True
    
    def save(self):
        """Save BM25 index to disk"""
        try:
            # Save BM25 object
            with open(self.index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'chunk_ids': self.chunk_ids
                }, f)
            
            # Save metadata separately
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            
            self.logger.info(f"Saved BM25 index to {self.index_path}")
        except Exception as e:
            self.logger.error(f"Error saving BM25 index: {e}")
    
    def load(self) -> bool:
        """
        Load BM25 index from disk
        
        Returns:
            True if loaded successfully
        """
        if not self.index_path.exists():
            self.logger.warning(f"BM25 index not found at {self.index_path}")
            return False
        
        try:
            # Load BM25 object
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunk_ids = data['chunk_ids']
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            
            self.logger.info(f"Loaded BM25 index from {self.index_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading BM25 index: {e}")
            return False

    def update(self, chunks: List[Dict[str, Any]]):
        """
        Update existing BM25 index with new chunks
        
        Args:
            chunks: List of new chunk dictionaries
        """
        self.logger.info(f"Updating BM25 index with {len(chunks)} new chunks")
        
        # If no existing index, build from scratch
        if self.bm25 is None:
            self.logger.warning("No existing BM25 index found. Building new index...")
            self.build_index(chunks)
            return
        
        # Tokenize new documents
        new_tokenized_corpus = []
        for chunk in chunks:
            tokens = self.tokenize(chunk['text'])
            new_tokenized_corpus.append(tokens)
            
            # Store metadata
            self.chunk_ids.append(chunk['chunk_id'])
            self.chunks_metadata.append({
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'text': chunk['text'],
                **chunk.get('metadata', {})
            })
        
        # Rebuild BM25 with combined corpus (BM25 doesn't support incremental updates)
        # Get existing tokenized corpus
        existing_corpus = []
        if hasattr(self.bm25, 'corpus_size'):
            # Reconstruct corpus from existing data
            for i in range(self.bm25.corpus_size):
                # Re-tokenize existing documents
                existing_text = self.chunks_metadata[i]['text']
                existing_corpus.append(self.tokenize(existing_text))
        
        # Combine corpora
        combined_corpus = existing_corpus + new_tokenized_corpus
        
        # Rebuild BM25 index with combined corpus
        self.bm25 = BM25Okapi(
            combined_corpus,
            k1=self.config.k1,
            b=self.config.b
        )
        
        self.logger.info(f"Updated BM25 index. Total documents: {len(self.chunk_ids)}")