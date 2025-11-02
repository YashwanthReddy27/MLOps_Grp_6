from typing import List, Dict, Any
import logging
from sentence_transformers import CrossEncoder
from config.settings import config
logger = logging.getLogger(__name__)

class Reranker:
    """Cross-encoder based reranking"""
    
    def __init__(self):
        self.config = config.retrieval
        self.logger = logging.getLogger(__name__)
        
        # Load cross-encoder
        self.logger.info(f"Loading reranker: {self.config.reranker_model}")
        self.model = CrossEncoder(self.config.reranker_model)
        self.logger.info("Reranker loaded")
    
    def rerank(self, query: str, 
               candidates: List[Dict[str, Any]], 
               top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder
        
        Args:
            query: Query text
            candidates: List of candidate results
            top_k: Number of top results to return
            
        Returns:
            Reranked results
        """
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [
            [query, candidate['metadata']['text']] 
            for candidate in candidates
        ]
        
        # Get reranking scores
        rerank_scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Combine with original scores
        for i, candidate in enumerate(candidates):
            original_score = candidate['score']
            rerank_score = float(rerank_scores[i])
            
            combined_score = (
                self.config.rerank_weight * rerank_score +
                self.config.original_score_weight * original_score
            )
            
            candidate['original_score'] = original_score
            candidate['rerank_score'] = rerank_score
            candidate['score'] = combined_score
        
        # Sort by combined score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        self.logger.debug(f"Reranked {len(candidates)} candidates, returning top {top_k}")
        return candidates[:top_k]