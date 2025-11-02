from typing import List, Dict, Any, Optional
import logging

from data_processing.embedding import HybridEmbedder
from retrieval.vector_store import FAISSVectorStore
from retrieval.bm25_index import BM25Index
from retrieval.reranker import Reranker
from config.settings import config

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining FAISS (dense) and BM25 (sparse)"""
    
    def __init__(self, embedder: HybridEmbedder):
        self.embedder = embedder
        self.config = config.retrieval
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector stores
        self.paper_dense = FAISSVectorStore(config.vector_store.papers_index_name)
        self.news_dense = FAISSVectorStore(config.vector_store.news_index_name)
        
        # Initialize BM25 indexes
        self.paper_sparse = BM25Index(config.vector_store.papers_index_name)
        self.news_sparse = BM25Index(config.vector_store.news_index_name)
        
        # Initialize reranker
        self.reranker = Reranker()
        
        self.logger.info("Initialized Hybrid Retriever (FAISS + BM25)")
    
    def index_documents(self, chunks: List[Dict[str, Any]], doc_type: str):
        """
        Index documents in both FAISS and BM25
        
        Args:
            chunks: List of chunks with embeddings
            doc_type: 'paper' or 'news'
        """
        self.logger.info(f"Indexing {len(chunks)} {doc_type} chunks")
        
        # Select appropriate indexes
        if doc_type == 'paper':
            dense_index = self.paper_dense
            sparse_index = self.paper_sparse
        else:
            dense_index = self.news_dense
            sparse_index = self.news_sparse
        
        # Add to FAISS
        dense_index.add(chunks)
        
        # Build BM25
        sparse_index.build_index(chunks)
        
        self.logger.info(f"Indexed {len(chunks)} {doc_type} chunks")
    
    def save_indexes(self):
        """Save all indexes to disk"""
        self.logger.info("Saving indexes...")
        
        self.paper_dense.save()
        self.news_dense.save()
        self.paper_sparse.save()
        self.news_sparse.save()
        
        self.logger.info("All indexes saved")
    
    def load_indexes(self) -> bool:
        """
        Load all indexes from disk
        
        Returns:
            True if all indexes loaded successfully
        """
        self.logger.info("Loading indexes...")
        
        success = True
        success &= self.paper_dense.load()
        success &= self.news_dense.load()
        success &= self.paper_sparse.load()
        success &= self.news_sparse.load()
        
        if success:
            self.logger.info("All indexes loaded successfully")
        else:
            self.logger.warning("Some indexes failed to load")
        
        return success
    
    def retrieve(self, query: str, 
                 top_k: Optional[int] = None,
                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval pipeline
        
        Args:
            query: User query
            top_k: Number of results to return
            filters: Search filters
            
        Returns:
            List of retrieved and reranked results
        """
        top_k = top_k or self.config.top_k
        
        self.logger.info(f"Retrieving for query: {query}")
        
        # Stage 1: Dense retrieval (FAISS)
        dense_results = self._dense_search(query, top_k, filters)
        
        # Stage 2: Sparse retrieval (BM25)
        sparse_results = self._sparse_search(query, top_k, filters)
        
        # Stage 3: Hybrid fusion
        fused_results = self._fuse_results(dense_results, sparse_results, top_k * 2)
        
        if not fused_results:
            self.logger.warning("No results found")
            return []
        
        # Stage 4: Reranking
        reranked = self.reranker.rerank(
            query=query,
            candidates=fused_results,
            top_k=self.config.rerank_top_k
        )
        
        # Stage 5: Diversity filtering
        diverse_results = self._diversify(reranked, self.config.diversity_top_k)
        
        self.logger.info(f"Retrieved {len(diverse_results)} results")
        return diverse_results
    
    def _dense_search(self, query: str, top_k: int,
                     filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dense retrieval using FAISS"""
        # Generate query embedding
        query_embedding = self.embedder.get_query_embedding(query)
        
        results = []
        
        # Search papers
        try:
            paper_results = self.paper_dense.search(
                query_embedding=query_embedding,
                top_k=top_k // 2,
                filters=filters
            )
            for result in paper_results:
                result['retrieval_method'] = 'dense'
                result['index_type'] = 'papers'
            results.extend(paper_results)
        except Exception as e:
            self.logger.error(f"Error in dense search (papers): {e}")
        
        # Search news
        try:
            news_results = self.news_dense.search(
                query_embedding=query_embedding,
                top_k=top_k // 2,
                filters=filters
            )
            for result in news_results:
                result['retrieval_method'] = 'dense'
                result['index_type'] = 'news'
            results.extend(news_results)
        except Exception as e:
            self.logger.error(f"Error in dense search (news): {e}")
        
        self.logger.debug(f"Dense search returned {len(results)} results")
        return results
    
    def _sparse_search(self, query: str, top_k: int,
                      filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sparse retrieval using BM25"""
        results = []
        
        # Search papers
        try:
            paper_results = self.paper_sparse.search(
                query=query,
                top_k=top_k // 2,
                filters=filters
            )
            for result in paper_results:
                result['retrieval_method'] = 'sparse'
                result['index_type'] = 'papers'
            results.extend(paper_results)
        except Exception as e:
            self.logger.error(f"Error in sparse search (papers): {e}")
        
        # Search news
        try:
            news_results = self.news_sparse.search(
                query=query,
                top_k=top_k // 2,
                filters=filters
            )
            for result in news_results:
                result['retrieval_method'] = 'sparse'
                result['index_type'] = 'news'
            results.extend(news_results)
        except Exception as e:
            self.logger.error(f"Error in sparse search (news): {e}")
        
        self.logger.debug(f"Sparse search returned {len(results)} results")
        return results
    
    def _fuse_results(self, dense_results: List[Dict[str, Any]], 
                     sparse_results: List[Dict[str, Any]],
                     top_k: int) -> List[Dict[str, Any]]:
        """Fuse dense and sparse results using reciprocal rank fusion"""
        fused_scores = {}
        k = 60 
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result['chunk_id']
            rrf_score = 1.0 / (k + rank + 1)
            
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    'result': result,
                    'dense_score': result['score'],
                    'sparse_score': 0.0,
                    'dense_rank': rank + 1,
                    'sparse_rank': None,
                    'rrf_score': 0.0
                }
            
            fused_scores[chunk_id]['rrf_score'] += rrf_score * self.config.dense_weight
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            chunk_id = result['chunk_id']
            rrf_score = 1.0 / (k + rank + 1)
            
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    'result': result,
                    'dense_score': 0.0,
                    'sparse_score': result['score'],
                    'dense_rank': None,
                    'sparse_rank': rank + 1,
                    'rrf_score': 0.0
                }
            else:
                fused_scores[chunk_id]['sparse_score'] = result['score']
                fused_scores[chunk_id]['sparse_rank'] = rank + 1
            
            fused_scores[chunk_id]['rrf_score'] += rrf_score * self.config.sparse_weight
        
        # Sort by fused score
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )[:top_k]
        
        # Format results
        final_results = []
        for item in sorted_results:
            result = item['result'].copy()
            result['score'] = item['rrf_score']
            result['fusion_details'] = {
                'dense_score': item['dense_score'],
                'sparse_score': item['sparse_score'],
                'dense_rank': item['dense_rank'],
                'sparse_rank': item['sparse_rank']
            }
            final_results.append(result)
        
        self.logger.debug(
            f"Fused {len(dense_results)} dense and {len(sparse_results)} sparse "
            f"results into {len(final_results)} final results"
        )
        return final_results
    
    def _diversify(self, results: List[Dict[str, Any]], 
                   top_k: int) -> List[Dict[str, Any]]:
        """Ensure diversity in sources and categories"""
        diverse_results = []
        seen_sources = set()
        seen_categories = set()
        
        for result in results:
            metadata = result['metadata']
            
            # Get source identifier
            source = metadata.get('source_name') or metadata.get('arxiv_id', '')
            # Get categories
            categories = set(metadata.get('categories', []))
            
            # Add if from new source or has new category
            is_new_source = source not in seen_sources
            has_new_category = len(categories - seen_categories) > 0
            
            if is_new_source or has_new_category:
                diverse_results.append(result)
                seen_sources.add(source)
                seen_categories.update(categories)
                
                if len(diverse_results) >= top_k:
                    break
        
        # Fill remaining slots if needed, without checking if they are contributing to the diversity
        if len(diverse_results) < top_k:
            for result in results:
                if result not in diverse_results:
                    diverse_results.append(result)
                    if len(diverse_results) >= top_k:
                        break
        
        self.logger.debug(
            f"Diversified to {len(diverse_results)} results "
            f"({len(seen_sources)} sources, {len(seen_categories)} categories)"
        )
        return diverse_results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for all indexes"""
        return {
            'papers': {
                'dense': self.paper_dense.get_stats(),
                'sparse': {
                    'total_documents': len(self.paper_sparse.chunk_ids)
                }
            },
            'news': {
                'dense': self.news_dense.get_stats(),
                'sparse': {
                    'total_documents': len(self.news_sparse.chunk_ids)
                }
            }
        }