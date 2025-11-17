
from typing import List, Dict, Any
import numpy as np
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class RAGMetrics:
    """Calculate metrics for RAG system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_retrieval_metrics(self, 
                                    retrieved_docs: List[Dict[str, Any]],
                                    relevant_doc_ids: List[str] = None) -> Dict[str, float]:
        """
        Calculate retrieval quality metrics
        
        Args:
            retrieved_docs: Retrieved documents
            relevant_doc_ids: Ground truth relevant document IDs (if available)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        metrics['num_retrieved'] = len(retrieved_docs)
        
        if retrieved_docs:
            scores = [doc['score'] for doc in retrieved_docs]
            metrics['avg_score'] = np.mean(scores)
            metrics['max_score'] = np.max(scores)
            metrics['min_score'] = np.min(scores)
            metrics['score_std'] = np.std(scores)
            
            diversity = self._calculate_diversity(retrieved_docs)
            metrics.update(diversity)
            
            if relevant_doc_ids:
                retrieved_ids = [doc['metadata']['doc_id'] for doc in retrieved_docs]
                precision_recall = self._calculate_precision_recall(
                    retrieved_ids, 
                    relevant_doc_ids
                )
                metrics.update(precision_recall)
        
        return metrics
    
    def calculate_generation_metrics(self, 
                                    response: str,
                                    ground_truth: str = None) -> Dict[str, float]:
        """
        Calculate generation quality metrics
        
        Args:
            response: Generated response
            ground_truth: Reference response (if available)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        metrics['response_length'] = len(response)
        metrics['num_sentences'] = response.count('.') + response.count('!') + response.count('?')
        metrics['avg_sentence_length'] = metrics['response_length'] / max(metrics['num_sentences'], 1)
        
        import re
        citations = re.findall(r'\[\d+\]', response)
        metrics['num_citations'] = len(citations)
        metrics['unique_citations'] = len(set(citations))
        
        if ground_truth:
            similarity = self._calculate_text_similarity(response, ground_truth)
            metrics['similarity_score'] = similarity
        
        return metrics
    
    def calculate_end_to_end_metrics(self,
                                     query: str,
                                     response: str,
                                     retrieved_docs: List[Dict[str, Any]],
                                     response_time: float) -> Dict[str, Any]:
        """
        Calculate end-to-end system metrics
        
        Args:
            query: User query
            response: Generated response
            retrieved_docs: Retrieved documents
            response_time: Total response time in seconds
            
        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {
            'query_length': len(query),
            'response_time_seconds': response_time,
            'retrieval_metrics': self.calculate_retrieval_metrics(retrieved_docs),
            'generation_metrics': self.calculate_generation_metrics(response)
        }
        
        metrics['tokens_per_second'] = len(response.split()) / max(response_time, 0.1)
        
        return metrics
    
    def _calculate_diversity(self, docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate diversity metrics"""
        sources = []
        categories = []
        
        for doc in docs:
            meta = doc['metadata']
            source = meta.get('source_name') or meta.get('arxiv_id', 'unknown')
            sources.append(source)
            categories.extend(meta.get('categories', []))
        
        unique_sources = len(set(sources))
        unique_categories = len(set(categories))
        
        source_counts = Counter(sources)
        total = len(sources)
        
        import math
        if total > 0 and len(source_counts) > 1:
            entropy = -sum(
                (count / total) * math.log2(count / total)
                for count in source_counts.values()
            )
            max_entropy = math.log2(len(source_counts))
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return {
            'source_diversity': unique_sources,
            'category_diversity': unique_categories,
            'source_entropy': normalized_entropy
        }
    
    def _calculate_precision_recall(self, 
                                    retrieved_ids: List[str],
                                    relevant_ids: List[str]) -> Dict[str, float]:
        """Calculate precision and recall"""
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        true_positives = len(retrieved_set & relevant_set)
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(relevant_set) if relevant_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0