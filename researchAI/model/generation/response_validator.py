from typing import Dict, Any, List
import re
import logging

logger = logging.getLogger(__name__)

class ResponseValidator:
    """Validate generated responses"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate(self, query: str, 
                response: str, 
                retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive response validation
        
        Args:
            query: Original query
            response: Generated response
            retrieved_docs: Retrieved documents
            
        Returns:
            Validation results
        """
        checks = {
            'has_content': self._check_has_content(response),
            'has_citations': self._check_citations(response),
            'citation_validity': self._check_citation_validity(response, retrieved_docs),
            'relevance': self._check_relevance(query, response),
            'length': self._check_length(response),
            'structure': self._check_structure(response),
            'no_hallucination_indicators': self._check_hallucination_indicators(response)
        }
        
        checks['overall_score'] = self._calculate_overall_score(checks)
        checks['is_valid'] = checks['overall_score'] >= 0.7
        
        return checks
    
    def _check_has_content(self, response: str) -> Dict[str, Any]:
        """Check if response has meaningful content"""
        has_content = len(response.strip()) > 50
        return {
            'passed': has_content,
            'length': len(response),
            'score': 1.0 if has_content else 0.0
        }
    
    def _check_citations(self, response: str) -> Dict[str, Any]:
        """Check if response includes citations"""
        citation_pattern = r'\[\d+\]'
        citations = re.findall(citation_pattern, response)
        
        has_citations = len(citations) > 0
        return {
            'passed': has_citations,
            'count': len(citations),
            'score': min(len(citations) / 3.0, 1.0)
        }
    
    def _check_citation_validity(self, response: str, 
                                 retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if all citations are valid"""
        citation_pattern = r'\[(\d+)\]'
        cited_numbers = set(int(n) for n in re.findall(citation_pattern, response))
        
        valid_range = set(range(1, len(retrieved_docs) + 1))
        invalid_citations = cited_numbers - valid_range
        
        is_valid = len(invalid_citations) == 0
        return {
            'passed': is_valid,
            'invalid_citations': list(invalid_citations),
            'score': 1.0 if is_valid else 0.5
        }
    
    def _check_relevance(self, query: str, response: str) -> Dict[str, Any]:
        """Check if response is relevant to query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'is', 'are', 'was', 'were'}
        query_words -= common_words
        response_words -= common_words
        
        if not query_words:
            return {'passed': True, 'score': 1.0}
        
        overlap = len(query_words & response_words)
        overlap_ratio = overlap / len(query_words)
        
        is_relevant = overlap_ratio >= 0.3
        return {
            'passed': is_relevant,
            'overlap_ratio': overlap_ratio,
            'score': min(overlap_ratio * 2, 1.0)
        }
    
    def _check_length(self, response: str) -> Dict[str, Any]:
        """Check if response length is appropriate"""
        length = len(response)
        
        if 200 <= length <= 2000:
            score = 1.0
        elif length < 200:
            score = length / 200.0
        else:
            score = max(0.7, 2000 / length)
        
        return {
            'passed': 200 <= length <= 2000,
            'length': length,
            'score': score
        }
    
    def _check_structure(self, response: str) -> Dict[str, Any]:
        """Check if response has good structure"""
        paragraphs = response.split('\n\n')
        has_paragraphs = len(paragraphs) > 1
        
        has_lists = bool(re.search(r'(^\d+\.|^-|^•)', response, re.MULTILINE))
        
        score = 0.5
        if has_paragraphs:
            score += 0.3
        if has_lists:
            score += 0.2
        
        return {
            'passed': has_paragraphs,
            'has_paragraphs': has_paragraphs,
            'has_lists': has_lists,
            'score': min(score, 1.0)
        }
    
    def _check_hallucination_indicators(self, response: str) -> Dict[str, Any]:
        """Check for common hallucination indicators"""
        hallucination_phrases = [
            "i don't have access",
            "i cannot find",
            "based on my training data",
            "as of my last update",
            "i apologize, but"
        ]
        
        response_lower = response.lower()
        found_indicators = [
            phrase for phrase in hallucination_phrases 
            if phrase in response_lower
        ]
        
        has_uncertainty_acknowledgment = len(found_indicators) > 0
        
        return {
            'passed': True,  
            'indicators_found': found_indicators,
            'acknowledges_uncertainty': has_uncertainty_acknowledgment,
            'score': 1.0
        }
    
    def _calculate_overall_score(self, checks: Dict[str, Any]) -> float:
        """Calculate overall validation score"""
        weights = {
            'has_content': 0.3,
            'has_citations': 0.2,
            'citation_validity': 0.15,
            'relevance': 0.2,
            'length': 0.05,
            'structure': 0.1
        }
        
        total_score = 0.0
        for key, weight in weights.items():
            if key in checks:
                total_score += checks[key]['score'] * weight
        
        return total_score