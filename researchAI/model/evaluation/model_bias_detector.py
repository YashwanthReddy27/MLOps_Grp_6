from typing import List, Dict, Any
import numpy as np
import logging
from datetime import datetime
import warnings

try:
    from fairlearn.metrics import (
        demographic_parity_difference, demographic_parity_ratio,
        selection_rate, MetricFrame
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("FairLearn not installed. Install with: pip install fairlearn")
    FAIRLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

SLICE_DEFINITIONS = {
    'query_complexity': {
        'simple': ['what is', 'define', 'explain', 'how to'],
        'intermediate': ['compare', 'analyze', 'evaluate', 'discuss'],
        'advanced': ['synthesize', 'critique', 'predict', 'optimize']
    },
    'domain_type': {
        'ai_ml': ['artificial intelligence', 'machine learning', 'deep learning', 'neural'],
        'cybersecurity': ['security', 'cyber', 'attack', 'vulnerability', 'encryption'],
        'cloud_computing': ['cloud', 'aws', 'azure', 'kubernetes', 'docker'],
        'web3_blockchain': ['blockchain', 'cryptocurrency', 'web3', 'defi', 'nft'],
        'robotics': ['robot', 'automation', 'autonomous', 'sensor', 'actuator']
    },
    'user_experience': {
        'beginner': ['basic', 'introduction', 'getting started', 'tutorial'],
        'intermediate': ['implementation', 'best practices', 'optimization'],
        'expert': ['advanced', 'research', 'cutting-edge', 'state-of-the-art']
    },
    'geographic_context': {
        'north_america': ['usa', 'canada', 'american', 'silicon valley'],
        'europe': ['european', 'eu', 'germany', 'uk', 'france'],
        'asia_pacific': ['china', 'japan', 'india', 'singapore', 'australia'],
        'global': ['international', 'worldwide', 'global', 'cross-border']
    },
    'source_type': {
        'academic': ['research', 'paper', 'study', 'journal', 'conference'],
        'industry': ['company', 'product', 'commercial', 'enterprise'],
        'news': ['news', 'article', 'report', 'announcement'],
        'community': ['blog', 'forum', 'discussion', 'open source']
    }
}

class RAGBiasDetector:
    """
    Bias detection for RAG systems using slicing techniques and FairLearn.
    
    Evaluates bias across query complexity, domain type, user experience,
    geographic context, and source type dimensions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.slice_definitions = SLICE_DEFINITIONS

    def _analyze_retrieval_diversity(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze diversity of retrieved documents
        
        Returns diversity metrics for sources, categories, and temporal distribution
        """
        if not retrieved_docs:
            return {
                'num_unique_sources': 0,
                'num_unique_categories': 0,
                'source_diversity_ratio': 0.0,
                'category_diversity_ratio': 0.0,
                'source_distribution': {},
                'category_distribution': {}
            }
        
        sources = []
        categories = []
        
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            
            source = metadata.get('source_name') or metadata.get('arxiv_id', 'unknown')
            sources.append(source)
            
            doc_categories = metadata.get('categories', [])
            categories.extend(doc_categories)
        
        unique_sources = set(sources)
        unique_categories = set(categories)
        
        source_diversity_ratio = len(unique_sources) / len(retrieved_docs)
        category_diversity_ratio = len(unique_categories) / max(len(retrieved_docs), 1)
        
        from collections import Counter
        source_counts = Counter(sources)
        source_distribution = {src: count/len(sources) for src, count in source_counts.items()}
        
        category_counts = Counter(categories)
        category_distribution = {cat: count/len(categories) for cat, count in category_counts.items()} if categories else {}
        
        return {
            'num_unique_sources': len(unique_sources),
            'num_unique_categories': len(unique_categories),
            'source_diversity_ratio': source_diversity_ratio,
            'category_diversity_ratio': category_diversity_ratio,
            'source_distribution': source_distribution,
            'category_distribution': category_distribution,
            'total_docs': len(retrieved_docs)
        }

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query characteristics for context in fairness evaluation
        """
        query_lower = query.lower()
        
        complexity = 'unknown'
        for complexity_level, keywords in self.slice_definitions['query_complexity'].items():
            if any(keyword in query_lower for keyword in keywords):
                complexity = complexity_level
                break
        
        domain = 'unknown'
        for domain_type, keywords in self.slice_definitions['domain_type'].items():
            if any(keyword in query_lower for keyword in keywords):
                domain = domain_type
                break
        
        experience = 'unknown'
        for exp_level, keywords in self.slice_definitions['user_experience'].items():
            if any(keyword in query_lower for keyword in keywords):
                experience = exp_level
                break
        
        return {
            'complexity': complexity,
            'domain': domain,
            'experience_level': experience
        }

    def _generate_fairness_warnings(self, 
                                    indicators: Dict[str, float],
                                    diversity_metrics: Dict[str, Any],
                                    query_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate warnings for potential fairness issues
        """
        warnings = []
        
        if indicators['source_diversity_score'] < 0.3:
            warnings.append({
                'type': 'low_source_diversity',
                'severity': 'medium',
                'message': f"Low source diversity: Only {diversity_metrics['num_unique_sources']} unique sources in {diversity_metrics['total_docs']} documents",
                'recommendation': "Consider retrieving from more diverse sources"
            })
        
        if diversity_metrics['source_distribution']:
            max_source_ratio = max(diversity_metrics['source_distribution'].values())
            if max_source_ratio > 0.7:
                warnings.append({
                    'type': 'source_dominance',
                    'severity': 'high',
                    'message': f"Single source dominates with {max_source_ratio*100:.1f}% of results",
                    'recommendation': "Diversify retrieval to include more sources"
                })
        
        if indicators['retrieval_quality_score'] < 0.4:
            warnings.append({
                'type': 'low_retrieval_quality',
                'severity': 'medium',
                'message': f"Low retrieval quality score: {indicators['retrieval_quality_score']:.2f}",
                'recommendation': "Retrieved documents may not be highly relevant to the query"
            })
        
        if indicators['response_quality_score'] < 0.5:
            warnings.append({
                'type': 'low_response_quality',
                'severity': 'high',
                'message': f"Low response quality score: {indicators['response_quality_score']:.2f}",
                'recommendation': "Response validation indicates potential issues with generated answer"
            })
        
        return warnings
    
    def evaluate_single_query_fairness_with_fairlearn(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single-query fairness evaluation using Fairlearn metrics.
        
        Uses a synthetic approach where we create a small evaluation dataset by:
        1. Analyzing retrieved documents as individual samples
        2. Treating document sources/categories as sensitive features
        3. Using Fairlearn to measure demographic parity across these groups
        
        Args:
            evaluation_data: Single query evaluation data
                
        Returns:
            Fairness report with Fairlearn metrics
        """
        self.logger.info("Running single-query fairness evaluation with Fairlearn")
        
        if not FAIRLEARN_AVAILABLE:
            self.logger.warning("Fairlearn not available. Falling back to basic fairness evaluation.")
            return self.evaluate_single_query_fairness(evaluation_data)
        
        query = evaluation_data.get('query', '')
        retrieved_docs = evaluation_data.get('retrieved_docs', [])
        response = evaluation_data.get('response', '')
        perf_metrics = evaluation_data.get('performance_metrics', {})
        
        diversity_metrics = self._analyze_retrieval_diversity(retrieved_docs)
        
        query_characteristics = self._classify_query(query)
        
        fairlearn_results = self._fairlearn_single_query_analysis(
            retrieved_docs, 
            query_characteristics
        )
        
        retrieval_score = perf_metrics.get('retrieval_score', 0.0)
        response_quality = perf_metrics.get('response_quality', 0.0)
        
        fairness_indicators = {
            'source_diversity_score': diversity_metrics['source_diversity_ratio'],
            'category_diversity_score': diversity_metrics['category_diversity_ratio'],
            'retrieval_quality_score': retrieval_score,
            'response_quality_score': response_quality,
        }
        
        if fairlearn_results and 'error' not in fairlearn_results:
            fairlearn_fairness_score = self._calculate_fairlearn_fairness_score(
                fairlearn_results
            )
            fairness_indicators['fairlearn_fairness_score'] = fairlearn_fairness_score
        else:
            fairness_indicators['fairlearn_fairness_score'] = 0.5  
        
        overall_fairness = self._calculate_single_query_fairness_score_with_fairlearn(
            fairness_indicators
        )
        
        warnings = self._generate_fairness_warnings_with_fairlearn(
            fairness_indicators,
            diversity_metrics,
            query_characteristics,
            fairlearn_results
        )
        
        return {
            'overall_fairness_score': overall_fairness,
            'query_characteristics': query_characteristics,
            'diversity_metrics': diversity_metrics,
            'fairness_indicators': fairness_indicators,
            'fairlearn_analysis': fairlearn_results,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'single_query_fairlearn'
        }

    def _fairlearn_single_query_analysis(self, 
                                        retrieved_docs: List[Dict[str, Any]],
                                        query_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Fairlearn analysis on retrieved documents for a single query.
        
        Strategy:
        - Treat each retrieved document as a sample
        - Use document source and category as sensitive features
        - Use retrieval rank/score as the outcome to measure
        - Check if high-quality results are equitably distributed across sources
        
        Returns:
            Fairlearn metrics including demographic parity
        """
        if not retrieved_docs or len(retrieved_docs) < 2:
            return {'error': 'Insufficient documents for Fairlearn analysis'}
        
        try:
            sources = []
            categories = []
            scores = []
            high_quality_flags = [] 
            all_scores = [doc.get('score', 0.0) for doc in retrieved_docs]
            if not all_scores:
                return {'error': 'No scores available'}
            
            median_score = np.median(all_scores)
            
            for doc in retrieved_docs:
                metadata = doc.get('metadata', {})
                score = doc.get('score', 0.0)
                
                source = metadata.get('source_name') or metadata.get('source', 'unknown')
                sources.append(source)
                
                doc_categories = metadata.get('categories', [])
                category = doc_categories[0] if doc_categories else 'unknown'
                categories.append(category)
                
                scores.append(score)
                
                high_quality = 1 if score >= median_score else 0
                high_quality_flags.append(high_quality)
            
            y_true = np.array(high_quality_flags)
            y_pred = np.array(high_quality_flags)  
            source_fairness = self._analyze_fairlearn_dimension(
                y_true, y_pred, sources, 'source'
            )
            
            category_fairness = self._analyze_fairlearn_dimension(
                y_true, y_pred, categories, 'category'
            )
            
            fairlearn_summary = {
                'source_analysis': source_fairness,
                'category_analysis': category_fairness,
                'num_documents_analyzed': len(retrieved_docs),
                'median_score': float(median_score),
                'score_distribution': {
                    'min': float(min(all_scores)),
                    'max': float(max(all_scores)),
                    'mean': float(np.mean(all_scores)),
                    'std': float(np.std(all_scores))
                }
            }
            
            return fairlearn_summary
            
        except Exception as e:
            self.logger.error(f"Fairlearn single-query analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_fairlearn_dimension(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    sensitive_features: List[str],
                                    dimension_name: str) -> Dict[str, Any]:
        """
        Analyze fairness along a specific dimension (source or category) using Fairlearn
        
        Returns:
            Dictionary with demographic parity metrics
        """
        try:
            unique_groups = set(sensitive_features)
            if len(unique_groups) < 2:
                return {
                    'warning': f'Only one unique {dimension_name} group found',
                    'unique_groups': len(unique_groups),
                    'groups': list(unique_groups)
                }
            
            mf = MetricFrame(
                metrics={
                    'selection_rate': selection_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=np.array(sensitive_features)
            )
            
            dp_diff = demographic_parity_difference(
                y_true, y_pred, 
                sensitive_features=np.array(sensitive_features)
            )
            
            dp_ratio = demographic_parity_ratio(
                y_true, y_pred,
                sensitive_features=np.array(sensitive_features)
            )
            
            by_group = mf.by_group.to_dict()
            
            group_rates = by_group.get('selection_rate', {})
            if group_rates:
                max_group = max(group_rates, key=group_rates.get)
                min_group = min(group_rates, key=group_rates.get)
            else:
                max_group = min_group = 'unknown'
            
            return {
                'demographic_parity_difference': float(dp_diff),
                'demographic_parity_ratio': float(dp_ratio) if not np.isnan(dp_ratio) else 0.0,
                'overall_selection_rate': float(mf.overall['selection_rate']),
                'by_group_selection_rates': {k: float(v) for k, v in group_rates.items()},
                'advantaged_group': max_group,
                'disadvantaged_group': min_group,
                'num_groups': len(unique_groups),
                'groups': list(unique_groups),
                'fairness_assessment': self._assess_fairlearn_fairness(dp_diff, dp_ratio)
            }
            
        except Exception as e:
            self.logger.warning(f"Fairlearn analysis for {dimension_name} failed: {e}")
            return {'error': str(e)}

    def _assess_fairlearn_fairness(self, dp_diff: float, dp_ratio: float) -> str:
        """
        Assess fairness based on Fairlearn demographic parity metrics
        
        Demographic Parity Difference (DP Diff): 
            - Closer to 0 is better (ideal = 0)
            - Range: [0, 1]
            - Threshold: < 0.1 is fair, > 0.2 is unfair
        
        Demographic Parity Ratio (DP Ratio):
            - Closer to 1 is better (ideal = 1)
            - Range: [0, 1]
            - Threshold: > 0.8 is fair, < 0.6 is unfair
        """
        if abs(dp_diff) < 0.1 and (np.isnan(dp_ratio) or dp_ratio > 0.8):
            return 'FAIR'
        elif abs(dp_diff) < 0.2 and (np.isnan(dp_ratio) or dp_ratio > 0.6):
            return 'MODERATE'
        else:
            return 'UNFAIR'

    def _calculate_fairlearn_fairness_score(self, fairlearn_results: Dict[str, Any]) -> float:
        """
        Calculate a fairness score [0, 1] from Fairlearn results
        
        Higher is better (1.0 = perfectly fair)
        """
        if 'error' in fairlearn_results:
            return 0.5  
        
        scores = []
        
        source_analysis = fairlearn_results.get('source_analysis', {})
        if 'demographic_parity_difference' in source_analysis:
            dp_diff = abs(source_analysis['demographic_parity_difference'])
            source_score = max(0.0, 1.0 - dp_diff)
            scores.append(source_score)
        
        category_analysis = fairlearn_results.get('category_analysis', {})
        if 'demographic_parity_difference' in category_analysis:
            dp_diff = abs(category_analysis['demographic_parity_difference'])
            category_score = max(0.0, 1.0 - dp_diff)
            scores.append(category_score)
        
        if not scores:
            return 0.5
        
        return np.mean(scores)

    def _calculate_single_query_fairness_score_with_fairlearn(self, 
                                                            indicators: Dict[str, float]) -> float:
        """
        Calculate overall fairness score including Fairlearn metrics
        
        Weighted combination:
        - Source diversity (20%)
        - Category diversity (15%)
        - Retrieval quality (20%)
        - Response quality (20%)
        - Fairlearn fairness (25%) - NEW!
        """
        weights = {
            'source_diversity_score': 0.20,
            'category_diversity_score': 0.15,
            'retrieval_quality_score': 0.20,
            'response_quality_score': 0.20,
            'fairlearn_fairness_score': 0.25
        }
        
        fairness_score = sum(
            indicators.get(key, 0.0) * weight 
            for key, weight in weights.items()
        )
        
        return min(1.0, max(0.0, fairness_score))

    def _generate_fairness_warnings_with_fairlearn(self,
                                                indicators: Dict[str, float],
                                                diversity_metrics: Dict[str, Any],
                                                query_characteristics: Dict[str, Any],
                                                fairlearn_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate warnings including Fairlearn-based insights
        """
        warnings = []
        
        warnings.extend(self._generate_fairness_warnings(
            indicators, diversity_metrics, query_characteristics
        ))
        
        if fairlearn_results and 'error' not in fairlearn_results:
            
            source_analysis = fairlearn_results.get('source_analysis', {})
            if 'fairness_assessment' in source_analysis:
                assessment = source_analysis['fairness_assessment']
                
                if assessment == 'UNFAIR':
                    dp_diff = source_analysis.get('demographic_parity_difference', 0)
                    disadvantaged = source_analysis.get('disadvantaged_group', 'unknown')
                    advantaged = source_analysis.get('advantaged_group', 'unknown')
                    
                    warnings.append({
                        'type': 'fairlearn_source_bias',
                        'severity': 'high',
                        'message': f"Fairlearn detected unfair source distribution (DP Diff: {dp_diff:.3f}). "
                                f"Source '{advantaged}' is over-represented vs '{disadvantaged}'",
                        'recommendation': "Adjust retrieval to balance representation across sources"
                    })
                
                elif assessment == 'MODERATE':
                    warnings.append({
                        'type': 'fairlearn_source_concern',
                        'severity': 'medium',
                        'message': "Fairlearn detected moderate source imbalance",
                        'recommendation': "Monitor source distribution in future queries"
                    })
            
            category_analysis = fairlearn_results.get('category_analysis', {})
            if 'fairness_assessment' in category_analysis:
                assessment = category_analysis['fairness_assessment']
                
                if assessment == 'UNFAIR':
                    dp_diff = category_analysis.get('demographic_parity_difference', 0)
                    disadvantaged = category_analysis.get('disadvantaged_group', 'unknown')
                    advantaged = category_analysis.get('advantaged_group', 'unknown')
                    
                    warnings.append({
                        'type': 'fairlearn_category_bias',
                        'severity': 'high',
                        'message': f"Fairlearn detected unfair category distribution (DP Diff: {dp_diff:.3f}). "
                                f"Category '{advantaged}' is over-represented vs '{disadvantaged}'",
                        'recommendation': "Diversify retrieval across topic categories"
                    })
        
        return warnings