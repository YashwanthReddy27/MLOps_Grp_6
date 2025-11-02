from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from collections import defaultdict
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

from sklearn.metrics import accuracy_score

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
    
    def evaluate_bias_comprehension(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive bias detection across all defined slices
        
        Args:
            evaluation_data: List with query, retrieved_docs, response, user_context, performance_metrics
            
        Returns:
            Comprehensive bias report with metrics, analysis, recommendations
        """
        self.logger.info(f"Starting comprehensive bias detection on {len(evaluation_data)} samples")
        
        # Slice and analyze data
        sliced_data = self._slice_evaluation_data(evaluation_data)
        slice_metrics = self._calculate_slice_metrics(sliced_data)
        bias_analysis = self._analyze_bias_across_slices(slice_metrics)
        
        # FairLearn analysis if available
        fairlearn_analysis = {}
        if FAIRLEARN_AVAILABLE:
            fairlearn_analysis = self._fairlearn_analysis(evaluation_data)
        
        
        return {
            'slice_metrics': slice_metrics,
            'bias_analysis': bias_analysis,
            'fairlearn_analysis': fairlearn_analysis,
            'overall_fairness_score': self._calculate_fairness_score(bias_analysis),
            'timestamp': datetime.now().isoformat()
        } 

    def _slice_evaluation_data(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
        """Slice evaluation data according to defined dimensions"""
        sliced_data = {slice_name: defaultdict(list) for slice_name in self.slice_definitions.keys()}
        
        for record in evaluation_data:
            query = record.get('query', '').lower()
            user_context = record.get('user_context', {})
            retrieved_docs = record.get('retrieved_docs', [])
            
            for slice_name, slice_categories in self.slice_definitions.items():
                category = self._assign_to_category(query, user_context, retrieved_docs, slice_categories)
                if category:
                    sliced_data[slice_name][category].append(record)
        
        return sliced_data
    
    def _assign_to_category(self, query: str, user_context: Dict, 
                           retrieved_docs: List, categories: Dict[str, List[str]]) -> Optional[str]:
        """Assign record to appropriate category based on keywords"""
        query_lower = query.lower()
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        # Check user context
        if user_context:
            context_str = str(user_context).lower()
            for category, keywords in categories.items():
                if any(keyword in context_str for keyword in keywords):
                    return category
        
        # Check retrieved docs metadata
        if retrieved_docs:
            for doc in retrieved_docs[:3]:  # Check top 3 docs
                doc_text = str(doc.get('metadata', {})).lower()
                for category, keywords in categories.items():
                    if any(keyword in doc_text for keyword in keywords):
                        return category
        
        return None
    
    def _calculate_slice_metrics(self, sliced_data: Dict[str, Dict[str, List]]) -> Dict[str, Dict]:
        """Calculate performance metrics for each slice"""
        slice_metrics = {}
        
        for slice_name, categories in sliced_data.items():
            slice_metrics[slice_name] = {}
            
            for category, records in categories.items():
                if not records:
                    continue
                
                # Extract and average metrics
                retrieval_scores = []
                response_qualities = []
                source_diversities = []
                response_times = []
                
                for record in records:
                    perf = record.get('performance_metrics', {})
                    retrieval_scores.append(perf.get('retrieval_score', 0.0))
                    response_qualities.append(perf.get('response_quality', 0.0))
                    source_diversities.append(perf.get('source_diversity', 0.0))
                    response_times.append(perf.get('response_time', 0.0))
                
                slice_metrics[slice_name][category] = {
                    'sample_size': len(records),
                    'metrics': {
                        'avg_retrieval_score': np.mean(retrieval_scores) if retrieval_scores else 0.0,
                        'avg_response_quality': np.mean(response_qualities) if response_qualities else 0.0,
                        'source_diversity': np.mean(source_diversities) if source_diversities else 0.0,
                        'response_time': np.mean(response_times) if response_times else 0.0
                    },
                    'statistical_significance': self._test_significance(len(records))
                }
        
        return slice_metrics
    
    def _test_significance(self, sample_size: int) -> Dict[str, Any]:
        """Test if sample size is statistically significant"""
        return {
            'sample_size': sample_size,
            'sufficient_sample': sample_size >= 30,
            'confidence_level': 0.95 if sample_size >= 30 else 0.8
        }
    
    def _analyze_bias_across_slices(self, slice_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze bias patterns across slices"""
        significant_biases = []
        slice_disparities = {}
        
        for slice_name, categories in slice_metrics.items():
            if len(categories) < 2:
                continue
            
            disparities = self._calculate_disparities(categories)
            slice_disparities[slice_name] = disparities
            
            # Detect significant biases
            for metric_name, disparity in disparities.items():
                bias = self._check_bias_significance(slice_name, metric_name, disparity)
                if bias:
                    significant_biases.append(bias)
        
        return {
            'slice_disparities': slice_disparities,
            'significant_biases': significant_biases,
            'fairness_violations': [b for b in significant_biases if b['severity'] == 'high']
        }
    
    def _calculate_disparities(self, categories: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate performance disparities between categories"""
        disparities = {}
        
        for metric_name in ['avg_retrieval_score', 'avg_response_quality', 'source_diversity']:
            category_values = {cat: data['metrics'].get(metric_name, 0.0) 
                             for cat, data in categories.items()}
            
            if category_values:
                min_val = min(category_values.values())
                max_val = max(category_values.values())
                ratio = min_val / max_val if max_val > 0 else 1.0
                
                disparities[metric_name] = {
                    'min_value': min_val,
                    'max_value': max_val,
                    'ratio': ratio,
                    'absolute_difference': max_val - min_val,
                    'category_values': category_values
                }
        
        return disparities
    
    def _check_bias_significance(self, slice_name: str, metric_name: str, 
                                 disparity: Dict) -> Optional[Dict[str, Any]]:
        """Check if disparity indicates significant bias"""
        ratio = disparity['ratio']
        abs_diff = disparity['absolute_difference']
        
        # Thresholds for bias detection
        if ratio < 0.7 or abs_diff > 0.3:
            severity = 'high' if (ratio < 0.6 or abs_diff > 0.4) else 'medium'
            
            category_values = disparity['category_values']
            min_cat = min(category_values, key=category_values.get)
            max_cat = max(category_values, key=category_values.get)
            
            return {
                'slice': slice_name,
                'bias_type': metric_name,
                'severity': severity,
                'affected_groups': {
                    'disadvantaged': min_cat,
                    'advantaged': max_cat
                },
                'groups': {'disadvantaged': min_cat, 'advantaged': max_cat},
                'metric_details': disparity
            }
        
        return None
    
    def _fairlearn_analysis(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform FairLearn-based bias analysis"""
        try:
            df = self._prepare_fairlearn_data(evaluation_data)
            if df.empty:
                return {'error': 'Insufficient data'}
            
            sensitive_features = ['domain_type', 'query_complexity', 'user_experience']
            available = [f for f in sensitive_features if f in df.columns]
            
            if not available:
                return {'error': 'No sensitive features available'}
            
            fairness_results = {}
            y_true = (df['performance_score'] > df['performance_score'].median()).astype(int)
            y_pred = y_true
            
            for feature in available:
                try:
                    mf = MetricFrame(
                        metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
                        y_true=y_true, y_pred=y_pred, sensitive_features=df[feature]
                    )
                    
                    fairness_results[feature] = {
                        'overall_metrics': mf.overall.to_dict(),
                        'by_group_metrics': mf.by_group.to_dict(),
                        'difference_metrics': {
                            'demographic_parity_diff': demographic_parity_difference(
                                y_true, y_pred, sensitive_features=df[feature]
                            ),
                            'demographic_parity_ratio': demographic_parity_ratio(
                                y_true, y_pred, sensitive_features=df[feature]
                            )
                        }
                    }
                except Exception as e:
                    self.logger.warning(f"FairLearn analysis failed for {feature}: {e}")
                    fairness_results[feature] = {'error': str(e)}
            
            return fairness_results
        except Exception as e:
            self.logger.error(f"FairLearn analysis failed: {e}")
            return {'error': str(e)}
    
    def _prepare_fairlearn_data(self, evaluation_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare data for FairLearn analysis"""
        records = []
        
        for record in evaluation_data:
            query = record.get('query', '').lower()
            perf = record.get('performance_metrics', {})
            
            # Assign to slices
            domain = self._assign_to_category(query, {}, [], self.slice_definitions['domain_type'])
            complexity = self._assign_to_category(query, {}, [], self.slice_definitions['query_complexity'])
            experience = self._assign_to_category(query, {}, [], self.slice_definitions['user_experience'])
            
            # Calculate composite score
            performance_score = np.mean([
                perf.get('retrieval_score', 0.5),
                perf.get('response_quality', 0.5),
                perf.get('source_diversity', 0.5)
            ])
            
            records.append({
                'query': query,
                'domain_type': domain or 'unknown',
                'query_complexity': complexity or 'unknown',
                'user_experience': experience or 'unknown',
                'performance_score': performance_score
            })
        
        return pd.DataFrame(records)
    
    
    def _calculate_fairness_score(self, bias_analysis: Dict[str, Any]) -> float:
        """Calculate overall fairness score (0-1, higher is better)"""
        significant_biases = bias_analysis.get('significant_biases', [])
        
        if not significant_biases:
            return 1.0
        
        penalty = sum(0.3 if b['severity'] == 'high' else 0.2 if b['severity'] == 'medium' else 0.1
                     for b in significant_biases)
        
        return max(0.0, 1.0 - penalty)