
"""
Enhanced Bias Detection using Fairlearn
Detects fairness issues across different sensitive attributes
"""

import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime

# Fairlearn imports
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    selection_rate
)


class FairlearnBiasDetector:
    """
    Enhanced bias detector using Fairlearn for fairness metrics
    Analyzes distribution fairness across sensitive attributes
    """
    
    def __init__(self, data: Dict[str, Any], data_type: str = "arxiv"):
        """
        Initialize Fairlearn bias detector
        
        Args:
            data: Dictionary containing the processed data
            data_type: Type of data - "arxiv" or "news"
        """
        self.data = data
        self.data_type = data_type
        self.bias_report = {}
        self.df = None
        
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert JSON data to pandas DataFrame for Fairlearn analysis"""
        
        if self.data_type == "arxiv":
            items = self.data.get("papers", [])
            
            records = []
            for item in items:
                record = {
                    'id': item.get('arxiv_id', ''),
                    'primary_category': item.get('primary_category', 'unknown'),
                    'author_count': item.get('author_count', 1),
                    'published_date': item.get('published_date', ''),
                    'primary_arxiv_category': item.get('primary_arxiv_category', 'unknown'),
                    'num_categories': len(item.get('all_categories', [])),
                    'relevance_score': item.get('overall_relevance', 0.0)
                }
                records.append(record)
                
        else:  # news
            items = self.data.get("articles", [])
            
            records = []
            for item in items:
                record = {
                    'id': item.get('article_id', ''),
                    'primary_category': item.get('primary_category', 'unknown'),
                    'source_name': item.get('source_name', 'unknown'),
                    'published_date': item.get('published_at', ''),
                    'num_categories': len(item.get('all_categories', [])),
                    'relevance_score': item.get('overall_relevance', 0.0)
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        
        # Add derived features
        if 'published_date' in df.columns:
            df['month'] = pd.to_datetime(df['published_date'], errors='coerce').dt.to_period('M').astype(str)
        
        # Create binary target: high relevance vs low relevance
        if 'relevance_score' in df.columns:
            median_score = df['relevance_score'].median()
            df['high_relevance'] = (df['relevance_score'] >= median_score).astype(int)
        else:
            df['high_relevance'] = 1  # Default if no relevance score
        
        return df
    
    def detect_all_biases(self) -> Dict[str, Any]:
        """Run Fairlearn-based bias detection"""
        
        self.df = self._prepare_dataframe()
        
        self.bias_report = {
            "data_type": self.data_type,
            "total_items": len(self.df),
            "timestamp": datetime.now().isoformat(),
            "fairness_metrics": []
        }
        
        # Run fairness analyses
        self._analyze_category_fairness()
        self._analyze_source_fairness()
        self._analyze_temporal_fairness()
        
        if self.data_type == "arxiv":
            self._analyze_author_fairness()
        
        # Calculate overall fairness score
        self.bias_report["overall_fairness_score"] = self._calculate_fairness_score()
        return self.bias_report
    
    def _analyze_category_fairness(self):
        """Analyze fairness across primary categories using Fairlearn"""
        if 'primary_category' not in self.df.columns:
            return
        
        # Get top categories (at least 5 samples each for meaningful analysis)
        category_counts = self.df['primary_category'].value_counts()
        valid_categories = category_counts[category_counts >= 5].index.tolist()
        
        if len(valid_categories) < 2:
            return
        
        df_filtered = self.df[self.df['primary_category'].isin(valid_categories)].copy()
        
        # Calculate selection rates (what proportion are marked as high relevance)
        metric_frame = MetricFrame(
            metrics=selection_rate,
            y_true=df_filtered['high_relevance'],
            y_pred=df_filtered['high_relevance'],  # Use actual as pred for distribution analysis
            sensitive_features=df_filtered['primary_category']
        )
        
        # Calculate demographic parity
        try:
            dp_diff = demographic_parity_difference(
                y_true=df_filtered['high_relevance'],
                y_pred=df_filtered['high_relevance'],
                sensitive_features=df_filtered['primary_category']
            )
            
            dp_ratio = demographic_parity_ratio(
                y_true=df_filtered['high_relevance'],
                y_pred=df_filtered['high_relevance'],
                sensitive_features=df_filtered['primary_category']
            )
        except:
            dp_diff = 0.0
            dp_ratio = 1.0
        
        # Interpret results
        severity = self._interpret_demographic_parity(dp_diff, dp_ratio)
        
        self.bias_report["fairness_metrics"].append({
            "analysis_type": "Category Fairness",
            "sensitive_attribute": "primary_category",
            "severity": severity,
            "metrics": {
                "demographic_parity_difference": round(float(dp_diff), 4),
                "demographic_parity_ratio": round(float(dp_ratio), 4),
                "selection_rates": {k: round(float(v), 4) for k, v in metric_frame.by_group.items()},
                "num_groups_analyzed": len(valid_categories),
                "groups_analyzed": valid_categories[:5]  # Show top 5
            }
        })
    
    def _analyze_source_fairness(self):
        """Analyze fairness across data sources"""
        
        source_col = 'primary_arxiv_category' if self.data_type == 'arxiv' else 'source_name'
        
        if source_col not in self.df.columns:
            return
        
        # Get sources with sufficient samples
        source_counts = self.df[source_col].value_counts()
        valid_sources = source_counts[source_counts >= 5].index.tolist()
        
        if len(valid_sources) < 2:
            return
        
        df_filtered = self.df[self.df[source_col].isin(valid_sources)].copy()
        
        # Calculate selection rates
        metric_frame = MetricFrame(
            metrics=selection_rate,
            y_true=df_filtered['high_relevance'],
            y_pred=df_filtered['high_relevance'],
            sensitive_features=df_filtered[source_col]
        )
        
        # Calculate disparities
        try:
            dp_diff = demographic_parity_difference(
                y_true=df_filtered['high_relevance'],
                y_pred=df_filtered['high_relevance'],
                sensitive_features=df_filtered[source_col]
            )
            
            dp_ratio = demographic_parity_ratio(
                y_true=df_filtered['high_relevance'],
                y_pred=df_filtered['high_relevance'],
                sensitive_features=df_filtered[source_col]
            )
        except:
            dp_diff = 0.0
            dp_ratio = 1.0
        
        severity = self._interpret_demographic_parity(dp_diff, dp_ratio)
        
        self.bias_report["fairness_metrics"].append({
            "analysis_type": "Source Fairness",
            "sensitive_attribute": source_col,
            "severity": severity,
            "metrics": {
                "demographic_parity_difference": round(float(dp_diff), 4),
                "demographic_parity_ratio": round(float(dp_ratio), 4),
                "selection_rates": {k: round(float(v), 4) for k, v in metric_frame.by_group.items()},
                "num_sources_analyzed": len(valid_sources),
                "sources_analyzed": valid_sources[:5]
            }
        })
    
    def _analyze_temporal_fairness(self):
        """Analyze fairness across time periods"""
        
        if 'month' not in self.df.columns or self.df['month'].isna().all():
            return
        
        df_filtered = self.df.dropna(subset=['month']).copy()
        
        # Get months with sufficient samples
        month_counts = df_filtered['month'].value_counts()
        valid_months = month_counts[month_counts >= 5].index.tolist()
        
        if len(valid_months) < 2:
            return
        
        df_filtered = df_filtered[df_filtered['month'].isin(valid_months)].copy()
        
        # Calculate temporal disparities
        metric_frame = MetricFrame(
            metrics=selection_rate,
            y_true=df_filtered['high_relevance'],
            y_pred=df_filtered['high_relevance'],
            sensitive_features=df_filtered['month']
        )
        
        try:
            dp_diff = demographic_parity_difference(
                y_true=df_filtered['high_relevance'],
                y_pred=df_filtered['high_relevance'],
                sensitive_features=df_filtered['month']
            )
            
            dp_ratio = demographic_parity_ratio(
                y_true=df_filtered['high_relevance'],
                y_pred=df_filtered['high_relevance'],
                sensitive_features=df_filtered['month']
            )
        except:
            dp_diff = 0.0
            dp_ratio = 1.0
        
        severity = self._interpret_demographic_parity(dp_diff, dp_ratio)
        
        self.bias_report["fairness_metrics"].append({
            "analysis_type": "Temporal Fairness",
            "sensitive_attribute": "month",
            "severity": severity,
            "metrics": {
                "demographic_parity_difference": round(float(dp_diff), 4),
                "demographic_parity_ratio": round(float(dp_ratio), 4),
                "selection_rates": {k: round(float(v), 4) for k, v in metric_frame.by_group.items()},
                "num_periods_analyzed": len(valid_months),
                "periods_analyzed": sorted(valid_months)
            }
        })
    
    def _analyze_author_fairness(self):
        """Analyze fairness based on author count (ArXiv only)"""
        
        if 'author_count' not in self.df.columns:
            return
        
        # Create bins: single author vs multi-author
        self.df['author_group'] = self.df['author_count'].apply(
            lambda x: 'single_author' if x == 1 else 'multi_author'
        )
        
        # Calculate disparities
        metric_frame = MetricFrame(
            metrics=selection_rate,
            y_true=self.df['high_relevance'],
            y_pred=self.df['high_relevance'],
            sensitive_features=self.df['author_group']
        )
        
        try:
            dp_diff = demographic_parity_difference(
                y_true=self.df['high_relevance'],
                y_pred=self.df['high_relevance'],
                sensitive_features=self.df['author_group']
            )
            
            dp_ratio = demographic_parity_ratio(
                y_true=self.df['high_relevance'],
                y_pred=self.df['high_relevance'],
                sensitive_features=self.df['author_group']
            )
        except:
            dp_diff = 0.0
            dp_ratio = 1.0
        
        severity = self._interpret_demographic_parity(dp_diff, dp_ratio)
        
        self.bias_report["fairness_metrics"].append({
            "analysis_type": "Author Count Fairness",
            "sensitive_attribute": "author_group",
            "severity": severity,
            "metrics": {
                "demographic_parity_difference": round(float(dp_diff), 4),
                "demographic_parity_ratio": round(float(dp_ratio), 4),
                "selection_rates": {k: round(float(v), 4) for k, v in metric_frame.by_group.items()},
                "single_author_percentage": round(float((self.df['author_count'] == 1).mean() * 100), 2)
            }
        })
    
    def _interpret_demographic_parity(self, dp_diff: float, dp_ratio: float) -> str:
        """
        Interpret demographic parity metrics
        
        Demographic Parity Difference: closer to 0 is better (range: -1 to 1)
        Demographic Parity Ratio: closer to 1 is better (range: 0 to inf)
        """
        
        # Ideal: dp_diff near 0, dp_ratio near 1
        abs_diff = abs(dp_diff)
        
        if abs_diff < 0.1 and 0.8 <= dp_ratio <= 1.25:
            return "Low"  # Fair
        elif abs_diff < 0.25 and 0.6 <= dp_ratio <= 1.67:
            return "Medium"  # Some disparity
        else:
            return "High"  # Significant disparity
    
    def _calculate_fairness_score(self) -> Dict[str, Any]:
        """Calculate overall fairness score"""
        
        severity_weights = {"Low": 1, "Medium": 2, "High": 3}
        
        total_score = 0
        max_score = 0
        
        for metric in self.bias_report["fairness_metrics"]:
            severity = metric.get("severity", "Low")
            weight = severity_weights.get(severity, 1)
            total_score += weight
            max_score += 3
        
        normalized_score = (total_score / max_score * 100) if max_score > 0 else 0
        
        return {
            "score": round(normalized_score, 2),
            "interpretation": self._interpret_fairness_score(normalized_score),
            "high_severity_count": sum(1 for m in self.bias_report["fairness_metrics"] 
                                      if m.get("severity") == "High"),
            "total_metrics_analyzed": len(self.bias_report["fairness_metrics"])
        }
    
    def _interpret_fairness_score(self, score: float) -> str:
        """Interpret the overall fairness score"""
        if score < 35:
            return "Fair - Low bias across groups"
        elif score < 65:
            return "Moderate unfairness - Some disparities present"
        else:
            return "Unfair - Significant disparities detected"
    

    def save_report(self, output_path: str):
        """Save Fairlearn bias report to JSON file"""
        
        f = open(output_path, 'w')
        json.dump(self.bias_report, f, indent=2)
