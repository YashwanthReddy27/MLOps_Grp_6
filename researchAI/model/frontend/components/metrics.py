"""
Metrics and visualization components
"""
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List


def render_metrics_dashboard():
    """Render comprehensive metrics dashboard"""
    st.header("📊 System Metrics")
    
    # Placeholder for system-wide metrics
    st.info("System-wide metrics will be displayed here")


def render_response_metrics(metrics: Dict[str, Any]):
    """
    Render response-level metrics
    
    Args:
        metrics: Metrics dictionary from API response
    """
    st.subheader("Response Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Retrieval Score",
            f"{metrics.get('retrieval_metrics', {}).get('avg_score', 0):.3f}"
        )
    
    with col2:
        st.metric(
            "Generation Length",
            f"{metrics.get('generation_metrics', {}).get('response_length', 0)} chars"
        )
    
    with col3:
        st.metric(
            "Citations",
            metrics.get('generation_metrics', {}).get('num_citations', 0)
        )


def render_fairness_gauge(fairness_score: float):
    """
    Render fairness score as a gauge chart
    
    Args:
        fairness_score: Fairness score (0-1)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fairness_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fairness Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': get_fairness_color(fairness_score)},
            'steps': [
                {'range': [0, 40], 'color': "#FFCDD2"},
                {'range': [40, 60], 'color': "#FFE0B2"},
                {'range': [60, 80], 'color': "#FFF9C4"},
                {'range': [80, 100], 'color': "#C8E6C9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def get_fairness_color(score: float) -> str:
    """Get color based on fairness score"""
    if score >= 0.8:
        return "#4CAF50"  # Green
    elif score >= 0.6:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red


def render_diversity_chart(diversity_metrics: Dict[str, Any]):
    """
    Render diversity metrics as bar chart
    
    Args:
        diversity_metrics: Diversity metrics dictionary
    """
    if not diversity_metrics:
        return
    
    metrics = {
        "Source Diversity": diversity_metrics.get('source_diversity_ratio', 0) * 100,
        "Category Diversity": diversity_metrics.get('category_diversity_ratio', 0) * 100,
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=['#1976D2', '#388E3C']
        )
    ])
    
    fig.update_layout(
        title="Diversity Metrics",
        yaxis_title="Percentage",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_source_distribution(sources: List[Dict[str, Any]]):
    """
    Render source distribution pie chart
    
    Args:
        sources: List of source dictionaries
    """
    if not sources:
        return
    
    # Count sources by type
    source_counts = {}
    for source in sources:
        source_type = source.get('source', 'Unknown')
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
    
    fig = go.Figure(data=[
        go.Pie(
            labels=list(source_counts.keys()),
            values=list(source_counts.values()),
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="Source Distribution",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_performance_timeline(query_history: List[Dict[str, Any]]):
    """
    Render performance metrics over time
    
    Args:
        query_history: List of query history items
    """
    if not query_history:
        st.info("No query history available yet")
        return
    
    # Extract metrics
    timestamps = []
    response_times = []
    validation_scores = []
    fairness_scores = []
    
    for item in query_history:
        timestamps.append(item['timestamp'])
        
        response = item.get('response', {})
        response_times.append(response.get('response_time', 0))
        
        validation = response.get('validation', {})
        validation_scores.append(validation.get('overall_score', 0) * 100)
        
        bias_report = response.get('bias_report', {})
        fairness_scores.append(bias_report.get('overall_fairness_score', 0) * 100)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=response_times,
        name="Response Time (s)",
        yaxis="y",
        line=dict(color='#1976D2')
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=validation_scores,
        name="Validation Score (%)",
        yaxis="y2",
        line=dict(color='#388E3C')
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=fairness_scores,
        name="Fairness Score (%)",
        yaxis="y2",
        line=dict(color='#F57C00')
    ))
    
    fig.update_layout(
        title="Performance Over Time",
        xaxis=dict(title="Time"),
        yaxis=dict(
            title="Response Time (seconds)",
            titlefont=dict(color="#1976D2"),
            tickfont=dict(color="#1976D2")
        ),
        yaxis2=dict(
            title="Score (%)",
            titlefont=dict(color="#388E3C"),
            tickfont=dict(color="#388E3C"),
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)