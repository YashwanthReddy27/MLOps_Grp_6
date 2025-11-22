"""
Streamlit Frontend for Tech Trends RAG Application
"""
import streamlit as st
import requests
import time
from datetime import datetime
from typing import List, Dict, Any

# Import custom components
from components.chat import render_chat_interface, display_message
from components.sidebar import render_sidebar
from components.metrics import render_metrics_dashboard
from utils.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="ResearchAI- Your Personal Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .assistant-message {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Source cards */
    .source-card {
        background-color: #FAFAFA;
        border-left: 4px solid #1976D2;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    /* Metrics */
    .metric-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Fairness indicator */
    .fairness-excellent {
        color: #2E7D32;
        font-weight: bold;
    }
    
    .fairness-moderate {
        color: #F57C00;
        font-weight: bold;
    }
    
    .fairness-concerning {
        color: #C62828;
        font-weight: bold;
    }
    
    /* Headers */
    h1 {
        color: #1976D2;
    }
    
    h2 {
        color: #424242;
    }
    
    h3 {
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'api_client' not in st.session_state:
        # Get API URL from environment or use default
        api_url = st.secrets.get("API_URL", "http://localhost:8000")
        st.session_state.api_client = APIClient(base_url=api_url)
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None


def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🤖 Tech Trends RAG Assistant")
    st.markdown("*Powered by AI with Fairness Detection*")
    
    # Sidebar
    sidebar_config = render_sidebar()
    
    # Check API health
    api_client = st.session_state.api_client
    
    with st.spinner("Connecting to backend..."):
        health = api_client.health_check()
    
    if not health or health.get("status") != "healthy":
        st.error("❌ Backend API is not available. Please start the backend service.")
        st.code("python model/api/main.py", language="bash")
        st.stop()
    
    # Display health status in sidebar
    with st.sidebar:
        st.success("✅ Backend Connected")
        st.caption(f"Indexes loaded: {health.get('indexes_loaded', False)}")
    
    # Main layout - two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat Interface
        st.header("💬 Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            display_message(message)
        
        # Query input
        query = st.chat_input("Ask about technology trends...")
        
        if query:
            # Add user message
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_message)
            display_message(user_message)
            
            # Process query
            with st.spinner("🤔 Thinking..."):
                try:
                    # Apply filters from sidebar if any
                    filters = None
                    if sidebar_config.get("categories"):
                        filters = {"categories": sidebar_config["categories"]}
                    
                    # Call API
                    response = api_client.query(
                        query=query,
                        filters=filters
                    )
                    
                    if response:
                        # Store response
                        st.session_state.current_response = response
                        
                        # Add assistant message
                        assistant_message = {
                            "role": "assistant",
                            "content": response["response"],
                            "sources": response.get("sources", []),
                            "metrics": response.get("metrics", {}),
                            "bias_report": response.get("bias_report", {}),
                            "response_time": response.get("response_time", 0),
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.messages.append(assistant_message)
                        
                        # Add to history
                        st.session_state.query_history.append({
                            "query": query,
                            "response": response,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Rerun to display new message
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_response = None
            st.rerun()
    
    with col2:
        # Metrics and Information Panel
        st.header("📊 Response Insights")
        
        if st.session_state.current_response:
            response = st.session_state.current_response
            
            # Performance Metrics
            st.subheader("Performance")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric(
                    "Response Time",
                    f"{response.get('response_time', 0):.2f}s"
                )
            
            with metrics_col2:
                st.metric(
                    "Sources",
                    response.get('num_sources', 0)
                )
            
            # Validation Score
            validation = response.get('validation', {})
            validation_score = validation.get('overall_score', 0)
            
            st.metric(
                "Validation Score",
                f"{validation_score:.2%}",
                delta=None
            )
            
            # Progress bar for validation
            st.progress(validation_score)
            
            # Fairness Report
            st.subheader("⚖️ Fairness Analysis")
            
            bias_report = response.get('bias_report', {})
            fairness_score = bias_report.get('overall_fairness_score', 0)
            
            # Fairness indicator
            if fairness_score >= 0.8:
                fairness_status = "EXCELLENT"
                fairness_class = "fairness-excellent"
                fairness_emoji = "✅"
            elif fairness_score >= 0.6:
                fairness_status = "MODERATE"
                fairness_class = "fairness-moderate"
                fairness_emoji = "⚠️"
            else:
                fairness_status = "CONCERNING"
                fairness_class = "fairness-concerning"
                fairness_emoji = "🔴"
            
            st.markdown(
                f'<div class="{fairness_class}">'
                f'{fairness_emoji} {fairness_status}: {fairness_score:.2%}'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Progress bar for fairness
            st.progress(fairness_score)
            
            # Diversity metrics
            diversity_metrics = bias_report.get('diversity_metrics', {})
            
            if diversity_metrics:
                st.caption("**Diversity Metrics**")
                
                div_col1, div_col2 = st.columns(2)
                
                with div_col1:
                    st.caption(f"Unique Sources: {diversity_metrics.get('num_unique_sources', 0)}")
                
                with div_col2:
                    st.caption(f"Categories: {diversity_metrics.get('num_unique_categories', 0)}")
            
            # Warnings
            warnings = bias_report.get('warnings', [])
            if warnings:
                st.subheader("⚠️ Warnings")
                for warning in warnings:
                    with st.expander(f"{warning.get('type', 'Warning')}", expanded=False):
                        st.caption(f"**Severity:** {warning.get('severity', 'unknown')}")
                        st.write(warning.get('message', ''))
                        if warning.get('recommendation'):
                            st.info(f"💡 {warning['recommendation']}")
            
            # Sources
            st.subheader("📚 Sources")
            
            sources = response.get('sources', [])
            if sources:
                for source in sources:
                    with st.expander(f"[{source['number']}] {source['title']}", expanded=False):
                        st.caption(f"**Source:** {source.get('source', 'Unknown')}")
                        st.caption(f"**Date:** {source.get('date', 'N/A')}")
                        if source.get('url'):
                            st.markdown(f"[🔗 View Source]({source['url']})")
            else:
                st.caption("No sources available")
            
            # Feedback section
            st.subheader("💭 Feedback")
            
            with st.form("feedback_form"):
                rating = st.slider("Rate this response", 1, 5, 3)
                feedback_text = st.text_area("Additional feedback (optional)")
                
                issues = st.multiselect(
                    "Issues (if any)",
                    ["Inaccurate", "Biased", "Incomplete", "Outdated", "Other"]
                )
                
                submit_feedback = st.form_submit_button("Submit Feedback")
                
                if submit_feedback:
                    feedback_response = api_client.submit_feedback(
                        query=st.session_state.messages[-2]['content'],  # Last user query
                        rating=rating,
                        feedback_text=feedback_text if feedback_text else None,
                        issues=[issue.lower() for issue in issues] if issues else None
                    )
                    
                    if feedback_response:
                        st.success("✅ Thank you for your feedback!")
        
        else:
            st.info("💡 Ask a question to see response insights and metrics")
    
    # Footer
    st.markdown("---")
    st.caption("Tech Trends RAG v1.0 | Powered by Google Gemini & FAISS")


if __name__ == "__main__":
    main()