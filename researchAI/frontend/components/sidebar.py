"""
Sidebar components for configuration and settings
"""
import streamlit as st
from typing import Dict, Any


def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar with configuration options
    
    Returns:
        Dictionary with selected configuration
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Query filters
        st.subheader("🔍 Query Filters")
        
        # Category filter
        available_categories = [
            "artificial_intelligence",
            "machine_learning",
            "cybersecurity",
            "cloud_computing",
            "web3_blockchain",
            "robotics",
            "quantum_computing",
            "edge_computing"
        ]
        
        selected_categories = st.multiselect(
            "Filter by categories",
            options=available_categories,
            default=None,
            help="Select specific technology categories to focus the search"
        )
        
        # Date range filter (future enhancement)
        # use_date_filter = st.checkbox("Filter by date range", value=False)
        # if use_date_filter:
        #     date_range = st.date_input("Date range", value=[])
        
        st.markdown("---")
        
        # Display settings
        st.subheader("🎨 Display Settings")
        
        show_metrics = st.checkbox("Show response metrics", value=True)
        show_sources = st.checkbox("Show sources", value=True)
        show_fairness = st.checkbox("Show fairness analysis", value=True)
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            st.caption("**Model Configuration**")
            st.text("Embedding: all-MiniLM-L6-v2")
            st.text("LLM: Gemini 2.0 Flash")
            st.text("Reranker: ms-marco-MiniLM")
            
            st.caption("**Retrieval Settings**")
            st.text("Top-k: 20 → 10 → 8")
            st.text("Hybrid: 70% dense, 30% sparse")
        
        st.markdown("---")
        
        # Export functionality
        st.subheader("💾 Export")
        
        if st.button("📥 Export Chat History", use_container_width=True):
            from components.chat import export_chat_history
            
            export_text = export_chat_history()
            if export_text:
                st.download_button(
                    label="Download as TXT",
                    data=export_text,
                    file_name=f"chat_history_{st.session_state.get('session_id', 'export')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning("No chat history to export")
        
        st.markdown("---")
        
        # System info
        st.subheader("ℹ️ System Info")
        st.caption("Version: 1.0.0")
        st.caption("Backend: FastAPI")
        st.caption("Frontend: Streamlit")
        
        # API status (will be updated by main app)
        # st.caption("API Status: ...")
        
        st.markdown("---")
        
        # Help section
        with st.expander("❓ Help", expanded=False):
            st.markdown("""
            **How to use:**
            
            1. Type your question about technology trends
            2. View the AI-generated response with sources
            3. Check fairness metrics and warnings
            4. Rate and provide feedback on responses
            
            **Tips:**
            - Be specific in your queries
            - Use filters to narrow results
            - Check source citations for accuracy
            - Monitor fairness scores for bias
            
            **Categories:**
            - AI/ML: Artificial Intelligence & Machine Learning
            - Cybersecurity: Security trends and threats
            - Cloud: Cloud computing and infrastructure
            - Web3: Blockchain and decentralized tech
            - Robotics: Robotics and automation
            """)
        
        # About section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **Tech Trends RAG Assistant**
            
            An AI-powered research assistant for exploring 
            technology trends using Retrieval-Augmented 
            Generation (RAG) with built-in fairness detection.
            
            **Features:**
            - Hybrid retrieval (FAISS + BM25)
            - Source citations and validation
            - Fairness and bias monitoring
            - Real-time performance metrics
            
            **Data Sources:**
            - arXiv research papers
            - Technology news articles
            
            Built with ❤️ using FastAPI, Streamlit, and Google Gemini
            """)
    
    # Return configuration
    config = {
        "categories": selected_categories if selected_categories else None,
        "show_metrics": show_metrics,
        "show_sources": show_sources,
        "show_fairness": show_fairness
    }
    
    return config