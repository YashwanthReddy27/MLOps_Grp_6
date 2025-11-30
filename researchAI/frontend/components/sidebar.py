import streamlit as st
from typing import Dict, Any


def render_sidebar() -> Dict[str, Any]:
    """
    Render minimal sidebar with only chat history (ChatGPT style)
    
    Returns:
        Dictionary with selected configuration
    """
    with st.sidebar:
        # New Chat Button at the top
        st.markdown("### 💬 Tech Trends RAG")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_response = None
            st.rerun()
        
        st.markdown("---")
        
        # Chat History
        st.markdown("### 📜 Recent Chats")
        
        if 'query_history' in st.session_state and st.session_state.query_history:
            for idx, item in enumerate(reversed(st.session_state.query_history[-20:])):  # Show last 20
                query = item['query']
                timestamp = item['timestamp']
                
                # Truncate long queries
                display_query = query[:40] + "..." if len(query) > 40 else query
                
                if st.button(
                    f"💬 {display_query}",
                    key=f"history_{idx}",
                    use_container_width=True
                ):
                    # Load this conversation
                    st.session_state.messages = []
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query,
                        "timestamp": timestamp
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": item['response']['response'],
                        "sources": item['response'].get('sources', []),
                        "metrics": item['response'].get('metrics', {}),
                        "bias_report": item['response'].get('bias_report', {}),
                        "validation": item['response'].get('validation', {}),
                        "response_time": item['response'].get('response_time', 0),
                        "timestamp": timestamp
                    })
                    st.rerun()
        else:
            st.caption("No chat history yet")
    
    # Return minimal config (no filters)
    config = {
        "categories": None,
        "show_metrics": True,
        "show_sources": True,
        "show_fairness": True
    }
    
    return config