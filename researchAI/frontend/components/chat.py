import streamlit as st
from datetime import datetime
from typing import Dict, Any

def render_chat_interface():
    st.header("💬 Chat")

def display_message(message: Dict[str, Any]):
    """
    Display a chat message
    
    Args:
        message: Message dictionary with role, content, and metadata
    """
    role = message.get("role")
    content = message.get("content")
    timestamp = message.get("timestamp")
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
            if timestamp:
                st.caption(f"*{format_timestamp(timestamp)}*")
    
    elif role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            # Display sources if available
            sources = message.get("sources", [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                    for source in sources:
                        st.markdown(
                            f"**[{source['number']}]** {source['title']}"
                        )
                        if source.get('url'):
                            st.caption(f"{source['source']} - {source.get('date', 'Date N/A')} - [🔗 Link]({source['url']})")
                        else:
                            st.caption(f"{source['source']} - {source.get('date', 'Date N/A')}")
            
            # Display performance metrics if available (NEW SECTION)
            metrics = message.get("metrics", {})
            response_time = message.get("response_time", 0)
            bias_report = message.get("bias_report", {})
            
            if metrics or response_time or bias_report:
                with st.expander("📊 Performance Metrics", expanded=False):
                    # Response time and validation
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Response Time", f"{response_time:.2f}s")
                    
                    with col2:
                        validation_score = message.get("validation", {}).get("overall_score", 0)
                        st.metric("Validation", f"{validation_score:.1%}")
                    
                    with col3:
                        fairness_score = bias_report.get('overall_fairness_score', 0)
                        st.metric("Fairness", f"{fairness_score:.1%}")
                    
                    # Retrieval metrics
                    if metrics and 'retrieval_metrics' in metrics:
                        st.markdown("**Retrieval Metrics**")
                        retrieval = metrics['retrieval_metrics']
                        
                        met_col1, met_col2 = st.columns(2)
                        with met_col1:
                            st.caption(f"Retrieved: {retrieval.get('num_retrieved', 0)} docs")
                            st.caption(f"Avg Score: {retrieval.get('avg_score', 0):.3f}")
                        
                        with met_col2:
                            st.caption(f"Sources: {retrieval.get('source_diversity', 0)}")
                            st.caption(f"Categories: {retrieval.get('category_diversity', 0)}")
                    
                    # Generation metrics
                    if metrics and 'generation_metrics' in metrics:
                        st.markdown("**Generation Metrics**")
                        generation = metrics['generation_metrics']
                        
                        gen_col1, gen_col2 = st.columns(2)
                        with gen_col1:
                            st.caption(f"Length: {generation.get('response_length', 0)} chars")
                        
                        with gen_col2:
                            st.caption(f"Citations: {generation.get('num_citations', 0)}")
                    
                    # Fairness warnings
                    warnings = bias_report.get('warnings', [])
                    if warnings:
                        st.markdown("**⚠️ Fairness Warnings**")
                        for warning in warnings:
                            st.warning(f"**{warning.get('type', 'Warning')}**: {warning.get('message', '')}")
            
            # Display timestamp
            if timestamp:
                st.caption(
                    f"*{format_timestamp(timestamp)} • "
                    f"Response time: {response_time:.2f}s*"
                )

def format_timestamp(timestamp: str) -> str:
    """
    Format ISO timestamp to readable format
    
    Args:
        timestamp: ISO format timestamp string
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%I:%M %p")
    except:
        return timestamp

def render_chat_history_sidebar():
    """Render chat history in sidebar"""
    st.sidebar.header("📜 Chat History")
    
    if 'query_history' in st.session_state and st.session_state.query_history:
        for idx, item in enumerate(reversed(st.session_state.query_history[-10:])):
            query = item['query']
            timestamp = item['timestamp']
            
            if st.sidebar.button(
                f"{format_timestamp(timestamp)}: {query[:50]}...",
                key=f"history_{idx}"
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
                    "timestamp": timestamp
                })
                st.rerun()
    else:
        st.sidebar.caption("No chat history yet")

    """Export chat history as text"""
    if 'messages' not in st.session_state or not st.session_state.messages:
        return None
    
    export_text = "Tech Trends RAG - Chat History\n"
    export_text += "=" * 50 + "\n\n"
    
    for message in st.session_state.messages:
        role = message.get("role", "").upper()
        content = message.get("content", "")
        timestamp = message.get("timestamp", "")
        
        export_text += f"{role} [{format_timestamp(timestamp)}]:\n"
        export_text += f"{content}\n"
        
        # Add sources for assistant messages
        if role == "ASSISTANT" and message.get("sources"):
            export_text += "\nSources:\n"
            for source in message.get("sources", []):
                export_text += f"  [{source['number']}] {source['title']}\n"
                export_text += f"      {source.get('url', '')}\n"
        
        export_text += "\n" + "-" * 50 + "\n\n"
    
    return export_text