"""
Chat interface components
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Any


def render_chat_interface():
    """Render the main chat interface"""
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
                        st.caption(f"{source['source']} - {source.get('date', 'N/A')}")
                        if source.get('url'):
                            st.markdown(f"[🔗 Link]({source['url']})")
                        st.markdown("---")
            
            # Display timestamp and response time
            if timestamp:
                response_time = message.get("response_time", 0)
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


def export_chat_history():
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