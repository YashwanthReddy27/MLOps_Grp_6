"""
Streamlit Frontend for Tech Trends RAG Application
"""
import streamlit as st
from datetime import datetime
import os
import hashlib
import json
from pathlib import Path

# Import custom components
from components.chat import display_message
from components.sidebar import render_sidebar
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
    /* Hide Streamlit header and menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Optional: Hide "Deploy" button specifically */
    .stDeployButton {display: none;}
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

def init_session_state():
    """Initialize session state variables with persistence"""
    
    # Create cache directory
    cache_dir = Path('.streamlit_cache')
    cache_dir.mkdir(exist_ok=True)
    
    # Generate or retrieve session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:8]
    
    # Try to load existing session
    session_file = cache_dir / f"session_{st.session_state.session_id}.json"
    
    if session_file.exists():
        try:
            with open(session_file, 'r') as f:
                saved_data = json.load(f)
                st.session_state.messages = saved_data.get('messages', [])
                st.session_state.query_history = saved_data.get('query_history', [])
        except:
            pass
    
    # Initialize defaults if not loaded
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'api_client' not in st.session_state:
        try:
            api_url = st.secrets["API_URL"]
        except Exception:
            api_url = os.environ.get("API_URL", None)
            if api_url is None:
                api_url = "http://localhost:8000"
        
        st.session_state.api_client = APIClient(base_url=api_url)
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'current_response' not in st.session_state:
        st.session_state.current_response = None


def save_session():
    """Save current session to file"""
    try:
        cache_dir = Path('.streamlit_cache')
        cache_dir.mkdir(exist_ok=True)
        
        session_file = cache_dir / f"session_{st.session_state.session_id}.json"
        
        session_data = {
            'messages': st.session_state.messages,
            'query_history': st.session_state.query_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        pass  # Silently fail

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("Tech Trends RAG Assistant")
    
    # Sidebar
    sidebar_config = render_sidebar()
    
    # Check API health
    api_client = st.session_state.api_client
    
    with st.spinner("Connecting to backend..."):
        health = api_client.health_check()
    
    # Handle connection failure
    if health is None:
        st.error("❌ Cannot connect to backend API")
        
        st.warning("**Please ensure the backend is running:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.code("cd model\npython -m uvicorn api.main:app --reload", language="bash")
        
        with col2:
            st.code("# Or in Docker:\ndocker-compose -f docker-compose-rag.yml up", language="bash")
        
        st.info(f"📍 Trying to connect to: `{api_client.base_url}`")
        
        if st.button("🔄 Retry Connection"):
            st.rerun()
        
        st.stop()
    
    # Handle unhealthy status
    if health.get("status") != "healthy":
        st.error(f"❌ Backend API returned unhealthy status: {health.get('status')}")
        st.json(health)
        st.stop()
    
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
                    
                    # Add assistant message with ALL data
                    assistant_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "sources": response.get("sources", []),
                        "metrics": response.get("metrics", {}),
                        "bias_report": response.get("bias_report", {}),
                        "validation": response.get("validation", {}),
                        "response_time": response.get("response_time", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    if len(st.session_state.messages) == 2:  # User + Assistant = first exchange
                        st.session_state.query_history.append({
                            "query": query,
                            "response": response,
                            "timestamp": datetime.now().isoformat(),
                            "conversation_id": len(st.session_state.query_history)  # Track conversation
                        })
                
                    # Rerun to display new message
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()