"""
Legal AI Assistant - Main Streamlit Application.

This is the main entry point for the Legal AI Assistant application.
It provides a modern, user-friendly interface for document upload,
processing, and legal Q&A using RAG.

Author: Legal AI Team
Date: 2024
"""

import streamlit as st
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from dotenv import load_dotenv

from auth_db import AuthDatabase
from db import ChatDatabase
from ingestion import DocumentManager
from rag_chatbot import RAGPipeline
from utils import ConfigManager, LoggerSetup, DirectoryManager

# Load environment variables
load_dotenv()

# Initialize directories and logging
DirectoryManager.create_required_directories()
logger = LoggerSetup.setup_logger(__name__)

# Configuration
PAGE_ICON = "⚖️"
PAGE_TITLE = "Legal AI Assistant"

# Set page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Legal AI Assistant - Powered by RAG, LangChain, and Google Gemini"
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #0066cc;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #155724;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #721c24;
        border-left: 4px solid #f5c6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #856404;
        border-left: 4px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


# Initialize services
@st.cache_resource
def get_auth_db():
    """Get authentication database instance."""
    config = ConfigManager()
    db_path = config.get('database.users_db_path', './database/users.db')
    return AuthDatabase(db_path)


@st.cache_resource
def get_chat_db():
    """Get chat database instance."""
    config = ConfigManager()
    db_path = config.get('database.chats_db_path', './database/chats.db')
    return ChatDatabase(db_path)


@st.cache_resource
def get_document_manager():
    """Get document manager instance."""
    return DocumentManager()


@st.cache_resource
def get_rag_pipeline():
    """Get RAG pipeline instance."""
    return RAGPipeline()


# Authentication functions
def login_user(username: str, password: str) -> bool:
    """
    Authenticate user login.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        True if login successful
    """
    auth_db = get_auth_db()
    success, message, user_id = auth_db.login_user(username, password)
    
    if success:
        st.session_state.authenticated = True
        st.session_state.user_id = user_id
        st.session_state.username = username
        logger.info(f"User logged in: {username}")
        return True
    else:
        st.error(message)
        logger.warning(f"Login failed for user: {username}")
        return False


def register_user(username: str, email: str, password: str) -> bool:
    """
    Register a new user.
    
    Args:
        username: Username
        email: Email address
        password: Password
        
    Returns:
        True if registration successful
    """
    auth_db = get_auth_db()
    success, message = auth_db.register_user(username, email, password)
    
    if success:
        st.success(message)
        logger.info(f"User registered: {username}")
        return True
    else:
        st.error(message)
        logger.warning(f"Registration failed for user: {username}")
        return False


def logout_user():
    """Logout current user."""
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.conversation_id = None
    st.session_state.chat_history = []
    logger.info("User logged out")
    st.success("Logged out successfully!")


# Authentication page
def show_auth_page():
    """Display authentication page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"# {PAGE_ICON} {PAGE_TITLE}")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_btn", use_container_width=True):
                if username and password:
                    if login_user(username, password):
                        st.rerun()
                else:
                    st.error("Please enter both username and password")
        
        with tab2:
            st.subheader("Create a New Account")
            
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
            
            if st.button("Register", key="register_btn", use_container_width=True):
                if not all([reg_username, reg_email, reg_password, reg_password_confirm]):
                    st.error("Please fill in all fields")
                elif reg_password != reg_password_confirm:
                    st.error("Passwords do not match")
                elif register_user(reg_username, reg_email, reg_password):
                    st.info("Registration successful! Please log in.")


# Document management sidebar
def show_document_sidebar():
    """Display document management in sidebar."""
    with st.sidebar:
        st.markdown("### 📄 Document Management")
        st.markdown("---")
        
        # Upload PDF
        st.subheader("Upload Legal Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            if st.button("Process Documents", key="process_btn", use_container_width=True):
                document_manager = get_document_manager()
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Save temporary file
                    temp_path = f"./data/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process document
                    status_placeholder.info(f"Processing: {uploaded_file.name}...")
                    success, message, stats = document_manager.ingest_pdf(temp_path)
                    
                    if success:
                        status_placeholder.success(f"✓ {uploaded_file.name}")
                    else:
                        status_placeholder.error(f"✗ {uploaded_file.name}: {message}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success(f"Processed {len(uploaded_files)} documents!")
                logger.info(f"User {st.session_state.username} uploaded {len(uploaded_files)} documents")
        
        st.markdown("---")
        
        # Vector store status
        st.subheader("Vector Store Status")
        rag_pipeline = get_rag_pipeline()
        status = rag_pipeline.get_system_status()
        
        vs_status = status['vector_store']
        
        if vs_status['status'] == 'ready':
            st.markdown(f"""
            <div class="success-box">
            ✓ <b>Status:</b> Ready<br>
            📊 <b>Documents:</b> {vs_status['total_documents']}<br>
            🕐 <b>Updated:</b> {vs_status['last_updated'][:10]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
            ⚠ <b>Status:</b> Empty<br>
            📊 <b>Documents:</b> 0<br>
            Upload documents to get started
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clear database
        if st.button("Clear All Documents", key="clear_btn", use_container_width=True):
            if st.session_state.get("confirm_clear"):
                document_manager = get_document_manager()
                success, message = document_manager.clear_all_documents()
                if success:
                    st.success("All documents cleared!")
                    logger.info(f"User {st.session_state.username} cleared vector store")
                else:
                    st.error(f"Error: {message}")
                st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm clearing all documents")
        
        st.markdown("---")
        
        # Chat history
        st.subheader("Conversation History")
        chat_db = get_chat_db()
        conversations = chat_db.get_user_conversations(st.session_state.user_id, limit=10)
        
        if conversations:
            selected_conv = st.selectbox(
                "Select conversation",
                conversations,
                format_func=lambda x: x['title'][:30] + "..." if len(x['title']) > 30 else x['title'],
                key="conv_selector"
            )
            
            if selected_conv:
                st.session_state.conversation_id = selected_conv['id']
                history = chat_db.get_conversation_history(selected_conv['id'])
                st.session_state.chat_history = history
        else:
            st.info("No conversations yet")


# Main chat interface
def show_chat_interface():
    """Display main chat interface."""
    st.markdown(f"# {PAGE_ICON} Legal AI Assistant")
    st.markdown("Ask questions about your uploaded legal documents using AI-powered RAG")
    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    st.write(message['content'])
                    
                    if message['role'] == 'assistant' and message.get('sources'):
                        with st.expander("📚 Sources"):
                            sources = json.loads(message['sources']) if isinstance(message['sources'], str) else message['sources']
                            for source in sources:
                                st.markdown(f"""
                                **Source {source['index']}:**
                                - Document: {source['document']}
                                - Page: {source['page']}
                                - Similarity: {source['similarity_score']}
                                """)
        else:
            st.info("Start a conversation by asking a question about the uploaded legal documents.")
    
    st.markdown("---")
    
    # Chat input
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about legal documents:",
            placeholder="E.g., What is the Indian Penal Code?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", key="send_btn", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        # Create conversation if not exists
        chat_db = get_chat_db()
        
        if not st.session_state.conversation_id:
            st.session_state.conversation_id = chat_db.create_conversation(
                st.session_state.user_id,
                f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        
        # Add user message to history
        chat_db.add_message(
            st.session_state.conversation_id,
            st.session_state.user_id,
            "user",
            user_input
        )
        
        # Get RAG response
        with st.spinner("Processing your question..."):
            rag_pipeline = get_rag_pipeline()
            response = rag_pipeline.process_question(user_input)
        
        # Add assistant message to history
        chat_db.add_message(
            st.session_state.conversation_id,
            st.session_state.user_id,
            "assistant",
            response['answer'],
            json.dumps(response['sources'])
        )
        
        logger.info(f"User {st.session_state.username} asked: {user_input}")
        
        # Reload to show new messages
        st.rerun()


# Main app
def main():
    """Main application entry point."""
    initialize_session_state()
    
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        # Sidebar with user info and logout
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"👤 **Logged in as:** {st.session_state.username}")
            
            if st.button("Logout", key="logout_btn", use_container_width=True):
                logout_user()
                st.rerun()
        
        show_document_sidebar()
        show_chat_interface()


if __name__ == "__main__":
    main()
