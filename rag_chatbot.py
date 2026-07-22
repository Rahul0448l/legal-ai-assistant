"""
RAG Chatbot using LangGraph and Google Gemini.

This module implements the Retrieval-Augmented Generation (RAG) pipeline
using LangGraph for workflow orchestration and Google Gemini for LLM.

Author: Legal AI Team
Date: 2024
"""

import logging
from typing import Optional, Dict, List, Any
import json
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from ingestion import VectorStore
from utils import ConfigManager, retry_on_exception, sanitize_input

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """State type for the chat workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    context: str
    sources: List[Dict]
    answer: str
    error: Optional[str]


class RAGChatbot:
    """
    RAG-based Legal AI Chatbot using LangGraph and Google Gemini.
    
    Handles:
    - Document retrieval
    - Prompt construction
    - LLM inference
    - Response generation with citations
    """
    
    _instance: Optional['RAGChatbot'] = None
    
    def __new__(cls) -> 'RAGChatbot':
        """Implement singleton pattern for RAGChatbot."""
        if cls._instance is None:
            cls._instance = super(RAGChatbot, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize RAGChatbot with configuration and LLM."""
        self.config = ConfigManager()
        self._initialize_llm()
        self._initialize_retriever()
        self._build_workflow()
        logger.info("RAGChatbot initialized")
    
    def _initialize_llm(self) -> None:
        """Initialize Google Gemini LLM."""
        try:
            model_name = self.config.get('llm.model_name', 'gemini-2.5-flash')
            temperature = self.config.get('llm.temperature', 0.3)
            max_tokens = self.config.get('llm.max_tokens', 2048)
            
            logger.info(f"Initializing LLM: {model_name}")
            
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                convert_system_message_to_human=True
            )
            
            logger.info("LLM initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _initialize_retriever(self) -> None:
        """Initialize document retriever."""
        try:
            logger.info("Initializing retriever")
            self.vector_store = VectorStore()
            logger.info("Retriever initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise
    
    def _build_workflow(self) -> None:
        """Build LangGraph workflow for RAG pipeline."""
        workflow = StateGraph(ChatState)
        
        # Define workflow nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("format", self._format_response)
        
        # Define edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "format")
        workflow.add_edge("format", END)
        
        self.workflow = workflow.compile()
        logger.info("RAG workflow built successfully")
    
    def _retrieve_documents(self, state: ChatState) -> Dict[str, Any]:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated state with retrieved context and sources
        """
        try:
            question = state["question"]
            logger.info(f"Retrieving documents for question: {question}")
            
            # Perform similarity search
            results = self.vector_store.similarity_search(question)
            
            if not results:
                logger.warning("No documents found for the query")
                state["context"] = ""
                state["sources"] = []
                state["error"] = "No relevant documents found in the database"
                return state
            
            # Extract context and sources
            context_parts = []
            sources = []
            
            for i, result in enumerate(results, 1):
                content = result['content']
                metadata = result['metadata']
                score = result['score']
                
                context_parts.append(f"Source {i}:\n{content}")
                
                sources.append({
                    'index': i,
                    'document': metadata.get('source', 'Unknown'),
                    'page': metadata.get('page', 'Unknown'),
                    'similarity_score': round(score, 3),
                    'chunk_id': metadata.get('chunk_id', 'Unknown')
                })
            
            state["context"] = "\n\n".join(context_parts)
            state["sources"] = sources
            state["error"] = None
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return state
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            state["error"] = f"Error retrieving documents: {str(e)}"
            return state
    
    def _generate_response(self, state: ChatState) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated state with generated answer
        """
        try:
            if state["error"]:
                state["answer"] = state["error"]
                return state
            
            question = state["question"]
            context = state["context"]
            
            logger.info(f"Generating response for question: {question}")
            
            # Build prompt
            system_prompt = self.config.get('system_prompt')
            
            prompt_template = PromptTemplate(
                input_variables=["system_prompt", "context", "question"],
                template="""
{system_prompt}

---

RETRIEVED CONTEXT FROM LEGAL DOCUMENTS:

{context}

---

USER QUESTION:
{question}

IMPORTANT INSTRUCTIONS:
1. Answer ONLY based on the retrieved context above.
2. Do NOT provide general legal information outside the context.
3. If the answer is NOT in the context, state: "I couldn't find sufficient information in the uploaded legal documents to answer this question."
4. Cite the source documents when providing answers.
5. Be precise and professional.

ANSWER:
"""
            )
            
            prompt = prompt_template.format(
                system_prompt=system_prompt,
                context=context,
                question=question
            )
            
            # Generate response
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            state["answer"] = response.content
            
            logger.info("Response generated successfully")
            return state
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["error"] = f"Error generating response: {str(e)}"
            state["answer"] = "I encountered an error while processing your question. Please try again."
            return state
    
    def _format_response(self, state: ChatState) -> Dict[str, Any]:
        """
        Format the response with sources and metadata.
        
        Args:
            state: Current chat state
            
        Returns:
            Final formatted state
        """
        try:
            logger.info("Formatting response with sources")
            # Response formatting is already handled, just log it
            return state
        
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return state
    
    @retry_on_exception(max_retries=2, delay=1.0)
    def chat(self, question: str) -> Dict[str, Any]:
        """
        Process a user question and generate a response.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Sanitize input
            question = sanitize_input(question, max_length=5000)
            
            logger.info(f"Processing user question: {question}")
            
            # Check if vector store has documents
            stats = self.vector_store.get_collection_stats()
            if stats.get('total_documents', 0) == 0:
                logger.warning("No documents in vector store")
                return {
                    'answer': "No legal documents have been uploaded yet. Please upload PDF documents first.",
                    'sources': [],
                    'success': False,
                    'error': "Empty vector store",
                    'timestamp': datetime.now().isoformat()
                }
            
            # Initialize state
            initial_state = {
                "messages": [],
                "question": question,
                "context": "",
                "sources": [],
                "answer": "",
                "error": None
            }
            
            # Run workflow
            result = self.workflow.invoke(initial_state)
            
            # Format output
            response = {
                'answer': result['answer'],
                'sources': result['sources'],
                'success': result['error'] is None,
                'error': result['error'],
                'timestamp': datetime.now().isoformat(),
                'context_count': len(result['sources'])
            }
            
            logger.info(f"Question processed successfully with {len(result['sources'])} sources")
            return response
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                'answer': "An unexpected error occurred. Please try again.",
                'sources': [],
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_vector_store_status(self) -> Dict[str, Any]:
        """
        Check the status of the vector store.
        
        Returns:
            Status information
        """
        try:
            stats = self.vector_store.get_collection_stats()
            return {
                'status': 'ready' if stats.get('total_documents', 0) > 0 else 'empty',
                'total_documents': stats.get('total_documents', 0),
                'last_updated': stats.get('last_updated', 'Unknown'),
                'collection_name': stats.get('collection_name', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error checking vector store status: {e}")
            return {'status': 'error', 'error': str(e)}


class RAGPipeline:
    """
    High-level interface for the RAG pipeline.
    
    Provides simplified access to chatbot functionality.
    """
    
    def __init__(self):
        """Initialize RAG pipeline."""
        self.chatbot = RAGChatbot()
        logger.info("RAG Pipeline initialized")
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a user question through the RAG pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Response with answer and sources
        """
        return self.chatbot.chat(question)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'vector_store': self.chatbot.check_vector_store_status(),
            'llm_model': self.chatbot.config.get('llm.model_name'),
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test RAG chatbot
    pipeline = RAGPipeline()
    
    # Check status
    status = pipeline.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    
    # Example question (if documents exist)
    response = pipeline.process_question("What is the Indian Penal Code?")
    print(f"\nResponse: {json.dumps(response, indent=2)}")
