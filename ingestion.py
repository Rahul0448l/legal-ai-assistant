"""
Document ingestion and processing module.

This module handles PDF loading, text extraction, chunking,
and embedding generation for the Legal AI Assistant.

Author: Legal AI Team
Date: 2024
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils import ConfigManager, retry_on_exception, validate_file_path

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes PDF documents for ingestion into the vector store.
    
    Handles:
    - PDF loading
    - Text extraction
    - Document chunking
    - Metadata extraction
    """
    
    def __init__(self):
        """Initialize DocumentProcessor with configuration."""
        self.config = ConfigManager()
        self.chunk_size = self.config.get('document_processing.chunk_size', 1000)
        self.chunk_overlap = self.config.get('document_processing.chunk_overlap', 200)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info("DocumentProcessor initialized")
    
    @retry_on_exception(max_retries=3, delay=1.0)
    def load_pdf(self, file_path: str) -> List[Dict]:
        """
        Load a PDF file and extract text.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of document chunks with metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF loading fails
        """
        if not validate_file_path(file_path, '.pdf'):
            raise FileNotFoundError(f"Invalid PDF file: {file_path}")
        
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents from PDF loader
            
        Returns:
            List of chunked documents with metadata
        """
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks...")
            
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_count'] = len(chunks)
            
            logger.info(f"Created {len(chunks)} chunks from documents")
            return chunks
        
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    def process_pdf(self, file_path: str) -> Tuple[List[Dict], bool, str]:
        """
        Complete PDF processing pipeline.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (chunks: List, success: bool, message: str)
        """
        try:
            documents = self.load_pdf(file_path)
            chunks = self.split_documents(documents)
            
            message = f"Successfully processed {file_path}: {len(chunks)} chunks created"
            logger.info(message)
            
            return chunks, True, message
        
        except Exception as e:
            error_msg = f"PDF processing failed for {file_path}: {str(e)}"
            logger.error(error_msg)
            return [], False, error_msg


class EmbeddingManager:
    """
    Manages embeddings generation using Hugging Face models.
    
    Uses sentence-transformers for efficient embeddings.
    """
    
    _instance: Optional['EmbeddingManager'] = None
    _embeddings = None
    
    def __new__(cls) -> 'EmbeddingManager':
        """Implement singleton pattern for EmbeddingManager."""
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize EmbeddingManager."""
        if EmbeddingManager._embeddings is None:
            self._load_embeddings()
    
    def _load_embeddings(self) -> None:
        """Load embedding model from Hugging Face."""
        try:
            config = ConfigManager()
            model_name = config.get('embeddings.model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            
            logger.info(f"Loading embeddings model: {model_name}")
            
            EmbeddingManager._embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )
            
            logger.info("Embeddings model loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def get_embeddings(self):
        """Get embeddings object."""
        return EmbeddingManager._embeddings


class VectorStore:
    """
    Manages ChromaDB vector store operations.
    
    Handles:
    - Vector store initialization
    - Document insertion
    - Similarity search
    - Collection management
    """
    
    _instance: Optional['VectorStore'] = None
    _vector_store = None
    
    def __new__(cls) -> 'VectorStore':
        """Implement singleton pattern for VectorStore."""
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize VectorStore."""
        if VectorStore._vector_store is None:
            self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> None:
        """Initialize ChromaDB vector store."""
        try:
            config = ConfigManager()
            persist_dir = config.get('vector_store.persist_directory', './chroma_db')
            collection_name = config.get('vector_store.collection_name', 'legal_documents')
            
            embeddings = EmbeddingManager().get_embeddings()
            
            logger.info(f"Initializing vector store at {persist_dir}")
            
            VectorStore._vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_dir
            )
            
            logger.info("Vector store initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    @retry_on_exception(max_retries=3, delay=1.0)
    def add_documents(
        self,
        documents: List[Dict],
        batch_size: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of document chunks
            batch_size: Batch size for insertion (optional)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            config = ConfigManager()
            if batch_size is None:
                batch_size = config.get('document_processing.batch_size', 32)
            
            logger.info(f"Adding {len(documents)} documents to vector store...")
            
            # Check for existing documents to avoid duplicates
            existing_ids = set()
            try:
                collection = VectorStore._vector_store._collection
                existing_ids = set(collection.get()['ids']) if collection.get() else set()
            except:
                pass
            
            # Filter out duplicates
            new_documents = [
                doc for doc in documents 
                if doc.metadata.get('source', '') not in str(existing_ids)
            ]
            
            if not new_documents:
                logger.warning("All documents already exist in vector store")
                return True, "Documents already exist in vector store"
            
            # Add in batches
            for i in range(0, len(new_documents), batch_size):
                batch = new_documents[i:i + batch_size]
                VectorStore._vector_store.add_documents(batch)
                logger.info(f"Added batch {i//batch_size + 1}/{(len(new_documents) + batch_size - 1)//batch_size}")
            
            message = f"Successfully added {len(new_documents)} documents to vector store"
            logger.info(message)
            
            return True, message
        
        except Exception as e:
            error_msg = f"Error adding documents to vector store: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            config = ConfigManager()
            if k is None:
                k = config.get('vector_store.top_k', 5)
            
            results = VectorStore._vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                })
            
            logger.info(f"Similarity search found {len(formatted_results)} results")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Collection statistics dictionary
        """
        try:
            collection = VectorStore._vector_store._collection
            ids = collection.get()['ids'] if collection.get() else []
            
            stats = {
                'total_documents': len(ids),
                'collection_name': collection.name if hasattr(collection, 'name') else 'unknown',
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"Collection stats: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> Tuple[bool, str]:
        """
        Clear all documents from collection.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            collection = VectorStore._vector_store._collection
            collection.delete(where={})
            
            message = "Collection cleared successfully"
            logger.info(message)
            return True, message
        
        except Exception as e:
            error_msg = f"Error clearing collection: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def delete_documents_by_source(self, source: str) -> Tuple[bool, str]:
        """
        Delete documents from a specific source.
        
        Args:
            source: Source document name
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            collection = VectorStore._vector_store._collection
            collection.delete(where={"source": {"$eq": source}})
            
            message = f"Documents from {source} deleted successfully"
            logger.info(message)
            return True, message
        
        except Exception as e:
            error_msg = f"Error deleting documents: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


class DocumentManager:
    """
    High-level interface for document management.
    
    Coordinates document processing, embedding, and storage.
    """
    
    def __init__(self):
        """Initialize DocumentManager."""
        self.processor = DocumentProcessor()
        self.embeddings = EmbeddingManager()
        self.vector_store = VectorStore()
    
    def ingest_pdf(self, file_path: str) -> Tuple[bool, str, Dict]:
        """
        Complete PDF ingestion pipeline.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (success: bool, message: str, stats: Dict)
        """
        try:
            logger.info(f"Starting PDF ingestion: {file_path}")
            
            # Process PDF
            chunks, success, message = self.processor.process_pdf(file_path)
            if not success:
                return False, message, {}
            
            # Add to vector store
            success, store_message = self.vector_store.add_documents(chunks)
            if not success:
                return False, store_message, {}
            
            # Get stats
            stats = self.vector_store.get_collection_stats()
            
            final_message = f"PDF ingestion successful: {file_path} ({len(chunks)} chunks)"
            logger.info(final_message)
            
            return True, final_message, stats
        
        except Exception as e:
            error_msg = f"PDF ingestion failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {}
    
    def get_collection_info(self) -> Dict:
        """Get information about the vector store collection."""
        return self.vector_store.get_collection_stats()
    
    def clear_all_documents(self) -> Tuple[bool, str]:
        """Clear all documents from vector store."""
        return self.vector_store.clear_collection()


if __name__ == "__main__":
    # Test document ingestion
    manager = DocumentManager()
    
    # Example usage
    pdf_path = "./data/sample.pdf"
    if Path(pdf_path).exists():
        success, message, stats = manager.ingest_pdf(pdf_path)
        print(f"Success: {success}")
        print(f"Message: {message}")
        print(f"Stats: {stats}")
