# ⚖️ Legal AI Assistant

A modern, production-ready Retrieval-Augmented Generation (RAG) chatbot designed specifically for Indian legal document analysis and Q&A. Built with LangChain, LangGraph, Google Gemini, ChromaDB, and Streamlit.

## 🌟 Features

### Core Capabilities
- **RAG-powered Legal Q&A**: Ask questions about uploaded legal documents and get accurate, contextual answers
- **Multi-document Support**: Upload and process multiple PDF documents simultaneously
- **Citation Tracking**: Every answer includes source references from the original documents
- **Document Chunking**: Intelligent text splitting for optimal retrieval
- **Semantic Search**: Uses Hugging Face embeddings for accurate document retrieval

### User Management
- **User Authentication**: Secure login and registration system with bcrypt password hashing
- **Conversation History**: Save and retrieve past conversations
- **Multi-user Support**: Isolated data for each authenticated user
- **Session Management**: Persistent user sessions with SQLite backend

### Technical Features
- **Vector Store Management**: ChromaDB for efficient similarity search
- **LangGraph Workflow**: Orchestrated RAG pipeline with clear processing steps
- **LLM Integration**: Google Gemini 2.5 Flash for fast, accurate responses
- **Configurable Settings**: YAML-based configuration for easy customization
- **Logging & Monitoring**: Comprehensive logging with rotating file handlers
- **Error Handling**: Graceful error handling with retry mechanisms

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit UI                           │
│        (Authentication, Chat, Document Upload)         │
└────────────────────┬────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼────┐   ┌─────▼──────┐  ┌────▼────────┐
│ Auth DB  │   │ Document   │  │ Chat        │
│(SQLite)  │   │ Ingestion  │  │ Database    │
└──────────┘   │            │  │ (SQLite)    │
               │  ┌──────┐  │  └─────────────┘
               │  │ PDF  │  │
               │  │Loader│  │
               └──┼──────┼──┘
                  │      │
          ┌───────▼──────▼────────┐
          │   RAG Pipeline         │
          │  ┌─────────────────┐  │
          │  │ LangGraph Flow  │  │
          │  ├─────────────────┤  │
          │  │ 1. Retrieve     │  │
          │  │ 2. Generate     │  │
          │  │ 3. Format       │  │
          │  └─────────────────┘  │
          └───────┬──────┬────────┘
                  │      │
         ┌────────▼┐   ┌─▼──────────┐
         │ ChromaDB│   │Google      │
         │(Vectors)│   │Gemini LLM  │
         └─────────┘   └────────────┘
```

## 📋 Requirements

- Python 3.9+
- Google Generative AI API Key
- 2GB RAM minimum
- 500MB disk space (excluding documents)

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/Rahul0448l/legal-ai-assistant.git
cd legal-ai-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Initialize Directories
```bash
python -c "from utils import DirectoryManager; DirectoryManager.create_required_directories()"
```

## 🎯 Usage

### Start the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Register/Login**
   - Create a new account or login with existing credentials
   - All data is securely stored and isolated per user

2. **Upload Documents**
   - Use the sidebar to upload PDF documents
   - Click "Process Documents" to ingest them
   - Monitor the vector store status

3. **Ask Questions**
   - Type your legal question in the chat interface
   - The system retrieves relevant document sections
   - Google Gemini generates an answer with source citations
   - View sources by clicking the "Sources" expander

4. **Manage History**
   - View past conversations in the sidebar
   - Search through previous questions and answers
   - Create new conversations as needed

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Document Processing
```yaml
document_processing:
  chunk_size: 1000          # Size of text chunks
  chunk_overlap: 200        # Overlap between chunks
  batch_size: 32            # Batch processing size
```

### Embeddings
```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dim: 384
```

### LLM Settings
```yaml
llm:
  model_name: "gemini-2.5-flash"
  temperature: 0.3          # Lower = more deterministic
  max_tokens: 2048
```

### Vector Store
```yaml
vector_store:
  persist_directory: "./chroma_db"
  collection_name: "legal_documents"
  top_k: 5                  # Number of results to retrieve
```

## 📁 Project Structure

```
legal-ai-assistant/
├── app.py                 # Main Streamlit application
├── rag_chatbot.py         # RAG pipeline with LangGraph
├── ingestion.py           # Document processing and embeddings
├── auth_db.py             # User authentication
├── db.py                  # Chat history management
├── utils.py               # Utility functions
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── .env.example            # Environment template
│
├── database/              # SQLite databases
│   ├── users.db
│   └── chats.db
│
├── chroma_db/             # Vector store (ChromaDB)
├── data/                  # Uploaded documents
├── logs/                  # Application logs
└── README.md              # This file
```

## 🔐 Security Features

- **Password Hashing**: Bcrypt with 12 rounds for secure password storage
- **Input Sanitization**: Validates and sanitizes all user inputs
- **SQL Injection Prevention**: Parameterized queries throughout
- **API Key Protection**: Environment variables for sensitive credentials
- **User Isolation**: Separate data storage per authenticated user

## 🐛 Troubleshooting

### API Key Error
```
Error: Could not authenticate with Google Gemini API
```
**Solution**: Ensure `GOOGLE_API_KEY` is set in `.env` file

### Vector Store Empty
```
Warning: No documents in vector store
```
**Solution**: Upload and process PDF documents using the sidebar

### PDF Processing Failed
```
Error: Error loading PDF filename.pdf
```
**Solution**: Ensure PDF is valid and not corrupted

### Out of Memory
```
Error: CUDA out of memory
```
**Solution**: Reduce `batch_size` in config.yaml or process fewer documents

### Database Lock
```
Error: database is locked
```
**Solution**: Ensure no other instances are accessing the database

## 📊 Performance Tips

1. **Chunk Optimization**: Adjust `chunk_size` based on document type
2. **Batch Processing**: Increase `batch_size` for faster document ingestion
3. **Model Selection**: Use smaller embedding models for speed
4. **Vector Search**: Adjust `top_k` to balance relevance vs latency

## 🔄 API Integration

### Process Question
```python
from rag_chatbot import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.process_question("What is IPC Section 302?")

# Response structure
{
    'answer': 'The answer text...',
    'sources': [
        {
            'index': 1,
            'document': 'filename.pdf',
            'page': 5,
            'similarity_score': 0.85,
            'chunk_id': 0
        }
    ],
    'success': True,
    'error': None,
    'timestamp': '2024-01-01T12:00:00',
    'context_count': 1
}
```

### Document Ingestion
```python
from ingestion import DocumentManager

manager = DocumentManager()
success, message, stats = manager.ingest_pdf('path/to/document.pdf')

# stats contains collection information
```

### User Management
```python
from auth_db import AuthDatabase

auth = AuthDatabase()
success, message = auth.register_user('username', 'email@example.com', 'password')
success, message, user_id = auth.login_user('username', 'password')
```

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Google Gemini API](https://ai.google.dev/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙋 Support

For issues, questions, or suggestions:

1. Check existing GitHub Issues
2. Create a new Issue with detailed description
3. Include error messages and logs
4. Provide steps to reproduce

## 👨‍💻 Authors

- **Legal AI Team** - Initial development

## 🙏 Acknowledgments

- LangChain for the excellent framework
- Google for Gemini API
- Hugging Face for embeddings models
- Chroma for vector database
- Streamlit for the amazing UI framework

## 📞 Contact

For more information, visit: [GitHub Repository](https://github.com/Rahul0448l/legal-ai-assistant)

---

**Made with ⚖️ and ❤️ for the legal community**
