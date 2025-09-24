# Multi-Modal RAG API

A comprehensive Retrieval Augmented Generation (RAG) API that supports multiple file formats including PDF, DOC/DOCX, XLSX, PPT/PPTX, TXT, video, audio, and images. The system is organized around repositories, allowing users to manage their documents in separate collections with advanced AI-powered search and chat capabilities.

## 🚀 Overview

This Multi-Modal RAG API provides a powerful solution for document processing, intelligent search, and conversational AI. It combines state-of-the-art language models with advanced document processing capabilities to deliver accurate and contextually relevant responses across various file types and formats.

### Key Highlights

- **Multi-Format Support**: Process PDFs, Word docs, Excel files, PowerPoint presentations, text files, videos, audio, and images
- **Intelligent Processing**: OCR for images and PDFs, transcription for audio/video, and smart text extraction
- **Repository Management**: Organize documents in separate collections with access control
- **Advanced Search**: Semantic search using embeddings with metadata filtering
- **Conversational AI**: Chat interface with authentication and context-aware responses
- **Scalable Architecture**: Built with FastAPI and designed for production deployment

## Features

- Process and extract text from multiple file types:
  - PDF documents (with OCR capabilities)
  - Word documents (DOC/DOCX)
  - Excel spreadsheets (XLSX)
  - PowerPoint presentations (PPT/PPTX)
  - Plain text files (TXT)
  - Video files (with audio transcription)
  - Audio files (transcription)
  - Image files (OCR)
- Repository-based document organization
  - Create and manage multiple repositories
  - Upload files to specific repositories
  - Access control per repository
- Intelligent chunking and vector storage
- Semantic search using embeddings
- Natural language query interface
- Document and metadata management
- Support for metadata filtering
- Query across multiple repositories

## Architecture

The system consists of the following components:

1. **Repository Manager**: Handles the creation and management of repositories
2. **Document Processor**: Handles the extraction of text from various file types
3. **Vector Store**: Stores document chunks as embeddings for retrieval
4. **Retriever**: Manages semantic search and retrieval of relevant information
5. **Database**: Handles document metadata and status tracking
6. **API Layer**: FastAPI-based REST interface for client interaction

## 📋 Requirements

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large documents)
- **Storage**: At least 10GB free space for models and processed files
- **GPU**: CUDA-compatible GPU recommended for optimal performance (optional)

### System Dependencies
- **FFmpeg**: For video and audio processing
- **Tesseract OCR**: For image and PDF text extraction
- **CUDA Toolkit**: For GPU acceleration (optional)

## 🛠️ Installation

### Prerequisites

1. **Install Python 3.9+**
   ```bash
   # Check Python version
   python --version
   ```

2. **Install FFmpeg**
   
   **Windows:**
   ```bash
   # Using Chocolatey
   choco install ffmpeg
   
   # Or download from https://ffmpeg.org/download.html
   ```
   
   **macOS:**
   ```bash
   # Using Homebrew
   brew install ffmpeg
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

3. **Install Tesseract OCR**
   
   **Windows:**
   - Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add to PATH environment variable
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt install tesseract-ocr
   ```

### Project Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rag-api.git
   cd rag-api
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Create environment configuration:**
   ```bash
   cp .env.example .env
   ```

5. **Configure environment variables:**
   Edit the `.env` file with your settings:
   ```env
   # API Keys (Required)
   SERPAPI_KEY=your_serp_api_key_here
   SCRAPFLY_KEY=your_scrapfly_key_here
   
   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   
   # File Storage Paths
   UPLOAD_FOLDER=./uploads
   PROCESSED_FOLDER=./processed
   
   # Optional: OpenAI API Key (if using OpenAI models)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: Database Configuration
   DATABASE_URL=sqlite:///./rag_api.db
   ```

6. **Create necessary directories:**
   ```bash
   mkdir -p uploads processed
   ```

7. **Verify installation:**
   ```bash
   python -c "import fastapi, langchain, torch; print('Installation successful!')"
   ```

## 🚀 Quick Start

1. **Start the API server:**
   ```bash
   python app.py
   ```

2. **Access the API:**
   - API Base URL: `http://localhost:8000`
   - Interactive API Documentation: `http://localhost:8000/docs`
   - Alternative Documentation: `http://localhost:8000/redoc`

3. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
The API uses header-based authentication. Include the `X-User-ID` header in all requests:
```http
X-User-ID: your_user_id_here
```

### Core Endpoints

#### Repository Management

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/repositories` | Create a new repository | `name`, `description`, `metadata` |
| `GET` | `/repositories` | List all repositories for a user | `X-User-ID` header |
| `GET` | `/repositories/{repository_id}` | Get a specific repository | `repository_id` |
| `PUT` | `/repositories/{repository_id}` | Update a repository | `repository_id`, update data |
| `DELETE` | `/repositories/{repository_id}` | Delete a repository and all its documents | `repository_id` |

#### Document Management

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/repositories/{repository_id}/documents/upload` | Upload a document to a repository | `file`, `metadata` |
| `GET` | `/repositories/{repository_id}/documents` | List all documents in a repository | `repository_id` |
| `GET` | `/documents` | List all documents (with optional filtering) | `repository_id`, `status` |
| `GET` | `/documents/{document_id}` | Get information about a specific document | `document_id` |
| `DELETE` | `/documents/{document_id}` | Delete a document and its chunks | `document_id` |

#### Chat & AI Features

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `POST` | `/auth-chat` | Authentication chat interface | `messages`, `sampling_params` |
| `POST` | `/chat` | General chat with RAG capabilities | `messages`, `k`, `repository_ids`, `document_ids`, `metadata_filter` |

#### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API health check |
| `GET` | `/status` | System status and metrics |

## Examples

### Create a Repository

```python
import requests
import json

url = "http://localhost:8000/repositories"
headers = {"X-User-ID": "user123"}
data = {
    "name": "Research Papers",
    "description": "Collection of research papers",
    "metadata": {"category": "research"}
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

### Upload a Document to a Repository

```python
import requests
import json

url = "http://localhost:8000/repositories/repo_id_here/documents/upload"
headers = {"X-User-ID": "user123"}
files = {"file": open("document.pdf", "rb")}
metadata = {"source": "research", "author": "John Doe"}

response = requests.post(
    url,
    headers=headers,
    files=files,
    data={"metadata": json.dumps(metadata)}
)

print(response.json())

```

### Chat with users for authentication

```python
import requests
import json

url = "http://localhost:8000/auth-chat/"

messages = [
   {"role": "user", "content": "Hello, world!"},
   {"role": "assistant", "content": "Hi, would you like sing in or sing up?"},
   {"role": "user", "content": "sign up"}
]

headers = {"Content-Type": "application/json"}
payload = {
   "request": {
         "messages": messages,
         "sampling_params": None,
         "chat_template_content_format": None
   }
}

response = requests.post(
    url,
    json=payload, 
    headers=headers
)

print(response.json())

```

### Chat with users for general chat

```python
import requests
import json

url = "http://localhost:8000/chat/"

messages = [
   {"role": "user", "content": "Hello!"},
   {"role": "assistant", "content": "Hi, How can I assist today?"},
   {"role": "user", "content": "Tell me the history of Italy."}
]

headers = {
   "X-User-ID": "user123",
   "Content-Type": "application/json"
}
payload = {
   "messages": chat_history,
   "k": 3,
   "repository_ids": [],
   "document_ids": None,
   "metadata_filter": None
}

response = requests.post(
    url,
    json=payload, 
    headers=headers
)

print(response.json())

```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SERPAPI_KEY` | Yes | - | SerpAPI key for web search functionality |
| `SCRAPFLY_KEY` | Yes | - | Scrapfly API key for web scraping |
| `HOST` | No | `0.0.0.0` | Server host address |
| `PORT` | No | `8000` | Server port number |
| `UPLOAD_FOLDER` | No | `./uploads` | Directory for uploaded files |
| `PROCESSED_FOLDER` | No | `./processed` | Directory for processed files |
| `OPENAI_API_KEY` | No | - | OpenAI API key for OpenAI models |
| `DATABASE_URL` | No | `sqlite:///./rag_api.db` | Database connection string |

### Model Configuration

The system supports multiple language models:

- **VLLM Models**: High-performance local inference
- **OpenAI Models**: Cloud-based API access
- **Custom Models**: Configurable model endpoints

### File Processing Settings

- **Max File Size**: 100MB (configurable)
- **Supported Formats**: PDF, DOC/DOCX, XLSX, PPT/PPTX, TXT, MP4, MP3, WAV, PNG, JPG, JPEG
- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)

### Performance Tuning

For optimal performance, consider these settings:

```env
# GPU Settings (if using CUDA)
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# Memory Settings
MAX_MEMORY_USAGE=0.8
BATCH_SIZE=32

# Processing Settings
MAX_WORKERS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🔐 Authentication

The API uses a simple header-based authentication mechanism where each request must include an `X-User-ID` header with the user's unique identifier. 

### Security Considerations

- **Development**: Current implementation is suitable for development and testing
- **Production**: Replace with secure authentication methods:
  - OAuth 2.0 with JWT tokens
  - API key authentication
  - Session-based authentication
  - Multi-factor authentication

### Example Authentication Headers

```http
X-User-ID: user_12345
Content-Type: application/json
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Installation Problems

**Problem**: `ModuleNotFoundError` when starting the application
```bash
# Solution: Ensure virtual environment is activated and dependencies are installed
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Problem**: FFmpeg not found
```bash
# Solution: Install FFmpeg and ensure it's in PATH
# Windows: Download from https://ffmpeg.org/download.html
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

**Problem**: Tesseract OCR not found
```bash
# Solution: Install Tesseract and ensure it's in PATH
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract
# Linux: sudo apt install tesseract-ocr
```

#### 2. Runtime Issues

**Problem**: CUDA out of memory
```bash
# Solution: Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
# Or reduce memory usage in .env file
```

**Problem**: File upload fails
- Check file size (max 100MB)
- Verify file format is supported
- Ensure upload directory exists and has write permissions

**Problem**: Document processing fails
- Check if Tesseract is properly installed
- Verify FFmpeg installation for video/audio files
- Check available disk space

#### 3. API Issues

**Problem**: 401 Unauthorized
```bash
# Solution: Include X-User-ID header in requests
curl -H "X-User-ID: your_user_id" http://localhost:8000/repositories
```

**Problem**: 500 Internal Server Error
- Check server logs for detailed error messages
- Verify all environment variables are set
- Ensure all dependencies are installed

### Performance Optimization

1. **Memory Usage**
   - Use GPU acceleration when available
   - Adjust batch sizes based on available memory
   - Monitor memory usage during processing

2. **Processing Speed**
   - Use SSD storage for better I/O performance
   - Increase worker processes for parallel processing
   - Cache frequently accessed models

3. **Storage Management**
   - Regularly clean up processed files
   - Use compression for stored embeddings
   - Monitor disk space usage

### Logging and Debugging

Enable detailed logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs for:
- File processing errors
- Model loading issues
- API request/response details
- Memory usage warnings

## ❓ Frequently Asked Questions

### Q: What file formats are supported?
A: The API supports PDF, DOC/DOCX, XLSX, PPT/PPTX, TXT, MP4, MP3, WAV, PNG, JPG, and JPEG files.

### Q: How much memory do I need?
A: Minimum 8GB RAM is required, but 16GB+ is recommended for processing large documents and running language models.

### Q: Can I use this without a GPU?
A: Yes, the system can run on CPU, but GPU acceleration significantly improves performance for language model inference.

### Q: How do I add support for new file formats?
A: Extend the document processing module by adding new processors for your desired format.

### Q: Is this production-ready?
A: The core functionality is stable, but consider implementing proper authentication, monitoring, and scaling for production use.

### Q: How do I scale this for multiple users?
A: Consider using a load balancer, database clustering, and implementing proper user authentication and authorization.

## 🛠️ Development

### Setting up Development Environment

1. **Clone and setup:**
   ```bash
   git clone https://github.com/yourusername/rag-api.git
   cd rag-api
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install development dependencies:**
   ```bash
   pip install pytest black flake8 mypy
   ```

3. **Run tests:**
   ```bash
   pytest tests/
   ```

4. **Code formatting:**
   ```bash
   black .
   flake8 .
   mypy .
   ```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write comprehensive docstrings
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/rag-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-api/discussions)
- **Email**: support@yourdomain.com

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the retrieval framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [VLLM](https://github.com/vllm-project/vllm) for high-performance language model inference
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Transformers](https://github.com/huggingface/transformers) for model support

## ▶️ Running Guide (Step-by-Step)

Follow these steps to run the Multi-Modal RAG API locally.

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Configure environment variables

- Copy the sample file and fill your values:

```bash
cp env.sample .env
```

- Required keys to review in `.env`:
  - `SERPAPI_KEY` and `SCRAPFLY_KEY` (if you use web search/scraping)
  - `HOST`, `PORT` (server bind and port)
  - `UPLOAD_FOLDER`, `PROCESSED_FOLDER`
  - Optional: `EMBEDDING_MODEL_PATH` (defaults to `./model`) for the embedding model used by `rag/vector_store.py`

3) Start the vLLM server

- Linux/macOS:

```bash
chmod +x start_vllm.sh
./start_vllm.sh
```

- Windows (PowerShell via WSL or Git Bash recommended):

```powershell
# In Git Bash / WSL
chmod +x start_vllm.sh && ./start_vllm.sh
```

Notes:
- If you see errors like "invalid choice: 'string\r'", convert line endings to Unix style:
  - `sed -i 's/\r$//' start_vllm.sh`
- Ensure the model name in `start_vllm.sh` exists and you have GPU drivers if using CUDA.

4) Start the API server

```bash
python app.py
```

The API will be available at:
- Base URL: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### Optional configuration

- Embedding model path
  - `rag/vector_store.py` reads `EMBEDDING_MODEL_PATH` from the environment. To override:
    - Linux/macOS: `export EMBEDDING_MODEL_PATH=/path/to/model`
    - Windows (PowerShell): `$env:EMBEDDING_MODEL_PATH = "D:\\path\\to\\model"`

