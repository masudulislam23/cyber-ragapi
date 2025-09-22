# Multi-Modal RAG API: System & Flow Documentation

## Overview

This API implements a **Retrieval Augmented Generation (RAG)** system that supports multi-modal document ingestion (PDF, DOCX, XLSX, PPTX, TXT, video, audio, images), semantic search, and advanced LLM-based chat. It integrates custom embedding models, real-time web search (SerpAPI), and web scraping (Scrapfly) to provide up-to-date, context-rich answers.

---

## System Architecture

```
flowchart TD
    A["User/API Client"] -->|"Upload/Query"| B["FastAPI Server"]
    B -->|"Document Upload"| C["Document Processor"]
    C -->|"Chunking & Embedding"| D["Vector Store"]
    B -->|"Query"| E["Retriever"]
    E -->|"Semantic Search"| D
    E -->|"RAG Context"| F["LLM (vLLM)"]
    F -->|"Needs Web Data?"| G{"Web Search Trigger"}
    G -- Yes --> H["SerpAPI"]
    H --> I["Scrapfly (Web Scraping)"]
    I -->|"Scraped Content"| F
    F -->|"Response"| A
```

---

## Core Components & Flow

### 1. Document Ingestion & Processing

- **Upload**: Users upload documents via `/repositories/{repository_id}/documents/upload/` or chunked upload endpoints.
- **Processing**: The `Document Processor`:
  - Detects file type (PDF, DOCX, XLSX, PPTX, TXT, video, audio, image).
  - Extracts text (OCR for images/PDFs, Whisper for audio/video, etc.).
  - Splits text into context-aware chunks using `RecursiveCharacterTextSplitter`.
  - Each chunk is assigned metadata (document ID, repository, user, etc.).

### 2. Embedding & Vector Storage

- **Embedding Model**: Uses a custom wrapper (`PerceiverEmbeddingWrapper`) around a Perceiver IO model (with fallback to SentenceTransformer) to generate dense vector embeddings for each chunk.
- **Vector Store**: Chunks and their embeddings are stored in a persistent instance, enabling fast semantic search.

### 3. Retrieval Augmented Generation (RAG) Flow

- **Query**: User sends a natural language query to `/chat/`.
- **Retriever**:
  - Embeds the query using the same embedding model.
  - Performs a semantic search in the vector store to retrieve the most relevant document chunks.
- **LLM Context Construction**: The retrieved context is formatted and sent to the LLM as part of the prompt.

### 4. LLM (Large Language Model) Integration

- **Model**: Uses a vision-language model (Pixtral-12B or similar) via the `vLLM` library, with LoRA fine-tuning and quantization for efficiency.
- **Prompting**: The LLM receives:
  - System instructions (response guidelines, current date/time, etc.).
  - Retrieved document context.
  - User's query.
- **Response**: The LLM generates a grounded, context-aware answer.

### 5. Real-Time Web Search & Scraping

- **Web Search Trigger**: The system uses a GPT-based classifier (`needs_web_search_with_gpt`) to decide if a query requires real-time data (e.g., "latest news", "current price").
- **SerpAPI**: If triggered, the system queries Google via SerpAPI for top search results.
- **Scrapfly**: For each result, Scrapfly is used to:
  - Scrape the main article/content.
  - Extract tables (as Markdown), images (with OCR), and clean text.
- **Augmented Context**: Scraped content is appended to the LLM prompt, ensuring the answer is up-to-date and grounded in real web data.

---

## Technology Stack

- **FastAPI**: REST API server.
- **vLLM**: Efficient LLM inference (Pixtral-12B, LoRA, quantized).
- **Perceiver IO / SentenceTransformer**: Embedding models for semantic search.
- **SerpAPI**: Google Search API for real-time web results.
- **Scrapfly**: Web scraping and content extraction.
- **LangChain**: Chunking, document management, and retrieval utilities.
- **Whisper**: Audio/video transcription.
- **Tesseract OCR**: Image and PDF text extraction.

---

## Environment & Configuration

- **.env** file (see `env.sample`):
  - `SERPAPI_KEY`: Your SerpAPI key.
  - `SCRAPFLY_KEY`: Your Scrapfly key.
  - `UPLOAD_FOLDER`, `PROCESSED_FOLDER`: File storage paths.
  - Other server and storage settings.

---

## Example Flow: End-to-End

1. **User uploads a PDF** to a repository.
2. **Document Processor** extracts text, splits into chunks, and stores embeddings.
3. **User asks a question** related to the document.
4. **Retriever** finds the most relevant chunks.
5. **LLM** receives the context and query, determines if real-time data is needed.
6. If needed, **SerpAPI** fetches search results, and **Scrapfly** scrapes the top pages.
7. **LLM** generates a final answer, grounded in both internal documents and fresh web data.

---

## Extending & Customizing

- **Add new file types**: Extend `Document Processor` with new extractors.
- **Swap embedding models**: Update `PerceiverEmbeddingWrapper` or use a different HuggingFace model.
- **Change LLM**: Adjust model loading in `app.py` and `rag/llm.py`.
- **Tune web search triggers**: Edit prompt logic in `rag/llm.py`.
- **Enhance scraping**: Add more extraction logic in `rag/scraper.py`.

---

## Security & Best Practices

- **API keys**: Never commit `.env` with real keys.
- **Resource limits**: Enforce file size and processing timeouts.
- **Data privacy**: Uploaded files and processed data are stored locally; ensure proper access controls.

---

## References

- [FastAPI](https://fastapi.tiangolo.com/)
- [vLLM](https://github.com/vllm-project/vllm)
- [SerpAPI](https://serpapi.com/)
- [Scrapfly](https://scrapfly.io/)
- [LangChain](https://python.langchain.com/)
- [Whisper](https://github.com/openai/whisper)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
