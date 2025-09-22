from typing import Dict, Any, List, Optional
import os
from langchain.prompts import PromptTemplate
from rag.vector_store import retrieve_relevant_chunks
from rag.models import QueryResponse, RetrievedChunk, ChatMessage
# Import the vLLM model instance from app.py
from vllm.sampling_params import SamplingParams
from rag.config import (
    MAX_CONTEXT_LENGTH,
    MIN_RELEVANCE_SCORE,
    MAX_CHUNKS_TO_INCLUDE,
    DEFAULT_K
)
from rag.vector_store import embedding_model  # Use the same embedding model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

async def query_documents(
    query: str,
    k: int = DEFAULT_K,  # Use configuration default
    document_ids: Optional[List[str]] = None,
    repository_ids: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> QueryResponse:
    """
    Query documents and generate an answer based on the retrieved chunks.
    
    Args:
        query: Query string
        k: Number of chunks to retrieve
        document_ids: Optional list of document IDs to filter by
        repository_ids: Optional list of repository IDs to filter by
        user_id: Optional user ID to filter by
        metadata_filter: Optional metadata filter
        
    Returns:
        QueryResponse with retrieved chunks and generated answer
    """
    # Retrieve relevant chunks
    step_start = time.time()
    chunks = await retrieve_relevant_chunks(query, k, document_ids, repository_ids, user_id, metadata_filter)
    logger.info(f"[TIMER] trieval: {time.time() - step_start:.3f}s")

    # Filter by relevance_score instead of semantic similarity
    filtered_chunks = [chunk for chunk in chunks if chunk.get('relevance_score', 0) >= MIN_RELEVANCE_SCORE]
    filtered_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    logger.info(f"Relevance filter: {len(filtered_chunks)} chunks above threshold {MIN_RELEVANCE_SCORE}")
    chunks = filtered_chunks
    
    # Convert to RetrievedChunk objects
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            repository_id=chunk["repository_id"],
            content=chunk["content"],
            metadata=chunk["metadata"],
            relevance_score=chunk["relevance_score"]
        )
        for chunk in chunks
    ]
    
    # Generate answer if chunks were 
    context = ""
    if chunks:
        # Prepare context from chunks
        context = "\n\n".join([f"Document {i+1}:\n{chunk['content']}" for i, chunk in enumerate(chunks)])
    
    # Create and return response
    response = QueryResponse(
        query=query,
        chunks=retrieved_chunks,
        context=context
    )
    
    return response
    
def ensure_llm_message_format(messages):
    formatted = []
    for m in messages:
        role = m['role'] if isinstance(m, dict) else getattr(m, 'role', None)
        content = m['content'] if isinstance(m, dict) else getattr(m, 'content', None)
        # Flatten content if it's a list
        if isinstance(content, list):
            content = ' '.join(str(x) for x in content)
        elif not isinstance(content, str):
            content = str(content)
        if role and content is not None:
            formatted.append({'role': role, 'content': content})
    return formatted